"""Same-channel AC WLS via elimination of the full OpenDSS phase state.

This is NOT a voltage-only net-injection WLS. The existing SCADA channels
measure nominal load-setting aggregates. Unknown individual load multipliers
and unobserved regulator taps are inferred, never obtained from target arrays.
Full phase AC equations are solved internally for each nonlinear prediction.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
import opendssdirect


def _complex_pairs(values):
    a = np.asarray(values, dtype=np.float64)
    return a[::2] + 1j * a[1::2]


def circular_difference(a, b):
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


class PhasePhysicsModel:
    """A private DSS context exposing full phase physics and original channels."""
    def __init__(self, dss_dir: Path, output_dir: Path, pmu_names, s_base_mva=1.0):
        self.dss_dir = Path(dss_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.s_base_mva = float(s_base_mva)
        self.dss = opendssdirect.NewContext()
        self.dss.Text.Command('Redirect "' + str(self.dss_dir / 'IEEE123Master.dss') + '"')
        self.dss.Basic.DataPath(str(self.output_dir))
        self.dss.Solution.Solve()
        if not self.dss.Solution.Converged():
            raise RuntimeError('Nominal feeder does not converge')
        self.bus_names = [n.lower() for n in self.dss.Circuit.AllBusNames()]
        self.bus_to_index = {n: i for i, n in enumerate(self.bus_names)}
        self.pmu_names = [str(n).lower() for n in pmu_names]
        self.pmu_indices = np.array([self.bus_to_index[n] for n in self.pmu_names])
        self.node_names = [n.lower() for n in self.dss.Circuit.YNodeOrder()]
        self.node_to_index = {n: i for i, n in enumerate(self.node_names)}
        self.node_base_volts = np.empty(len(self.node_names))
        self.projection = np.zeros((len(self.bus_names), len(self.node_names)), dtype=complex)
        for bi, name in enumerate(self.bus_names):
            self.dss.Circuit.SetActiveBus(name)
            nodes = [int(n) for n in self.dss.Bus.Nodes() if int(n) in (1, 2, 3)]
            base = 1000.0 * self.dss.Bus.kVBase()
            if not nodes or base <= 0:
                raise RuntimeError('Energized bus projection/base missing: ' + name)
            for phase in nodes:
                ni = self.node_to_index[name + '.' + str(phase)]
                self.node_base_volts[ni] = base
                self.projection[bi, ni] = np.exp(2j*np.pi*(phase - 1)/3)/len(nodes)/base
        nominal_phase = _complex_pairs(self.dss.Circuit.YNodeVArray())
        nominal_bus = self.projection @ nominal_phase
        self.asset_nominal_state = dict(phase_voltage=nominal_phase.copy(),
            v=np.abs(nominal_bus),angle=np.angle(nominal_bus))
        self.loads = []
        for name in self.dss.Loads.AllNames():
            self.dss.Loads.Name(name)
            self.dss.Circuit.SetActiveElement('Load.' + name)
            bus = self.dss.CktElement.BusNames()[0].split('.')[0].lower()
            self.loads.append(dict(name=name, bus=bus, bus_index=self.bus_to_index[bus],
                kw=float(self.dss.Loads.kW()), kvar=float(self.dss.Loads.kvar()),
                model=int(self.dss.Loads.Model()), phases=int(self.dss.Loads.Phases()),
                is_delta=bool(self.dss.Loads.IsDelta()),
                terminal_nodes=list(self.dss.CktElement.NodeOrder()),
                vminpu=float(self.dss.Loads.Vminpu()), vmaxpu=float(self.dss.Loads.Vmaxpu())))
        self.regulators = []
        for name in self.dss.Transformers.AllNames():
            if not name.lower().startswith('reg'):
                continue
            self.dss.Transformers.Name(name)
            self.dss.Transformers.Wdg(2)
            self.regulators.append(dict(name=name, winding=2,
                nominal_asset_tap=float(self.dss.Transformers.Tap()),
                min_tap=float(self.dss.Transformers.MinTap()),
                max_tap=float(self.dss.Transformers.MaxTap()),
                num_taps=int(self.dss.Transformers.NumTaps())))
        self.load_matrix = np.zeros((2*len(self.bus_names), len(self.loads)))
        for j, load in enumerate(self.loads):
            self.load_matrix[load['bus_index'], j] = load['kw']/(1000*self.s_base_mva)
            self.load_matrix[len(self.bus_names)+load['bus_index'], j] = load['kvar']/(1000*self.s_base_mva)
        self.nominal_taps = np.array([r['nominal_asset_tap'] for r in self.regulators])
        self.dss.Text.Command('Set ControlMode=OFF')
        self.dss.Text.Command('Set Tolerance=1e-10')
        self.dss.Solution.MaxIterations(100)
        self.forward_count = 0
        self.forward_wall_seconds = 0.0

    @property
    def dimension(self):
        return len(self.loads)+len(self.regulators)

    def physics_predict(self, parameters):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (self.dimension,):
            raise ValueError('Parameter dimension mismatch')
        start = time.perf_counter()
        a = parameters[:len(self.loads)]
        taps = parameters[len(self.loads):]
        for load, scale in zip(self.loads, a):
            self.dss.Loads.Name(load['name'])
            self.dss.Loads.kW(load['kw']*scale)
            self.dss.Loads.kvar(load['kvar']*scale)
        for reg, tap in zip(self.regulators, taps):
            self.dss.Transformers.Name(reg['name'])
            self.dss.Transformers.Wdg(reg['winding'])
            self.dss.Transformers.Tap(float(tap))
        self.dss.Solution.Solve()
        self.forward_count += 1
        self.forward_wall_seconds += time.perf_counter()-start
        if not self.dss.Solution.Converged():
            raise RuntimeError('Phase AC forward solve did not converge')
        phase_voltage = _complex_pairs(self.dss.Circuit.YNodeVArray())
        bus_voltage = self.projection @ phase_voltage
        pmu = np.r_[np.abs(bus_voltage[self.pmu_indices]), np.angle(bus_voltage[self.pmu_indices])]
        observation = np.r_[self.load_matrix @ a, pmu]
        return dict(observation=observation, phase_voltage=phase_voltage,
                    bus_voltage=bus_voltage, v=np.abs(bus_voltage), angle=np.angle(bus_voltage))

    def asset_record(self):
        return dict(bus_names=self.bus_names, node_names=self.node_names,
                    node_base_volts=self.node_base_volts.tolist(), loads=self.loads,
                    regulators=self.regulators, pmu_names=self.pmu_names,
                    full_phase_state_real_dimension=2*len(self.node_names),
                    reduced_unknown_dimension=self.dimension,
                    nominal_load_aggregate_matrix_rank=int(np.linalg.matrix_rank(self.load_matrix)),
                    description='Original nominal load aggregates and bus-aligned PMU projection; no net-injection reinterpretation',
                    dss_version=self.dss.Basic.Version())

    def terminal_kcl_residuals(self):
        """Check physical equality residuals once at the fitted operating point."""
        kcl=np.zeros(len(self.node_names),dtype=complex)
        nonzero_injection=set()
        for ename in self.dss.Circuit.AllElementNames():
            if self.dss.Circuit.SetActiveElement(ename)<0 or not self.dss.CktElement.Enabled():
                continue
            buses=self.dss.CktElement.BusNames();nodes=self.dss.CktElement.NodeOrder()
            currents=_complex_pairs(self.dss.CktElement.Currents())
            nc=self.dss.CktElement.NumConductors()
            for k,(node,current) in enumerate(zip(nodes,currents)):
                name=buses[k//nc].split('.')[0].lower()+'.'+str(node)
                if node and name in self.node_to_index:
                    ni=self.node_to_index[name];kcl[ni]+=current
                    if ename.lower().startswith(('load.','vsource.')):
                        nonzero_injection.add(ni)
        zero_indices=sorted(set(range(len(self.node_names)))-nonzero_injection)
        return dict(phase_ac_forward_converged=bool(self.dss.Solution.Converged()),
            full_phase_kcl_max_abs_amps=float(np.max(np.abs(kcl))),
            zero_injection_phase_count=len(zero_indices),
            zero_injection_phase_kcl_max_abs_amps=float(np.max(np.abs(kcl[zero_indices]))) if zero_indices else 0.0,
            zero_injection_definition='phase nodes without Load or Vsource terminals; shunts remain in network equality')


class SameChannelPhysicalGN:
    """Trust-region damped GN (QR/SVD internals) on whitened observed residuals.

    Predictions consume only raw SCADA/PMU values and masks. The caller supplies
    no labels, trajectory seed, oracle per-phase values, or actual tap position.
    """
    def __init__(self, model: PhasePhysicsModel, *, max_nfev=20, prior_scale=None,
                 prior_mean=None, loss='wls', huber_delta=1.345):
        self.model = model
        self.max_nfev = int(max_nfev)
        self.prior_scale = prior_scale
        self.loss = str(loss)
        self.huber_delta = float(huber_delta)
        if self.loss not in ('wls','huber'):
            raise ValueError('Loss must be wls or huber')
        self.lower = np.r_[np.full(len(model.loads), .55), [r['min_tap'] for r in model.regulators]]
        self.upper = np.r_[np.full(len(model.loads), 1.45), [r['max_tap'] for r in model.regulators]]
        self.nominal = np.r_[np.ones(len(model.loads)), model.nominal_taps]
        self.prior_mean = self.nominal.copy() if prior_mean is None else np.asarray(prior_mean,dtype=float)

    def observed_vector(self, scada, scada_mask, pmu, pmu_mask):
        scada, scada_mask = np.asarray(scada), np.asarray(scada_mask)
        pmu, pmu_mask = np.asarray(pmu), np.asarray(pmu_mask)
        if scada.shape != (len(self.model.bus_names), 3) or pmu.shape != (len(self.model.pmu_names), 2):
            raise ValueError('Raw original measurement shape required')
        z = np.r_[scada[:,0], scada[:,1], pmu[:,0], pmu[:,1]]
        mask = np.r_[scada_mask[:,0], scada_mask[:,1], pmu_mask[:,0], pmu_mask[:,1]] > .5
        # Plug-in variance from observed magnitudes: no hidden clean trajectory.
        sigma = np.r_[np.maximum(np.abs(scada[:,0])*.03, .001),
                      np.maximum(np.abs(scada[:,1])*.03, .001),
                      np.full(len(pmu), .002), np.full(len(pmu), .0005)]
        if not np.isfinite(z[mask]).all() or not np.isfinite(sigma[mask]).all():
            raise ValueError('Non-finite observed channel')
        return z, mask, sigma

    def measurement_initialization(self, z, mask):
        x = self.nominal.copy()
        n = len(self.model.bus_names)
        for bi in range(n):
            load_indices = [j for j,l in enumerate(self.model.loads) if l['bus_index']==bi]
            estimates = []
            for row in (bi, n+bi):
                base = self.model.load_matrix[row].sum()
                if mask[row] and base > 0:
                    estimates.append(z[row]/base)
            if estimates and load_indices:
                x[load_indices] = np.clip(np.mean(estimates), .55+1e-8, 1.45-1e-8)
        return x

    def estimate_snapshot(self, *, scada, scada_mask, pmu, pmu_mask, previous_parameters=None):
        start=time.perf_counter(); calls_start=self.model.forward_count
        try:
            return self._estimate_snapshot(scada=scada,scada_mask=scada_mask,
                pmu=pmu,pmu_mask=pmu_mask,previous_parameters=previous_parameters)
        except Exception as exc:
            # Every case contributes a finite terminal/fallback prediction. The
            # asset-only nominal PF is cached before any trajectory is consumed.
            state=self.model.asset_nominal_state
            return dict(v=state['v'].copy(),angle=state['angle'].copy(),
                phase_voltage=state['phase_voltage'].copy(),parameters=self.nominal.copy(),
                converged=False,status=-100,message=type(exc).__name__+': '+str(exc),
                fallback_used=True,fallback_rule='independent unit-load nominal operating-point asset PF',
                nfev=None,observed_count=int(np.count_nonzero(np.asarray(scada_mask)[:,:2]>.5)+np.count_nonzero(np.asarray(pmu_mask)>.5)),
                reduced_unknown_dimension=self.model.dimension,
                whitened_jacobian_rank=None,practical_rank_relative_1e_minus_6=None,
                jacobian_singular_values=[],jacobian_condition_number=None,
                observability_jacobian_excludes_priors_and_robust_weights=True,
                optimization_wall_seconds=time.perf_counter()-start,diagnostics_wall_seconds=0.0,
                optimization_forward_solves=self.model.forward_count-calls_start,diagnostics_forward_solves=0,
                actual_forward_solves=self.model.forward_count-calls_start,
                wall_seconds=time.perf_counter()-start,terminal_prediction_retained=False,
                diagnostic_failed=False,diagnostic_failure=None,failure_stage='optimization_or_terminal_forward',
                robust_loss=self.loss,prior_enabled=self.prior_scale is not None,
                method='Same-channel phase-physics constrained reduced-coordinate damped GN/'+('MAP-' if self.prior_scale is not None else '')+self.loss.upper())

    def _estimate_snapshot(self, *, scada, scada_mask, pmu, pmu_mask, previous_parameters=None):
        start = time.perf_counter()
        z, mask, sigma = self.observed_vector(scada, scada_mask, pmu, pmu_mask)
        x0 = self.measurement_initialization(z, mask) if previous_parameters is None else np.asarray(previous_parameters).copy()
        x0 = np.clip(x0, self.lower+1e-8, self.upper-1e-8)
        angle_start = 2*len(self.model.bus_names)+len(self.model.pmu_names)
        angular = np.arange(len(z)) >= angle_start
        calls_start = self.model.forward_count
        def residual(parameters):
            prediction = self.model.physics_predict(parameters)['observation']
            error = prediction-z
            error[angular] = circular_difference(prediction[angular], z[angular])
            r = error[mask]/sigma[mask]
            if self.prior_scale is not None:
                # If enabled, prior must be a predeclared asset/train-only prior.
                r = np.r_[r, (parameters-self.prior_mean)/np.asarray(self.prior_scale)]
            return r
        measurement_count = int(mask.sum())
        def huber_measurement_quadratic_prior(squared):
            rho = np.vstack([squared.copy(), np.ones_like(squared), np.zeros_like(squared)])
            delta = self.huber_delta
            large = np.flatnonzero(squared[:measurement_count] > delta*delta)
            root = np.sqrt(squared[large])
            rho[0,large] = 2*delta*root-delta*delta
            rho[1,large] = delta/root
            rho[2,large] = -.5*delta/(squared[large]*root)
            return rho
        fit = least_squares(residual, x0, bounds=(self.lower,self.upper),
                            method='trf', tr_solver='exact', jac='2-point', diff_step=1e-4,
                            loss='linear' if self.loss=='wls' else huber_measurement_quadratic_prior,
                            x_scale=np.r_[np.ones(len(self.model.loads)),np.full(len(self.model.regulators),.01)],
                            max_nfev=self.max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        prediction = self.model.physics_predict(fit.x)
        if not all(np.isfinite(prediction[key]).all() for key in ['v','angle','phase_voltage']):
            raise RuntimeError('Non-finite optimizer terminal phase/bus prediction')
        optimization_wall_seconds=time.perf_counter()-start
        optimization_forward_solves=self.model.forward_count-calls_start
        diagnostic_start=time.perf_counter()
        diagnostic_calls_start=self.model.forward_count
        try:
            diagnostics=self._terminal_diagnostics(fit.x,prediction,mask,sigma,angular,measurement_count)
        except Exception as exc:
            # A failed optional equality/Jacobian check must never change an
            # already finite optimizer terminal prediction or its error metric.
            diagnostics=dict(whitened_jacobian_rank=None,practical_rank_relative_1e_minus_6=None,
                smallest_singular_value=None,largest_singular_value=None,jacobian_singular_values=[],
                jacobian_condition_number=None,jacobian_coordinate_scale=None,
                observability_jacobian_excludes_priors_and_robust_weights=True,phase_equality_residuals=None,
                weak_direction_count=None,weak_direction_voltage_target_operator_norm=None,
                weak_direction_angle_target_operator_norm=None,diagnostic_failed=True,
                diagnostic_failure=type(exc).__name__+': '+str(exc))
        return dict(v=prediction['v'], angle=prediction['angle'], phase_voltage=prediction['phase_voltage'],
                    parameters=fit.x, converged=bool(fit.success), status=int(fit.status), message=str(fit.message),
                    nfev=int(fit.nfev), actual_forward_solves=self.model.forward_count-calls_start,
                    observed_count=int(mask.sum()),reduced_unknown_dimension=self.model.dimension,
                    cost=float(fit.cost), optimality=float(fit.optimality),
                    fallback_used=False,fallback_rule=None,failure_stage=None,
                    optimization_wall_seconds=optimization_wall_seconds,
                    diagnostics_wall_seconds=time.perf_counter()-diagnostic_start,
                    optimization_forward_solves=optimization_forward_solves,
                    diagnostics_forward_solves=self.model.forward_count-diagnostic_calls_start,
                    wall_seconds=time.perf_counter()-start, terminal_prediction_retained=True,
                    robust_loss=self.loss,huber_delta=self.huber_delta if self.loss=='huber' else None,
                    prior_enabled=self.prior_scale is not None,
                    method='Same-channel phase-physics constrained reduced-coordinate damped GN/'+('MAP-' if self.prior_scale is not None else '')+self.loss.upper(),
                    **diagnostics)

    def _terminal_diagnostics(self,parameters,prediction,mask,sigma,angular,measurement_count):
        kcl_residuals=self.model.terminal_kcl_residuals()
        # SciPy modifies its returned Jacobian for robust loss and appends prior
        # rows. Recompute the ORIGINAL measurement-only whitened Jacobian.
        base_observation = prediction['observation']
        measurement_jacobian = np.empty((measurement_count, self.model.dimension))
        target_jacobian = np.empty((2*len(self.model.bus_names), self.model.dimension))
        for j in range(self.model.dimension):
            shifted = parameters.copy()
            step = max(abs(shifted[j])*1e-4,1e-6)
            if shifted[j]+step >= self.upper[j]: step=-step
            shifted[j] += step
            changed = self.model.physics_predict(shifted)
            difference=changed['observation']-base_observation
            difference[angular]=circular_difference(changed['observation'][angular],base_observation[angular])
            measurement_jacobian[:,j]=(difference[mask]/sigma[mask])/step
            target_jacobian[:,j]=np.r_[changed['v']-prediction['v'],
                circular_difference(changed['angle'],prediction['angle'])]/step
        # Report dimensionless-coordinate singular values as well, scaling taps
        # by .01 and loads by 1. No prior/robust row can create observability.
        coordinate_scale=np.r_[np.ones(len(self.model.loads)),np.full(len(self.model.regulators),.01)]
        _, singular, vh=np.linalg.svd(measurement_jacobian*coordinate_scale[None,:],full_matrices=True)
        threshold = np.finfo(float).eps*max(measurement_jacobian.shape)*singular[0] if singular.size else 0
        rank = int(np.sum(singular > threshold))
        practical_rank=int(np.sum(singular > singular[0]*1e-6)) if singular.size else 0
        weak=vh[practical_rank:].T
        weak_target=(target_jacobian*coordinate_scale[None,:])@weak
        weak_v=weak_target[:len(self.model.bus_names)]
        weak_angle=weak_target[len(self.model.bus_names):]
        return dict(whitened_jacobian_rank=rank,
                    smallest_singular_value=float(singular[-1]) if singular.size else None,
                    largest_singular_value=float(singular[0]) if singular.size else None,
                    jacobian_singular_values=singular.tolist(),
                    practical_rank_relative_1e_minus_6=practical_rank,
                    jacobian_coordinate_scale=coordinate_scale.tolist(),
                    observability_jacobian_excludes_priors_and_robust_weights=True,
                    phase_equality_residuals=kcl_residuals,
                    weak_direction_count=int(weak.shape[1]),
                    weak_direction_voltage_target_operator_norm=float(np.linalg.norm(weak_v,2)) if weak_v.size else 0.0,
                    weak_direction_angle_target_operator_norm=float(np.linalg.norm(weak_angle,2)) if weak_angle.size else 0.0,
                    jacobian_condition_number=float(singular[0]/singular[-1]) if singular.size and singular[-1]>0 else None,
                    diagnostic_failed=False,diagnostic_failure=None)
