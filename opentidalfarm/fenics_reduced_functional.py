import os.path
from dolfin import *
from dolfin_adjoint import *
from solvers import Solver
from functionals import TimeIntegrator, PrototypeFunctional
import helpers

__all__ = ["FenicsReducedFunctional"]

class FenicsReducedFunctional(ReducedFunctional):
    """
    Following parameters are expected:

    :ivar functional: a :class:`PrototypeFunctional` class.
    :ivar controls: a (optionally list of) :class:`dolfin_adjoint.DolfinAdjointControl` object.
    :ivar solver: a :class:`Solver` object.

    This class has a parameter attribute for further adjustments.
    """

    def __init__(self, functional, controls, solver):

        self.solver = solver
        if not isinstance(solver, Solver):
            raise ValueError, "solver argument of wrong type."

        self.otf_functional = functional
        if not isinstance(functional, PrototypeFunctional):
            raise ValueError, "invalid functional argument."

        # Hidden attributes
        self._solver_params = solver.parameters
        self._problem_params = solver.problem.parameters
        self._time_integrator = None

        # Controls
        self.controls = enlisting.enlist(controls)
        self.evaluate()
        dolfin_adjoint_functional = self.time_integrator.dolfin_adjoint_functional(self.solver.state)

        super(FenicsReducedFunctional, self).__init__(dolfin_adjoint_functional, controls)

        # Caching variables that store which controls the last forward run was
        # performed
        self.last_m = None
        if self.solver.parameters.dump_period > 0:
            turbine_filename = os.path.join(solver.parameters.output_dir, "turbines.pvd")
            self.turbine_file = File(turbine_filename, "compressed")

    def evaluate(self, annotate=True):
        """ Return the functional value for the given control values. """

        log(INFO, 'Start evaluation of j')
        timer = dolfin.Timer("j evaluation")

        farm = self.solver.problem.parameters.tidal_farm

        # Configure dolfin-adjoint
        adj_reset()
        dolfin.parameters["adjoint"]["record_all"] = True

        # Solve the shallow water system and integrate the functional of
        # interest.
        final_only = (not self.solver.problem._is_transient or
                      self._problem_params.functional_final_time_only)
        self.time_integrator = TimeIntegrator(self.solver.problem,
                                              self.otf_functional, final_only)

        for sol in self.solver.solve(annotate=annotate):
            self.time_integrator.add(sol["time"], sol["state"], sol["tf"],
                                     sol["is_final"])

        j = self.time_integrator.integrate()

        timer.stop()

        log(INFO, 'Runtime: %f s.' % timer.value())
        log(INFO, 'j = %e.' % float(j))

        return j

    def _update_turbine_farm(self):
        """ Update the turbine farm from the flattened parameter array m. """
        farm = self.solver.problem.parameters.tidal_farm

        if farm.turbine_specification.smeared:
            farm._parameters["friction"] = self.controls[0].data().vector()

        #else:
        #    controlled_by = farm.turbine_specification.controls
        #    shift = 0
        #    if controlled_by.friction:
        #        shift = len(farm._parameters["friction"])
        #        farm._parameters["friction"] = m[:shift]
        #    elif controlled_by.dynamic_friction:
        #        shift = len(numpy.reshape(farm._parameters["friction"],-1))
        #        nb_turbines = len(farm._parameters["position"])
        #        farm._parameters["friction"] = (
        #            numpy.reshape(m[:shift], (-1, nb_turbines)).tolist())

        #    if controlled_by.position:
        #        m_pos = m[shift:]
        #        farm._parameters["position"] = (
        #            numpy.reshape(m_pos, (-1,2)).tolist())

        # Update the farm cache.
        farm.update()

    def derivative(self, forget=False, new_optimisation_iteration=True, **kwargs):
        """ Computes the first derivative of the functional with respect to its
        controls by solving the adjoint equations. """
        
        log(INFO, 'Start evaluation of dj')
        timer = dolfin.Timer("dj evaluation") 

        # If any of the parameters changed, the forward model needs to be re-run
        #self.evaluate

        J = self.time_integrator.dolfin_adjoint_functional(self.solver.state)
        #J = self.functional
        dj = compute_gradient(J, self.controls, forget=forget, **kwargs)
        dolfin.parameters["adjoint"]["stop_annotating"] = False

        log(INFO, "Runtime: " + str(timer.stop()) + " s")

        # We assume that the gradient is computed at and only at the beginning
        # of each new optimisation iteration. Hence, this is the right moment
        # to store the turbine friction field and to increment the optimisation
        # iteration counter.
        if new_optimisation_iteration:
            self.solver.optimisation_iteration += 1
            farm = self.solver.problem.parameters.tidal_farm
            self._update_turbine_farm()

            if (self.solver.parameters.dump_period > 0 and
                farm is not None):
                # A cache hit skips the turbine cache update, so we need
                # trigger it manually.
                #if self._compute_gradient_mem.has_cache(m, forget):
                #    self._update_turbine_farm(m)
                #if farm.turbine_specification.controls.dynamic_friction:
                #    log(WARNING, ("Turbine VTU output not yet implemented for "
                #                  " dynamic turbine control"))
                #else:
                    self.turbine_file << farm.turbine_cache['turbine_field']
                    # Compute the total amount of friction due to turbines
                    if farm.turbine_specification.smeared:
                        log(INFO, "Total amount of friction: %f" %
                            assemble(farm.turbine_cache["turbine_field"]*dx))
        return dj 

