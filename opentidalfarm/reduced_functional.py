import sys
import os.path
import numpy
import helpers
import dolfin_adjoint
from dolfin import *
from dolfin_adjoint import *
from solvers import Solver
from functionals import TimeIntegrator, PrototypeFunctional
from memoize import MemoizeMutable
from options import options


class ReducedFunctionalParameters(helpers.FrozenClass):
    """ A set of parameters for a :class:`ReducedFunctional`.

    Following parameters are available:

    :ivar scale: A scaling factor. Default: 1.0
    :ivar automatic_scaling: The reduced functional will be
        automatically scaled such that the maximum absolute value of the initial
        gradient is equal to the specified factor. Set to False to deactivate the
        automatic scaling. Default: 5.
    :ivar load_checkpoints: If True, the checkpoints are loaded from file and
        used. Default: False
    :ivar save_checkpoints: Automatically store checkpoints after each
        optimisation iteration. Default: False
    :ivar checkpoint_basefilename: The base filename (without extensions) for
        storing or loading the checkpoints. Default: 'checkpoints'.
    """

    scale = 1.
    automatic_scaling = 5.
    save_checkpoints = False
    load_checkpoints = False
    checkpoints_basefilename = "checkpoints"


class ReducedFunctional(dolfin_adjoint.ReducedFunctionalNumPy):
    """
    Following parameters are expected:

    :ivar functional: a :class:`PrototypeFunctional` class.
    :ivar solver: a :class:`Solver` object.
    :ivar parameters: a :class:`ReducedFunctionalParameters` object.

    This class has a parameter attribute for further adjustments.
    """

    def __init__(self, functional, solver, parameters):
        # For consistency with the dolfin-adjoint API.
        self.scale = parameters.scale

        self.solver = solver
        if not isinstance(solver, Solver):
            raise ValueError, "solver argument of wrong type."

        self.functional = functional
        if not PrototypeFunctional in functional.__bases__:
            raise ValueError, "invalid functional argument."

        self._farm = solver.problem.parameters.tidal_farm
        if self._farm is None:
            raise ValueError, "The problem does not have a tidal farm."

        # Create the default parameters
        self.parameters = parameters

        # Hidden attributes
        self._solver_params = solver.parameters
        self._problem_params = solver.problem.parameters
        self._time_integrator = None
        self._optimisation_iteration = 0
        self._automatic_scaling_factor = None

        # Caching variables that store which controls the last forward run was
        # performed
        self.last_m = None
        if self.solver.parameters.dump_period > 0:
            turbine_filename = (solver_parameters.output_dir + os.path.sep +
                                "turbines.pvd")
            self.turbine_file = File(turbine_filename, "compressed")

            if self._solver_params.output_turbine_power:
                power_filename = (solver_parameters.output_dir + os.path.sep +
                                  "power.pvd")
                self.power_file = File(power_filename, "compressed")

        # dolfin-adjoint requires the ReducedFunctional to have a member
        # variable `parameter` which must be a list comprising an instance of a
        # class (here, TurbineFarmParameter) which has a method named `data`
        # which returns a numpy.ndarray of the parameters used for optimisation,
        # e.g. the turbine frictions and positions.
        self.parameter = [TurbineFarmParameter(self._farm)]

        # For smeared turbine parametrisations we only want to store the
        # hash of the control values into the pickle datastructure
        use_hash_keys = self._farm.turbine_specification.smeared

        self._compute_functional_mem = MemoizeMutable(self._compute_functional,
                                                      use_hash_keys)
        self._compute_gradient_mem = MemoizeMutable(self._compute_gradient,
                                                    use_hash_keys)

        # Load checkpoints from file
        if self.parameters.load_checkpoints:
            self.load_checkpoints()


    @staticmethod
    def default_parameters():
        """ Return the default parameters for the :class:`ReducedFunctional`.
        """
        return ReducedFunctionalParameters()


    def _compute_gradient(self, m, forget=True):
        """ Compute the functional gradient for the turbine positions/frictions array """

        # If any of the parameters changed, the forward model needs to be re-run
        if numpy.any(m != self.last_m):
            self._compute_functional(m, annotate=True)

        J = self.time_integrator.dolfin_adjoint_functional()

        # Output power
        if self.solver.parameters.dump_period > 0:
            if options["output_individual_power"]:
                turbines = self._farm.turbine_cache["turbine_field"]
                power = self.functional(self._farm).power(solver.current_state,
                                                          turbines)
                self.power_file << project(power,
                                           self._farm.turbine_function_space,
                                           annotate=False)

        if self._farm.turbine_specification.controls.dynamic_friction:
            parameters = []
            for i in xrange(len(self._farm._parameters["friction"])):
                parameters.append(
                    FunctionControl("turbine_friction_cache_t_%i" % i))

        else:
            parameters = FunctionControl("turbine_friction_cache")

        djdtf = dolfin_adjoint.compute_gradient(J, parameters, forget=forget)
        dolfin.parameters["adjoint"]["stop_annotating"] = False

        # Decide if we need to apply the chain rule to get the gradient of
        # interest.
        if self._farm.turbine_specification.smeared:
            # We are looking for the gradient with respect to the friction
            dj = dolfin_adjoint.optimization.get_global(djdtf)

        else:
            # Let J be the functional, m the parameter and u the solution of the
            # PDE equation F(u) = 0.
            # Then we have
            # dJ/dm = (\partial J)/(\partial u) * (d u) / d m + \partial J / \partial m
            #               = adj_state * \partial F / \partial u + \partial J / \partial m
            # In this particular case m = turbine_friction, J = \sum_t(ft)
            dj = []

            if self._farm.turbine_specification.controls.friction:
                # Compute the derivatives with respect to the turbine friction
                for tfd in self._farm.turbine_cache["turbine_derivative_friction"]:
                    self._farm.turbine_cache.update(self._farm)
                    dj.append(djdtf.vector().inner(tfd.vector()))

            elif self._farm.turbine_specification.controls.dynamic_friction:
                # Compute the derivatives with respect to the turbine friction
                for djdtf_arr, t in zip(djdtf, self._farm.turbine_cache["turbine_derivative_friction"]):
                    for tfd in t:
                        self._farm.turbine_cache.update(self._farm)
                        dj.append(djdtf_arr.vector().inner(tfd.vector()))

            if self._farm.turbine_specification.controls.position:
                # Compute the derivatives with respect to the turbine position
                for d in self._farm.turbine_cache["turbine_derivative_pos"]:
                    for var in ('turbine_pos_x', 'turbine_pos_y'):
                        self._farm.turbine_cache.update(self._farm)
                        tfd = d[var]
                        dj.append(djdtf.vector().inner(tfd.vector()))

            dj = numpy.array(dj)

        return dj

    def _compute_functional(self, m, annotate=True):
        """ Compute the functional of interest for the turbine positions/frictions array """
        self.last_m = m
        self._update_turbine_farm(m)

        # Configure dolfin-adjoint
        adj_reset()
        dolfin.parameters["adjoint"]["record_all"] = True
        self._set_revolve_parameters()

        # Solve the shallow water system and integrate the functional of
        # interest.
        final_only = (not self.solver.problem._is_transient or
                      self._problem_params.functional_final_time_only)
        functional = self.functional(self._farm, rho=self._problem_params.rho)
        self.time_integrator = TimeIntegrator(self.solver.problem, functional,
                                              final_only)

        for sol in self.solver.solve(annotate=annotate):
            self.time_integrator.add(sol["time"], sol["state"], sol["tf"],
                                     sol["is_final"])

        return self.time_integrator.integrate()


    def _set_revolve_parameters(self):
        if (hasattr(self._solver_params, "revolve_parameters")
            and self._solver_params.revolve_parameters is not None):
          (strategy,
           snaps_on_disk,
           snaps_in_ram,
           verbose) = self._farm.params['revolve_parameters']
          adj_checkpointing(
              strategy,
              self._problem_params.finish_time/self._problem_params.dt,
              snaps_on_disk=snaps_on_disk,
              snaps_in_ram=snaps_in_ram,
              verbose=verbose)


    def _update_turbine_farm(self, m):
        """ Update the turbine farm from the flattened parameter array m. """
        if self._farm.turbine_specification.smeared:
            self._farm._parameters["friction"] = m

        else:
            controlled_by = self._farm.turbine_specification.controls
            shift = 0
            if controlled_by.friction:
                shift = len(self._farm._parameters["friction"])
                self._farm._parameters["friction"] = m[:shift]
            elif controlled_by.dynamic_friction:
                shift = len(numpy.reshape(self._farm._parameters["friction"],-1))
                nb_turbines = len(self._farm._parameters["position"])
                self._farm._parameters["friction"] = (
                    numpy.reshape(m[:shift], (-1, nb_turbines)).tolist())

            if controlled_by.position:
                m_pos = m[shift:]
                self._farm._parameters["position"] = (
                    numpy.reshape(m_pos, (-1,2)).tolist())

        # Update the farm cache.
        self._farm.turbine_cache.update(self._farm)


    def _save_checkpoint(self):
        """ Checkpoint the reduced functional from which can be used to restart
        the turbine optimisation. """
        base_filename = self.params.checkpoints_basefilename
        base_path = os.path.join(self.params["base_path"], base_filename)
        self._compute_functional_mem.save_checkpoint(base_path + "_fwd.dat")
        self._compute_gradient_mem.save_checkpoint(base_path + "_adj.dat")

    def _load_checkpoint(self):
        """ Checkpoint the reduceduced functional from which can be used to
        restart the turbine optimisation. """
        base_filename = self.params.checkpoints_basefilename
        base_path = os.path.join(self._farm.params["base_path"], base_filename)
        self._compute_functional_mem.load_checkpoint(base_path + "_fwd.dat")
        self._compute_gradient_mem.load_checkpoint(base_path + "_adj.dat")

    def evaluate(self, m, annotate=True):
        """ Return the functional value for the given parameter array. """
        log(INFO, 'Start evaluation of j')
        timer = dolfin.Timer("j evaluation")
        j = self._compute_functional_mem(m, annotate=annotate)
        timer.stop()

        if self.parameters.save_checkpoints:
            self._save_checkpoint()

        log(INFO, 'Runtime: %f s.' % timer.value())
        log(INFO, 'j = %e.' % float(j))
        self.last_j = j

        if self.parameters.automatic_scaling:
            if self._automatic_scaling_factor is None:
                # Computing dj will set the automatic scaling factor.
                log(INFO, ("Computing derivative to determine the automatic "
                           "scaling factor"))
                self._dj(m, forget=False, optimisation_iteration=False)
            return j*self.scale*self._automatic_scaling_factor
        else:
            return j*self.scale

    def _dj(self, m, forget, optimisation_iteration=True):
        """ This memoised function returns the gradient of the functional for the parameter choice m. """
        log(INFO, 'Start evaluation of dj')
        timer = dolfin.Timer("dj evaluation")
        dj = self._compute_gradient_mem(m, forget)

        # We assume that the gradient is computed at and only at the beginning
        # of each new optimisation iteration. Hence, this is the right moment
        # to store the turbine friction field and to increment the optimisation
        # iteration counter.
        if optimisation_iteration:
            self._optimisation_iteration += 1
            if self.solver.parameters.dump_period > 0:
                # A cache hit skips the turbine cache update, so we need
                # trigger it manually.
                if self._compute_gradient_mem.has_cache(m, forget):
                    self._update_turbine_farm(m)
                if self._farm.turbine_specification.controls.dynamic_friction:
                    log(WARNING, ("Turbine VTU output not yet implemented for "
                                  " dynamic turbine control"))
                else:
                    self.turbine_file << self._farm.turbine_cache["turbine_field"]
                    # Compute the total amount of friction due to turbines
                    if self._farm.turbine_specification.smeared:
                        log(INFO, "Total amount of friction: %f" %
                            assemble(self._farm.turbine_cache["turbine_field"]*dx))

        if self.parameters.save_checkpoints:
            self._save_checkpoint()

        log(INFO, "Runtime: " + str(timer.stop()) + " s")
        log(INFO, "|dj| = " + str(numpy.linalg.norm(dj)))

        if self.parameters.automatic_scaling:
            self._set_automatic_scaling_factor(dj)
            return dj*self.scale*self._automatic_scaling_factor
        else:
            return dj*self.scale

    def _set_automatic_scaling_factor(self, dj):
        """ Compute the scaling factor if never done before. """

        if self._automatic_scaling_factor is None:
            if not self._farm.turbine_specification.controls.position:
                raise NotImplementedError("Automatic scaling only works if "
                                          "the turbine positions are control "
                                          "parameters")

            if (self._farm.turbine_specification.controls.friction or
                self._farm.turbine_specification.controls.dynamic_friction):
                assert(len(dj) % 3 == 0)
                # Exclude the first third from the automatic scaling as it
                # contains the friction coefficients.
                djl2 = max(abs(dj[len(dj) / 3:]))
            else:
                djl2 = max(abs(dj))

            if djl2 == 0:
                log(ERROR, ("Automatic scaling failed: The gradient at the "
                            "parameter point is zero."))
            else:
                self._automatic_scaling_factor = abs(
                    self.parameters.automatic_scaling*
                    self._farm.turbine_specification.diameter/
                    djl2/
                    self.scale)
                log(INFO, "Set automatic scaling factor to %e." %
                    self._automatic_scaling_factor)


    def __call__(self, m):
        """ Interface function for dolfin_adjoint.ReducedFunctional """
        return self.evaluate(m)


    def derivative(self, m_array, forget=True, **kwargs):
        """ Computes the first derivative of the functional with respect to its
        parameters by solving the adjoint equations. """

        return self._dj(m_array, forget)


class ReducedFunctionalNumPy(ReducedFunctional):
    pass


class TurbineFarmParameter(object):
    """This class is required to that the parameter set works with
    dolfin-adjoint."""
    def __init__(self, farm):
        self.farm = farm

    def data(self):
        return numpy.asarray(self.farm.control_array)
