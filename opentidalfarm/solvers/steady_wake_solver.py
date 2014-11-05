import numpy
from ..helpers import FrozenClass
from solver import Solver
from ..problems import SteadyWakeProblem


# class SteadyWakeSolverParameters(FrozenClass):
#     """A set of parameters for a :class:`SteadyWakeSolver`."""
#     pass


class SteadyWakeSolver(Solver):
    """A steady-state solver for Wake models."""
    def __init__(self, problem):
        if not isinstance(problem, SteadyWakeProblem):
            raise TypeError("'problem' must be of type SteadyWakeProblem")

        self.problem = problem
        self._farm = self.problem.parameters.tidal_farm


    def solve(self):
        """Returns an iterator for solving the steady wake problem."""

        flow_speeds = []
        turbines = self._farm.turbine_positions
        wake_model = self.problem.parameters.wake_model
        for i, turbine in enumerate(turbines):
            combiner = self.problem.parameters.combination_model(turbine)
            other_turbines = numpy.delete(numpy.copy(turbines), i, axis=0)
            for due_to in other_turbines:
                flow_at_turbine = wake_model.flow_at(turbine)
                flow_multiplier = wake_model.multiplier(turbine, due_to)
                combiner.add(flow_at_turbine*flow_multiplier)

            flow_speeds.append(combiner.reduce())
        return numpy.asarray(flow_speeds)
