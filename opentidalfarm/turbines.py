import numpy
import configuration
import math
import numpy
from dolfin import *
from dolfin_adjoint import *
from math import log
from helpers import info, info_green, info_red, info_blue

class Turbines(object):

    def __init__(self, V, params, derivative_index_selector=-1):
        self.params = configuration.Parameters(params)
  
        # Precompute some turbine parameters for efficiency. 
        self.x = interpolate(Expression("x[0]"), V).vector().array()
        self.y = interpolate(Expression("x[1]"), V).vector().array()
        self.V = V

    def __call__(self, name = "", derivative_index_selector=None, derivative_var_selector=None):
        ''' If the derivative selector is i >= 0, the Expression will compute the derivative of the turbine with index i with respect 
          to either the x or y coorinate or its friction parameter. '''
        V = self.V
        params = self.params
  
        if derivative_index_selector == None: 
          turbine_pos = params["turbine_pos"]
          turbine_friction = params["turbine_friction"]
        else:
          turbine_pos = [params["turbine_pos"][derivative_index_selector]]
          turbine_friction = [params["turbine_friction"][derivative_index_selector]]

        ff = numpy.zeros(len(self.x))
        # We dont mind division by zero
        numpy.seterr(divide = 'ignore')
        eps = 1e-12
        for (x_pos, y_pos), friction in zip(turbine_pos, turbine_friction):
          x_unit = numpy.minimum(numpy.maximum((self.x - x_pos) / (0.5*self.params["turbine_x"]), -1+eps), 1-eps) 
          y_unit = numpy.minimum(numpy.maximum((self.y - y_pos) / (0.5*self.params["turbine_y"]), -1+eps), 1-eps) 

          # Apply chain rule to get the derivative with respect to the turbine friction 
          e = numpy.exp(-1/(1-x_unit**2) - 1./(1-y_unit**2)+2)
          if derivative_index_selector == None:
            ff += e * friction

          elif derivative_var_selector == 'turbine_friction':
            ff += e 

          if derivative_var_selector == 'turbine_pos_x':
            ff += e * (-2*x_unit / ((1.0-x_unit**2)**2)) * friction*(-1.0/(0.5*params["turbine_x"])) 

          elif derivative_var_selector == 'turbine_pos_y':
            ff += e * (-2*y_unit / ((1.0-y_unit**2)**2)) * friction*(-1.0/(0.5*params["turbine_y"])) 

        numpy.seterr(divide = 'warn')

        f = Function(V, name = name)
        f.vector().set_local(ff) 
        f.vector().apply("insert")
        return f

class TurbineCache:
    def __init__(self):
        self.cache = {}
        self.params = None

    def update(self, config):
        ''' Creates a list of all turbine function/derivative interpolations. This list is used as a cache 
          to avoid the recomputation of the expensive interpolation of the turbine expression. '''
        # If the parameters have not changed, then there is no need to do anything
        if self.params != None:
            if (self.params["turbine_friction"] == config.params["turbine_friction"]).all() and (self.params["turbine_pos"] == config.params["turbine_pos"]).all(): 
                return 

        info_green("Updating turbine cache")

        # Store the new turbine paramaters
        self.params = configuration.Parameters(config.params)
        self.params["turbine_friction"] = numpy.copy(config.params["turbine_friction"])
        self.params["turbine_pos"] = numpy.copy(config.params["turbine_pos"])

        # Precompute the interpolation of the friction function of all turbines
        turbines = Turbines(config.turbine_function_space, self.params)
        tf = turbines(name = "functional_turbine_friction")
        self.cache["turbine_field"] = tf

        # Precompute the interpolation of the friction function for each individual turbine
        if self.params["print_individual_turbine_power"]:
            info_green("Building individual turbine power friction functions for caching purposes...")
            self.cache["turbine_field_individual"] = [] 
            for i in range(len(self.params["turbine_friction"])):
                params_cpy = configuration.Parameters(self.params)
                params_cpy["turbine_pos"] = [self.params["turbine_pos"][i]]
                params_cpy["turbine_friction"] = [self.params["turbine_friction"][i]]
                turbine = Turbines(config.turbine_function_space, params_cpy)
                tf = turbines(name = "functional_turbine_friction") 
                self.cache["turbine_field_individual"].append(tf)

        # Precompute the derivatives with respect to the friction magnitude of each turbine
        if "turbine_friction" in self.params["controls"]:
            self.cache["turbine_derivative_friction"] = []
            for n in range(len(self.params["turbine_friction"])):
                tfd = turbines(derivative_index_selector = n, 
                               derivative_var_selector = 'turbine_friction', 
                               name = "functional_turbine_friction_derivative_with_respect_friction_magnitude_of_turbine_" + str(n)) 
                self.cache["turbine_derivative_friction"].append(tfd)

        # Precompute the derivatives with respect to the turbine position
        if "turbine_pos" in self.params["controls"]:
            self.cache["turbine_derivative_pos"] = []
            for n in range(len(self.params["turbine_pos"])):
                self.cache["turbine_derivative_pos"].append({})
                for var in ('turbine_pos_x', 'turbine_pos_y'):
                    tfd = turbines(derivative_index_selector = n, 
                                        derivative_var_selector = var,
                                        name = "functional_turbine_friction_derivative_with_respect_position_of_turbine_" + str(n))
                    self.cache["turbine_derivative_pos"][-1][var] = tfd

if __name__ == "__main__":
    mesh = RectangleMesh(-1, -1, 1, 1, 100, 100)
    V = FunctionSpace(mesh, "CG", 1)
    
    params = {"turbine_friction": [0.1, 0.2],
              "turbine_pos": [[0.5, 0.5], [-0.5, -0.5]],
              "turbine_x": 0.5, 
              "turbine_y": 0.5, 
             }

    turbines = Turbines(V, params)
    f = turbines()#1, "turbine_pos_y")
    plot(f, interactive = True)