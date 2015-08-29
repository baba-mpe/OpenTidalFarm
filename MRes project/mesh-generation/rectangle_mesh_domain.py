import os.path
import dolfin
from opentidalfarm.domains.domain import Domain

""" Generation of a rectangular mesh domain class, 
    based on the domain class in OpenTidalFarm"""

class RectangularMeshDomain(Domain):
    """ Create a rectangular domain.

    :param x0: The x coordinate of the bottom-left.
    :type x0: float
    :param y0: The y coordinate of the bottom-left.
    :type y0: float
    :param x1: The x coordinate of the top-right corner.
    :type x1: float
    :param y1: The y coordinate of the top-right corner.
    :type y1: float
    """


    def __init__(self, mesh, x0, y0, x1, y1):
        # A :class:`dolfin.Mesh` containing the mesh.
        self.mesh = mesh

        class Left(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and dolfin.near(x[0], x0)

        class Right(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and dolfin.near(x[0], x1)

        class Sides(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (dolfin.near(x[1], y0) or dolfin.near(x[1], y1))

        class TurbineArea(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return True if (x[0] <= 1250 and x[1] <= 650 and 750 <= x[0] and 350 <= x[1]) else False


        # Initialize sub-domain instances
        left = Left()
        right = Right()
        sides = Sides()
        turbine_area = TurbineArea()
        

        # Create facet markers
        # A :class:`dolfin.FacetFunction` containing the surface markers.
        self.facet_ids = dolfin.FacetFunction('size_t', self.mesh)
        self.facet_ids.set_all(0)
        left.mark(self.facet_ids, 1)
        right.mark(self.facet_ids, 2)
        sides.mark(self.facet_ids, 3)
        # A :class:`dolfin.Measure` for the facet parts.
        self._ds = dolfin.Measure('ds')[self.facet_ids]


        # A :class:`dolfin.CellFunction` containing the area markers.
        self.cell_ids = dolfin.CellFunction("size_t", self.mesh)
        self.cell_ids.set_all(0)
        

        # Mark the turbine area
        turbine_area.mark(self.cell_ids,1)


        # A :class:`dolfin.Measure` for the cell cell_ids.
        self._dx = dolfin.Measure("dx")[self.cell_ids]
