from rectangle_mesh_domain import *
from dolfin_adjoint import *
from opentidalfarm import *

""" Generation of non-uniformly randomly refined meshes based on given
initial rectangular mesh with rectangular farm region in the middle """

# Generate domains with cell_ids
domain = FileDomain("mesh/mesh.xml")
mesh = domain.mesh
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)
domains = domain.cell_ids

# define random refinement rule
def randomly_refine(initial_mesh, ratio_to_refine= .15):
    numpy.random.seed(1)
    cf = CellFunction('bool', initial_mesh)
    for k in xrange(len(cf)):
        if numpy.random.rand() < ratio_to_refine and domains.array()[k]==1:
            cf[k] = True
    return refine(initial_mesh, cell_markers = cf)

# 0-th refinement level
File('mesh0.xml') << mesh

# First refinement level
mesh = Mesh('mesh0.xml')
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)
domains = domain.cell_ids

mesh = randomly_refine(domain.mesh)
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)
domains = domain.cell_ids

print "Relation between hmin/hmax: ", mesh.hmax()/mesh.hmin()
File('mesh1.xml') << mesh

# Second refinement level
mesh = Mesh('mesh1.xml')
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)
domains = domain.cell_ids

mesh = randomly_refine(domain.mesh)
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)
domains = domain.cell_ids

print "Relation between hmin/hmax: ", mesh.hmax()/mesh.hmin()
File('mesh2.xml') << mesh

# Third refinement level
mesh = Mesh('mesh2.xml')
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)
domains = domain.cell_ids

mesh = randomly_refine(domain.mesh)
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)#
domains = domain.cell_ids

print "Relation between hmin/hmax: ", mesh.hmax()/mesh.hmin()
File('mesh3.xml') << mesh

# Fourth refinement level
mesh = Mesh('mesh3.xml')
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)
domains = domain.cell_ids

mesh = randomly_refine(domain.mesh)
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)
domains = domain.cell_ids

print "Relation between hmin/hmax: ", mesh.hmax()/mesh.hmin()
File('mesh4.xml') << mesh

# Fivth refinement level
mesh = Mesh('mesh4.xml')
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)
domains = domain.cell_ids

mesh = randomly_refine(domain.mesh)
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)
domains = domain.cell_ids

print "Relation between hmin/hmax: ", mesh.hmax()/mesh.hmin()
File('mesh5.xml') << mesh
