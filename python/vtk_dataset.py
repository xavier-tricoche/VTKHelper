import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
from vtk_misc_helper import *
import argparse

''' Convenience function to allow for individual values to be queried for size '''
def length(obj):
    if not hasattr(obj, '__len__'):
        return 1
    else:
        return len(obj)

''' Convenience function to turn any coordinates into 3D coordinates '''
def make3d(p):
    if length(p) == 3:
        return p
    elif length(p) == 2:
        return [p[0], p[1], 0.]
    elif length(p) == 1:
        return [p[0], 0., 0.]

''' Create vtkPoints out of a bunch of coordinates '''
def make_vtkpoints(positions):
    coords = np.ndarray((length(positions), 3), dtype=float)
    for i, p in enumerate(positions):
        coords[i] = make3d(p)
    pts = vtk.vtkPoints()
    data = numpy_to_vtk(coords)
    pts.SetData(data)
    return pts

''' Create a polydata out of a subset of points '''
def make_points(positions, selected=None):
    if selected is None:
        selected = [ i for i in range(length(positions) ) ]
    coords = np.ndarray((length(selected), 3), dtype=float)
    for i, id in enumerate(selected):
        coords[i] = make3d(positions[id])
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(coords))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    return polydata

''' Add sphere glyphs to a set of points '''
def make_spheres(dataset, radius=1,theta_res=12, phi_res=12, scale=False):
    source = vtk.vtkSphereSource()
    source.SetRadius(radius)
    source.SetPhiResolution(phi_res)
    source.SetThetaResolution(theta_res)
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetSourceConnection(source.GetOutputPort())
    connect(dataset, glyphs)
    glyphs.SetScaling(scale)
    glyphs.SetScaleModeToScaleByScalar()
    glyphs.Update()
    spheres = glyphs.GetOutput()
    return spheres

''' Add arrow glyphs to a set of points to represent vector values '''
def make_arrows(dataset, shaft_radius=0.2, shaft_resolution=12, tip_length=1, tip_radius=0.5, tip_resolution=12, scale=False):
    source = vtk.vtkArrowSource()
    source.SetShaftRadius(shaft_radius)
    source.SetShaftResolution(shaft_resolution)
    source.SetTipLength(tip_length)
    source.SetTipRadius(tip_radius)
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetSourceConnection(source.GetOutputPort())
    connect(dataset, glyphs)
    glyphs.SetScaling(scale)
    glyphs.SetScaleModeToScaleByVector()
    glyphs.Update()
    arrows = glyphs.GetOutput()
    return arrows

''' Add scalar values to a dataset. scalars is an array-like container of
    scalar values. '''
def add_scalars(inout, scalars, point_data=True, name="anonymous_scalars",
                active=True):
    values = np.ndarray((length(scalars)), dtype=float)
    for i, s in enumerate(scalars):
        values[i] = s
    values = numpy_to_vtk(values)
    values.SetName(name)
    if point_data:
        if active:
            inout.GetPointData().SetScalars(values)
        else:
            inout.GetPointData().AddArray(values)
    else:
        if active:
            inout.GetCellData().SetScalars(values)
        else:
            inout.GetCellData().AddArray(values)
    return inout

''' Add color attributes to point/cell data. "colors" is aan array-like
    container of RGB 3-vectors that can be cast to unsigned char'''
def add_colors(inout, colors, point_data=True, name="anonymous_colors",
               active=True):
    values = np.ndarray((length(colors), 3), dtype=np.ubyte)
    for i, c in enumerate(colors):
        values[i] = np.array([c[0], c[1], c[2]], dtype=np.ubyte)
    values = numpy_to_vtk(values)
    values.SetName(name)
    if point_data:
        if active:
            inout.GetPointData().SetScalars(values)
        else:
            inout.GetPointData().AddArray(values)
    else:
        if active:
            inout.GetCellData().SetScalars(values)
        else:
            inout.GetCellData().AddArray(values)
    return inout

''' Add vector attributes to point/cell data. vectors is an array-like
    container of 1D arrays'''
def add_vectors(inout, vectors, point_data=True, name="anonymous_vectors",
                active=True):
    values = np.ndarray((len(vectors), 3), dtype=float)
    for i, v in enumerate(vectors):
        values[i] = make3d(v)
    values = numpy_to_vtk(values)
    values.SetName(name)
    if point_data:
        if active:
            inout.GetPointData().SetVectors(values)
        else:
            inout.GetPointData().AddArray(values)
    else:
        if active:
            inout.GetCellData().SetVectors(_vectors)
        else:
            inout.GetCellData().AddArray(_vectors)
    return inout

''' Add tensor attributes to point/cell data. "tensors" is an array-like
    container of 1D arrays.'''
def add_tensors(inout, tensors, point_data=True, name="anonymous_tensors",
                active=True):
    size = length(tensors[0])
    values = np.ndarray((len(tensors), 9), dtype=float)
    if size==3:
        # symmetric 2d tensor
        for i, t in enumerate(tensors):
            values[i] = [ t[0], t[1], 0., t[1], t[2], 0., 0., 0., 0. ]
    elif size==4:
        # 2d tensors
        for i, t in enumerate(tensors):
            values[i] = [ t[0], t[1], 0., t[2], t[3], 0., 0., 0., 0. ]
    elif size==6:
        # symmetric 3d tensors
        for i, t in enumerate(tensors):
            values[i] = [ t[0], t[1], t[2], t[1], t[3], t[4], t[2], t[4], t[5] ]
    elif size==9:
        # 3d tensors
        for i, t in enumerate(tensors):
            values[i] = t
    values = numpy_to_vtk(values)
    values.SetName(name)
    if point_data:
        if active:
            inout.GetPointData().SetTensors(values)
        else:
            inout.GetPointData().AddArray(values)
    else:
        if active:
            inout.GetCellData().SetTensors(_vectors)
        else:
            inout.GetCellData().AddArray(_vectors)
    return inout

''' Add texture coordinates to point data. "tcoords" is a container
    of 1D arrays'''
def add_tcoords(inout, tcoords):
    values = np.ndarray((length(tcoords), 2), dtype=float)
    for i, c in enumerate(tcoords):
        values[i] = [ c[0], c[1] ]
    values = numpy_to_vtk(values)
    values.SetName('texture_coordinates')
    inout.GetPointData().SetTCoords(values)
    return inout

''' VTK does not show points unless they are included in some cells.
    To show them as points, they need to be associated with 1-cells that VTK
    calls vertices. '''
def add_vertices(inout):
    vertices = vtk.vtkCellArray()
    npts = inout.GetNumberOfPoints()
    vertices.InitTraversal()
    for i in range(npts):
        vertices.InsertNextCell(1)
    inout.SetVerts(vertices)
    return inout

''' Add line segments to a dataset '''
def add_segments(inout, segment):
    cells = vtk.vtkCellArray()
    for seg in segments:
        cells.InsertNextCell(2)
        cells.InsertCellPoint(seg[0])
        cells.InsertCellPoint(seg[1])
    inout.SetLines(cells)
    return inout

''' Add polylines to a dataset '''
def add_polylines(inout, lines):
    cells = vtk.vtkCellArray()
    for line in lines:
        cells.InsertNextCell(length(line))
        for id in line:
            cells.InsertCellPoint(id)
    inout.SetLines(cells)
    return inout

''' Construct them triangulation of a set of points'''
def add_mesh2d(inout):
    delaunay = vtk.vtkDelaunay2D()
    connect(inout, delaunay)
    delaunay.BoundingTriangulationOff()
    delaunay.Update()
    inout.SetPolys(delaunay.GetOutput().GetPolys())
    return inout

''' Compute 3D triangulation (tetrahedrization) of a bunch of points '''
def add_mesh3d(inout):
    delaunay = vtk.vtkDelaunay3D()
    connect(inout, delaunay)
    delaunay.BoundingTriangulationOff()
    delaunay.Update()
    inout.SetPolys(delaunay.GetOutput().GetCells())
    return inout

''' Apply clip filter to dataset '''
def clip_polydata(dataset, plane):
    clip = vtk.vtkClipPolyData()
    connect(dataset, clip)
    clip.SetClipFunction(plane)
    clip.GenerateClipScalarsOff()
    clip.GenerateClippedOutputOn()
    clip.SetValue(0)
    clip.Update()
    return clip.GetOutput()

''' Create a plane (e.g., for clipping) '''
def make_plane(normal, origin):
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin[0], origin[1], origin[2])
    plane.SetNormal(normal[0], normal[1], normal[2])
    return plane


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test vtk_dataset functionalities')
    parser.add_argument('-n', '--number', type=int, default=100, help='Number of data points to consider')
    args = parser.parse_args()

    for dim in range(2,4):
        array = np.random.rand(args.number, dim)
        poly= make_points(array)
        spheres = make_spheres(poly)
        scalars = np.random.rand(args.number)
        spheres = make_spheres(poly)
        vectors = np.random.rand(args.number, dim)
        tensors = np.random.rand(args.number, dim*dim)
        colors = 255.*np.random.rand(args.number, 3)
        tcoords = np.random.rand(args.number, 2)
        poly = add_scalars(poly, scalars)
        poly = add_vectors(poly, vectors)
        poly = add_tensors(poly, tensors)
        poly = add_colors(poly, colors)
        poly = add_tcoords(poly, tcoords)
        spheres = make_spheres(poly, radius=0.05)
        arrows = make_arrows(poly, tip_length=0.05, tip_radius=0.025, shaft_radius=0.01, tip_resolution=20)

        if dim == 2:
            poly = add_mesh2d(poly)
        else:
            poly = add_mesh3d(poly)

        mapper = vtk.vtkDataSetMapper()
        connect(poly, mapper)
        mapper.ScalarVisibilityOn()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        mapper2 = vtk.vtkPolyDataMapper()
        connect(spheres, mapper2)
        mapper2.ScalarVisibilityOn()
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper2)

        mapper3 = vtk.vtkPolyDataMapper()
        connect(arrows, mapper3)
        actor3 = vtk.vtkActor()
        actor3.SetMapper(mapper3)

        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.AddActor(actor2)
        renderer.AddActor(actor3)

        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        interactor.Initialize()
        window.Render()
        interactor.Start()
