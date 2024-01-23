import vtk
import numpy as np
import argparse
from vtk.util import numpy_support
from numpy.random import default_rng

'''
Create a mapper for an arbitrary object
'''
def get_mapper(object):
    is_algo = False
    if isinstance(object, vtk.vtkAlgorithm):
        print(object)
        data = object.GetOutputDataObject(0)
        is_algo = True
    else:
        data = object
    if isinstance(data, vtk.vtkPolyData):
        mapper = vtk.vtkPolyDataMapper()
    elif isinstance(data, vtk.vtkGraph):
        mapper = vtk.vtkGraphMapper()
    else:
        mapper = vtk.vtkDataSetMapper()
    if is_algo:
        mapper.SetInputConnection(object.GetOutputPort())
    else:
        mapper.SetInputData(object)
    return mapper

''' Create an actor from an arbitrary object'''
def make_actor(object):
    a = vtk.vtkActor()
    if isinstance(object, vtk.vtkMapper):
        m = object
    else:
        m = get_mapper(object)
    a.SetMapper(m)
    return a

'''
Tube filter
'''
def make_tubes(source, radius=0.1, resolution=12):
    if not isinstance(source, vtk.vtkAlgorithm):
        s = vtk.vtkTrivialProducer()
        s.SetOutput(source)
        source = s
    tubes = vtk.vtkTubeFilter()
    tubes.SetInputConnection(source.GetOutputPort())
    tubes.SetRadius(radius)
    tubes.SetNumberOfSides(resolution)
    return make_actor(tubes, True), tubes

'''
Make spheres
'''
def make_spheres(source, radius=0.2, resolution=12):
    if not isinstance(source, vtk.vtkAlgorithm):
        data = source
        source = vtk.vtkTrivialProducer()
        source.SetOutput(data)
    else:
        data = source.GetOutputDataObject()
    if data.GetVerts() is None:
        cells = vtk.vtkCellArray()
        n = data.GetNumberOfPoints()
        for i in range(n):
            data.InsertNextCell(1)
            data.InsertCellPoint(i)
        data.SetVerts(cells)
    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(resolution)
    sphere.SetPhiResolution(resolution)
    sphere.SetRadius(radius)
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetInputConnection(source.GetOutputPort())
    glyph.ScalingOff()
    return make_actor(glyph, True)

'''
Make tensor ellipsoid glyphs
'''
def make_ellipsoids(source, scaling=10, resolution=18):
    if not isinstance(source, vtk.vtkAlgorithm):
        data = source
        source = vtk.vtkTrivialProducer()
        source.SetOutput(data)
    else:
        data = source.GetOutputDataObject()
    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(resolution)
    sphere.SetPhiResolution(resolution)
    tglyph = vtk.vtkTensorGlyph()
    tglyph.SetSourceConnection(sphere.GetOutputPort())
    tglyph.SetInputConnection(source.GetOutputPort())
    tglyph.SetScaleFactor(scaling)
    tglyph.ClampScalingOn()
    tglyph.ThreeGlyphsOff()
    return make_actor(tglyph, True)

'''
Make actor from single fiber
'''
def make_fiber_actor(fiber, values=None, radius=0.1, resolution=12, as_tube=True):
    poly = make_points(fiber)
    if values is not None:
        if not isinstance(values, vtk.vtkObject):
            values = numpy_support.numpy_to_vtk(values)
        poly.GetPointData().SetScalars(values)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(len(fiber))
    for i in range(len(fiber)):
        cells.InsertCellPoint(i)
    poly.SetLines(cells)
    if as_tube:
        a, tubes = make_tubes(poly, radius=radius, resolution=resolution)
    else:
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(poly)
        m.ScalarVisibilityOn()
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().RenderLinesAsTubesOn()
        a.GetProperty().SetLineWidth(radius)
        tubes = poly
    if values is None:
        rng = default_rng()
        col = rng.random([3])
        a.GetProperty().SetColor(col)
    else:
        a.GetMapper().ScalarVisibilityOn()
    return a, tubes


if __name__ == '__main__':
    src = vtk.vtkSphereSource()
    src.SetThetaResolution(100)
    src.SetPhiResolution(100)
    actor = make_actor(src)
    actor.GetProperty().SetColor(1,0,0)

    src.Update()
    copy = vtk.vtkPolyData()
    copy.DeepCopy(src.GetOutput())
    xform = vtk.vtkTransform()
    xform.Translate(1,1,0)
    xformpd = vtk.vtkTransformPolyDataFilter()
    xformpd.SetInputData(copy)
    xformpd.SetTransform(xform)
    xformpd.Update()
    actor2 = make_actor(xformpd.GetOutput())
    actor2.GetProperty().SetColor(0,0,1)

    image = vtk.vtkImageData()
    image.SetDimensions(1000, 1000, 1)
    image.SetOrigin(0, 0, 0)
    image.SetSpacing(0.001, 0.001, 1)
    t = np.linspace(0, 2 * np.pi, 1000)
    data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]
    image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(data2d.flatten('F')))
    actor3 = make_actor(image)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor(actor2)
    renderer.AddActor(actor3)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1920, 1080)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.Initialize()
    window.Render()
    interactor.Start()