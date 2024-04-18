import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtk_misc_helper import *
import argparse
import vtk_io_helper
from random import *
import os
import nrrd

def import_dataset(filename):
    try:
        reader = vtk_io_helper.readVTK(filename)
        reader.Update()
        data = reader.GetOutput()
    except ValueError:
        if os.path.splitext(filename)[1].lower() == '.nrrd':
            data, _ = nrrd.read(filename)
        else:
            raise ValueError(f'Unrecognized file type: {filename}')
    return data

def get_attribute(dataset, name):
    if name.lower() == 'scalar' or name.lower() == 'scalars':
        return dataset.GetPointData().GetScalars()
    elif name.lower() == 'vector' or name.lower() == 'vectors':
        return dataset.GetPointData().GetVectors()
    elif name.lower() == 'tensor' or name.lower() == 'tensors':
        return dataset.GetPointData().GetTensors()
    else:
        return dataset.GetPointData().GetArray(name)

class Interpolator:
    def __init__(self, vtk_data, field):
        self.data = vtk_data
        self.field = field
        if isinstance(self.data, vtk.vtkPointSet):
            # dataset has explicit coordinates representation
            # non-trivial point location case
            self.locator = vtk.vtkCellTreeLocator()
            self.locator.SetDataSet(self.data)
            self.locator.BuildLocator()
        elif isinstance(self.data, vtk.vtkImageData) or isinstance(self.data, vtk.vtkRectilinearGrid):
            self.locator = None
        else:
            raise ValueError('Unrecognized dataset type')

    def __call__(self, t, p):
        p = np.array(p)
        acell = vtk.vtkGenericCell()
        subid = vtk.reference(0)
        pcoords = np.zeros(3)
        weights = np.zeros(8)
        if self.locator is not None:
            cellid = self.locator.FindCell(p, 0, acell, subid, pcoords, weights)
        else:
            cellid = self.data.FindCell(p, None, 0, 0.0, subid, pcoords, weights)
        if cellid == -1:
            raise ValueError(f'Position {p} is not in dataset domain')
        cellpts = vtk.vtkIdList()
        self.data.GetCellPoints(cellid, cellpts)
        # cellpts = cellpts.get()
        f = np.zeros(self.field.GetNumberOfComponents())
        # weights = weights.get()
        for i in range(cellpts.GetNumberOfIds()):
            id = cellpts.GetId(i)
            f += weights[i] * np.array(self.field.GetTuple(id))
        if len(f) == 1: return f[0]
        else: return f

class TimeInterpolator:
    def __init__(self, times, filenames, attributes=['vectors']):
        # import mesh - assumed invariant 
        self.mesh = import_dataset(self.filenames[0])
        self.attributes = attributes

        if isinstance(self.mesh, vtk.vtkPointSet):
            # dataset has explicit coordinates representation
            # non-trivial point location case
            self.locator = vtk.vtkCellTreeLocator()
            self.locator.SetDataSet(self.data)
            self.locator.BuildLocator()
        elif isinstance(self.mesh, vtk.vtkImageData) or isinstance(self.mesh, vtk.vtkRectilinearGrid):
            self.locator = None
        else:
            raise ValueError('Unrecognized dataset type')

        # remove attributes
        for i in self.mesh.GetPointData().GetNumberOfArrays():
            self.mesh.GetPointData().RemoveArray(i)
        self.update_data(times, filenames)
        self.nattributes = len(self.attributes)

    def update_data(self, newtimes, newfilenames):
        self.times = newtimes
        self.steps = []
        for filename in newfilenames:
            data = import_dataset(filename)
            values = []
            for name in self.attributes:
                values.append(get_attribute(data, name))
            self.steps.append(values)

    def __call__(self, t, p):
        tpos = np.searchsorted(self.times, t)
        if tpos == 0 or tpos == len(self.times):
            raise ValueError(f'Time {t} outside of temporal range {self.times[0]} - {self.times[-1]}')
        
        n0 = tpos-1
        n1 = tpos
        t0 = self.times[n0]
        t1 = self.times[n1]
        u = (t-t0)/(t1-t0)

        p = np.array(p)
        acell = vtk.vtkGenericCell()
        subid = vtk.reference(0)
        pcoords = np.zeros(3)
        weights = np.zeros(8)
        if self.locator is not None:
            cellid = self.locator.FindCell(p, 0, acell, subid, pcoords, weights)
        else:
            cellid = self.mesh.FindCell(p, None, 0, 0.0, subid, pcoords, weights)
        if cellid == -1:
            raise ValueError(f'Position {p} is not in dataset domain')
        cellpts = vtk.vtkIdList()
        self.mesh.GetCellPoints(cellid, cellpts)

        values = []
        for n in self.nattributes:
            f0 = np.zeros(self.steps[0][n].GetNumberOfComponents())
            f1 = np.zeros(self.steps[0][n].GetNumberOfComponents())
            for i in range(cellpts.GetNumberOfIds()):
                id = cellpts.GetId(i)
                f0 += weights[i] * np.array(self.steps[n0][n].GetTuple(id))
                f1 += weights[i] * np.array(self.steps[n1][n].GetTuple(id))
            f = (1-u)*f0 + u*f1
            if len(f) == 1: f = f[0]
            values.append(f)
        if len(values) == 1:
            return values[0]
        else:
            return values

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test RHS wrapper for VTK datasets')
    parser.add_argument('-i', '--input', required=True, help='Input dataset')
    parser.add_argument('-f', '--field', required=True, help='Field to interpolate')
    parser.add_argument('-n', '--number', default=100, help='Number of interpolations to perform')
    args = parser.parse_args()

    data = import_dataset(args.input)
    print(data)
    field = get_attribute(data, args.field)
    print(field)
    intp = Interpolator(data, field)


    xmin, xmax, ymin, ymax, zmin, zmax = data.GetBounds()
    pmin = np.array([xmin, ymin, zmin])
    pmax = np.array([xmax, ymax, zmax])

    for i in range(args.number):
        x = random()
        y = random()
        z = random()
        p = np.array([x,y,z])
        p = (np.ones(3)-p)*pmin + p*pmax
        try:
            value = intp(0, p)
            print(f'value at {p} is {value}')
        except ValueError:
            print(f'position {p} lies outside domain boundary')


