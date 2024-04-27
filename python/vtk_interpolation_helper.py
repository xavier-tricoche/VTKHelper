import scipy.integrate
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtk_misc_helper import *
import argparse
import vtk_io_helper
from random import *
import os
import nrrd
import scipy
import time


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
    def __init__(self, vtk_data, field, raise_oob_error=False):
        self.data = vtk_data
        self.field = field
        if not raise_oob_error:
            self.invalid_value = 10000000.*np.ones(self.field.GetNumberOfComponents())
            if len(self.invalid_value) == 1:
                self.invalid_value = self.invalid_value[0]
        self.oob_error = raise_oob_error
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
            if self.oob_error:
                raise ValueError(f'Position {p} is not in dataset domain')
            else:
                return self.invalid_value
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
    def __init__(self, times, filenames, attributes=['vectors'], scale=1, raise_oob_error=False):
        # import mesh - assumed invariant 
        self.mesh = import_dataset(filenames[0])
        self.attributes = attributes
        self.scale = scale
        self.oob_error = raise_oob_error
        self.invalid_values = None

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
        for i in range(self.mesh.GetPointData().GetNumberOfArrays()):
            self.mesh.GetPointData().RemoveArray(i)
        self.update_data(times, filenames)
        self.nattributes = len(self.attributes)

    def update_data(self, newtimes, newfilenames):
        self.times = newtimes
        self.steps = []
        for filename in newfilenames:
            print(f'loading {filename}... ', end='', flush=True)
            tic = time.perf_counter()
            data = import_dataset(filename)
            size = os.path.getsize(filename)
            toc = time.perf_counter()
            dt = toc - tic
            print(f'done ({dt:0.4f} s. / {size/dt/1024/1024:0.3f} MB/s.)', flush=True)
            values = []
            invalid_values = []
            for name in self.attributes:
                values.append(get_attribute(data, name))
                if not self.oob_error and self.invalid_values is None:
                    invalid_values.append(np.ones(values[-1].GetNumberOfComponents()))
                    if len(invalid_values[-1]) == 1:
                        invalid_values[-1] = invalid_values[-1][0]
            if not self.oob_error and self.invalid_values is None:
                if len(invalid_values) == 1:
                    self.invalid_values = invalid_values[0]
                else:
                    self.invalid_values = invalid_values
            self.steps.append(values)

    def __call__(self, t, p):
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(f'Time {t} outside of temporal range {self.times[0]} - {self.times[-1]}')

        tpos = np.searchsorted(self.times, t)
        if tpos == 0:
            tpos += 1
        elif tpos == len(self.times):
            tpos -= 1 
        
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
            if self.oob_error:
                raise ValueError(f'Time Interpolator: Position {p} is not in dataset domain')
            else:
                return self.invalid_values
        cellpts = vtk.vtkIdList()
        self.mesh.GetCellPoints(cellid, cellpts)

        values = []
        for n in range(self.nattributes):
            f0 = np.zeros(self.steps[0][n].GetNumberOfComponents())
            f1 = np.zeros(self.steps[0][n].GetNumberOfComponents())
            for i in range(cellpts.GetNumberOfIds()):
                id = cellpts.GetId(i)
                f0 += weights[i] * np.array(self.steps[n0][n].GetTuple(id))
                f1 += weights[i] * np.array(self.steps[n1][n].GetTuple(id))
            f = self.scale * ((1-u)*f0 + u*f1)
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
    parser.add_argument('-n', '--number', default=1, help='Number of interpolations to perform')
    args = parser.parse_args()

    data = import_dataset(args.input)
    print(data)
    field = get_attribute(data, args.field)
    print(field)
    intp = Interpolator(data, field)


    xmin, xmax, ymin, ymax, zmin, zmax = data.GetBounds()
    pmin = np.array([xmin, ymin, zmin])
    pmax = np.array([xmax, ymax, zmax])
    print(f'mins: {pmin}')
    print(f'maxes: {pmax}')

    MAX_TIMESTAMP = 11
    INITIAL_INDEX = 200000

    filenames = []
    for i in range(1, MAX_TIMESTAMP):
        timestamp = ""
        if (i < 10):
                timestamp = '0' + str(i)
        else:
            timestamp = str(i)
        filenames.append('velocity/velocity_' + timestamp + '.vti')
    #print(filenames)
    tintp = TimeInterpolator([i for i in range(1, MAX_TIMESTAMP)], filenames)


    for i in range(args.number):
        x = random()
        y = random()
        z = random()
        t = random() + 1
        p = np.array([x,y,z])
        p = (np.ones(3)-p)*pmin + p*pmax
        try:
            value = intp(0, p)
            print(f'value at {p} is {value}')
            time_pos_val = tintp(t, p)
            print(f'vector at {value} at time {t} is {time_pos_val}')
        except ValueError:
            print(f'position {p} lies outside domain boundary')
    y0 : vtk.vtkTypeFloat32Array = get_attribute(import_dataset(filenames[1]), 'vectors')
    intitial_value = (1000,1000,10)
    print(f'value of intitial value: {intitial_value}')
    result = scipy.integrate.solve_ivp(tintp, (1.01, MAX_TIMESTAMP - 1), intitial_value, dense_output=True)
    time_points = result['t']
    y = result['y']
    sol : scipy.integrate.OdeSolution = result['sol']
    success = result['success']
    print(f'Time points: {time_points}, y: {y}, sol: {sol}, success: {success}')
    print([sol(i) for i in range(1, 1000)])
