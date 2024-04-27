import scipy as sp 
import numpy as np 
import vtk
from vtk.util import numpy_support 
import argparse 
import random
from tqdm import tqdm
import os
import fnmatch

import sys

from vtk_interpolation_helper import *

sys.path.append('/Users/xmt/code/github/VTKHelper/python')

import vtk_io_helper
import vtk_dataset
import vtk_colors

def find_files(path, patterns):
    filenames = []
    for file in os.listdir(path):
        for pattern in patterns:
            if fnmatch.fnmatchcase(file, pattern):
                filenames.append(os.path.join(path, file))
                break
    return sorted(filenames)

class OutOfBoundsEvent:
    def __init__(self, bounds):
        self.bounds = bounds 
        self.terminal = True

    def __call__(self, t, y):
        d = min(y[0]-self.bounds[0], self.bounds[1]-y[0],
                   y[1]-self.bounds[2], self.bounds[3]-y[1],
                   y[2]-self.bounds[4], self.bounds[5]-y[2])
        return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute pathlines in a time dependent vector field')
    parser.add_argument('-p', '--path', type=str, default='.', help='Path of data files')
    parser.add_argument('-f', '--filenames', type=str, nargs='+', required=True, help='Files containing vector field timesteps (can be a unix pattern)')
    parser.add_argument('-t', '--times', type=float, nargs='+', help='Time coordinates of individual time steps')
    parser.add_argument('-n', '--number', type=int, default=1000, help='Number of pathlines to compute')
    parser.add_argument('--size', type=int, default=3, help='Number of timesteps to process at once')
    parser.add_argument('-v', '--value_name', type=str, default='vectors', help='Name of vector field variable')
    parser.add_argument('-x', '--scale', type=float, default=1, help='Scaling factor for velocity values')
    parser.add_argument('--delta_t', type=float, help='Time interval between timesteps (if uniform)')
    parser.add_argument('--t_init', type=float, default=0, help='Time coordinate of first time step')
    args = parser.parse_args()


    filenames = find_files(args.path, args.filenames)
    print('filenames are now:\n', filenames)


    if args.times is not None and len(filenames) != len(args.times):
        raise ValueError('Filenames and times do not match')
    elif args.times is None:
        if args.delta_t is None:
            raise ValueError('No time coordinates in input')
        else:
            args.times = [ args.t_init + n*args.delta_t for n, _ in enumerate(filenames) ]
    
    
    print(f'there are {args.number} pathlines to compute')
    
    nsteps = len(filenames)
    if nsteps == 0:
        raise RuntimeError('No files in input')
    elif nsteps == 1:
        raise RuntimeError('Only a single time step available')
    
    intp = None

    # determine bounds 
    reader = vtk_io_helper.readVTK(filenames[0])
    reader.Update()
    bounds = reader.GetOutput().GetBounds()
    event = OutOfBoundsEvent(bounds)
    lower = np.array([bounds[0], bounds[2], bounds[4]])
    upper = np.array([bounds[1], bounds[3], bounds[5]])

    seed(0) # for reproducibility

    seeds = []
    t0 = args.times[0]
    for i in range(args.number):
        q = np.array([random(), random(), random()])
        p = (np.ones(3) - q)*lower + q*upper
        seeds.append(p)

    pathlines = [ [seeds[i]] for i in range(args.number) ]
    ptimes = [ [t0] for i in range(args.number) ]

    next = 0
    stopped  = np.zeros(len(pathlines))
    while next < len(args.times)-1:
        cur = next
        next = min(cur+args.size-1, len(args.times)-1)
        nfiles = next - cur + 1
        times = args.times[cur:next+1]
        dt = (times[-1] - times[0])/100
        bwd = dt < 0
        names = filenames[cur:next+1]
        if intp is None:
            intp = TimeInterpolator(times, names, attributes=[args.value_name], scale=args.scale)
        else:
            intp.update_data(times, names)
        for i, pathline in tqdm(enumerate(pathlines)):
            # print(f'Integration pathline #{i}...', end='', flush=True)
            if stopped[i]:
                # print('already stopped')
                continue
        
            aseed = pathline[-1]
            try:
                res = sp.integrate.solve_ivp(
                    intp, t_span=(times[0], times[-1]), y0=aseed, method='DOP853', 
                    dense_output=True, events=event, first_step=dt, max_step=20*dt,
                    rtol=1.0e-3, atol=1.0e-6)
                if bwd:
                    steps = np.arange(res.sol.t_max, res.sol.t_min, dt)
                else:
                    steps = np.arange(res.sol.t_min, res.sol.t_max, dt)
                for t in steps[1:]:
                    pathlines[i].append(res.sol(t))
                    ptimes[i].append(t)
                if (bwd and res.sol.t_min > times[-1]) or (not bwd and res.sol.t_max < times[-1]):
                    stopped[i] = 1
                    if bwd:
                        t_stop = res.sol.t_min
                    else:
                        t_stop = res.sol.t_max
                    print(f'pathlines #{i} has ended at time {t_stop}')
            except:
                stopped[i] = 1
                print(f'pathline #{i} has ended')
        
        for i, p in enumerate(pathlines):
            print(f'pathline #{i} contains {len(p)} points')

    strides = [0]
    npts = 0

    all_pts = []
    all_times = []
    for p,t  in zip(pathlines, ptimes):
        strides.append(len(p) + strides[-1])
        npts += len(p)
        all_pts.extend(p)
        all_times.extend(t)
    # print(f'all_pts = \n{all_pts}')

    print(f'npts={npts}, strides={strides}, len(pathlines)={len(pathlines)}, number={args.number}')

    # pathlines = np.array(pathlines).flatten().reshape((-1, 3))

    poly = vtk_dataset.make_points(all_pts)
    poly = vtk_dataset.add_scalars(poly, all_times)
    line_ids = [ list(range(strides[i], strides[i+1])) for i in range(args.number) ]
    poly = vtk_dataset.add_polylines(poly, line_ids)

    min_t = np.min(all_times)
    max_t = np.max(all_times)
    ctf = vtk_colors.make_colormap('viridis', [min_t, max_t])
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOn()
    mapper.SetLookupTable(ctf)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0, 0, 1)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.Initialize()
    window.Render()
    interactor.Start()
            


