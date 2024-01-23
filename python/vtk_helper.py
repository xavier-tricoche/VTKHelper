import vtk
import os
#
#    Useful functions when working with VTK
#

def connect(_from, _to):
    if isinstance(_from, vtk.vtkAlgorithm) and isinstance(_to, vtk.vtkAlgorithm):
        _to.SetInputConnection(_from.GetOutputPort())
    elif isinstance(_from, vtk.vtkDataSet) and isinstance(_to, vtk.vtkAlgorithm):
        _to.SetInputData(_from)
    else:
        raise ValueError(f'Unrecognized combination {type(_from)} -> {type(_to)} in plug')
    return _to

def create_mapper(input):
    if not isinstance(input, vtk.vtkAlgorithm) and not isinstance(vtk.vtkDataSet):
        raise ValueError(f'Invalid type ({type(intpu)}) in input of create_mapper')
    if isinstance(input, vtk.vtkAlgorithm):
        if isinstance(input, vtk.vtkPolyDataAlgorithm):
            mapper = vtk.vtkPolyDataMapper()
            return plug(input, mapper)
        elif isinstance(input, vtk.DataSetAlgorithm):
            mapper = vtk.vtkDataSetMapper()
            return plug(input, mapper)
        else:
            raise ValueError(f'Unrecognized algorithm type ({type(input)}) in create_mapper')
    elif isinstance(input, vtk.vtkPolyData):
        mapper = vtk.vtkPolyDataMapper()
        return plug(input, mapper)
    elif isinstance(input, vtk.vtkDataSet):
        mapper = vtk.vtkDataSetMapper()
        return plug(input, mapper)
    else:
        raise ValueError(f'Unrecognized dataset type ({type(input)}) in create_mapper')

def replace_extension(filename, ext):
    if ext[0] == '.':
        ext = ext[1:]
    name, removed = os.path.splitext(filename)
    return os.path.join(name, ext)

def correct_reader(filename, _ext=None):
    if _ext is not None:
        filename = replace_extension(filename, _ext)
    name, ext = os.path.splitext(filename)
    reader = None
    if ext == '.vtk' or ext == '.VTK':
        reader = vtk.vtkDataSetReader()
    elif ext == '.vti' or ext == '.VTI':
        reader = vtk.vtkXMLImageDataReader()
    elif ext == '.vtu' or ext == '.VTU':
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif ext == '.vtp' or ext == '.VTP':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtr' or ext == '.VTR':
        reader = vtk.vtkXMLRectilinearGridReader()
    else:
        print('unrecognized vtk filename extension: ', ext)
        return None
    reader.SetFileName(filename)
    return reader

def correct_writer(filename, _ext=None):
    if _ext is not None:
        filename = replace_extension(filename, _ext)
    name, ext = os.path.splitext(filename)
    writer = None
    if ext == '.vtk' or ext == '.VTK':
        writer = vtk.vtkDataSetWriter()
    elif ext == '.vti' or ext == '.VTI':
        writer = vtk.vtkXMLImageDataWriter()
    elif ext == '.vtu' or ext == '.VTU':
        writer = vtk.vtkXMLUnstructuredGridWriter()
    elif ext == '.vtp' or ext == '.VTP':
        writer = vtk.vtkXMLPolyDataWriter()
    elif ext == '.vtr' or ext == '.VTR':
        writer = vtk.vtkXMLRectilinearGridWriter()
    else:
        print('unrecognized vtk filename extension: ', ext)
        return None
    writer.SetFileName(filename)
    return writer
