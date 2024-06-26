from typing import List
import os

import click
from tqdm import tqdm

import numpy as np
import h5py as h5

import vtk
from vtk.util import numpy_support

from atlasmpi import MPI
comm = MPI.COMM_WORLD


def write_vti(vti_file: str, volume: np.ndarray, spacing: List[int], array_type=vtk.VTK_UNSIGNED_CHAR):

    vtk_data = numpy_support.numpy_to_vtk(num_array=volume.transpose((2, 1, 0)).ravel(), deep=True, array_type=array_type)
    
    vtk_volume = vtk.vtkStructuredPoints()
    vtk_volume.SetDimensions(*volume.shape)
    vtk_volume.SetOrigin(0, 0, 0)
    vtk_volume.SetSpacing(*spacing)
    vtk_volume.GetPointData().SetScalars(vtk_data)
    
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(vti_file)
    writer.SetInputData(vtk_volume)
    writer.Write()


@click.command()
@click.argument("files", type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=-1)
@click.option("-o", "--out_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default="./")
@click.option("--file_per_value", is_flag=True, help="Flag if to create a new file per value")
def convert_folder(files, out_dir, file_per_value):
    rank = comm.Get_rank()
    size = comm.size

    file_list = [f for i, f in enumerate(files) if i % size == rank]

    for file in tqdm(file_list, desc=f"Rank {rank}"):
        try:
            with h5.File(file, 'r') as f:
                volume = f['volume'][:]
                spacing = f['volume'].attrs['spacing']
        except RuntimeError:
            print(f, "could not be loaded. Does it exist?")
            continue
        
        if file_per_value:
            print("Create one file per value")
            for v in np.unique(volume):
                vti_file = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(file))[0]}_{v}.vti")
                mask_volume = volume == v
                write_vti(vti_file, mask_volume, spacing)
        else:
            vti_file = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(file))[0]}.vti")
            write_vti(vti_file, volume, spacing)


if __name__ == "__main__":
    convert_folder()
