################ Request information

import h5py as h5

path = "/p/fastdata/pli/Private/oberstrass1/datasets/vervet1818-3d/data/aa/volume/ntrans/ntrans_5.h5"


with h5.File(path, 'r') as f:
    shape = f['volume'].shape[::-1]
    spacing = f['volume'].attrs['spacing'][::-1]

# Code for 'RequestInformation Script'.
executive = self.GetExecutive()
outInfo = executive.GetOutputInformation(0)
# we assume the dimensions are (48, 62, 42).

outInfo.Set(executive.WHOLE_EXTENT(), 0, shape[0] - 1, 0, shape[1] - 1, 0, shape[2] - 1)
outInfo.Set(vtk.vtkDataObject.SPACING(), *spacing)

################ Script

import h5py as h5
import numpy as np

path = "/p/fastdata/pli/Private/oberstrass1/datasets/vervet1818-3d/data/aa/volume/ntrans/ntrans_5.h5"

with h5.File(path, 'r') as f:
    volume = f['volume'][:]

dims = volume.shape[::-1]

output.SetExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)

volume = np.clip(volume.ravel()[..., None] * 255, 0, 255).astype(int)
alpha = np.full(volume.shape[0], 255, dtype=int)[..., None]
alpha[np.sum(volume, axis=-1) == 0] = 0

volume = np.hstack([volume, volume, volume, alpha])

output.PointData.append(volume, "RGB")
output.PointData.SetActiveScalars("RGB")
