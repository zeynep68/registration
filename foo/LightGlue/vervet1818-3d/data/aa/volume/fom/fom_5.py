################ Request information

import h5py as h5

path = "/p/fastdata/pli/Private/oberstrass1/datasets/PE-2021-00981-H-blockface/data/volume/bf_normalized/bf_3.h5"


with h5.File(path, 'r') as f:
    shape = f['volume'].shape[:-1][::-1]
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

path = "/p/fastdata/pli/Private/oberstrass1/datasets/PE-2021-00981-H-blockface/data/volume/bf_normalized/bf_3.h5"

with h5.File(path, 'r') as f:
    volume = f['volume'][:]

dims = volume.shape[:-1][::-1]

output.SetExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)

# volume = volume.transpose(3, 0, 1, 2)
volume = volume.reshape(-1, volume.shape[-1])
alpha = np.full(volume.shape[0], 255, dtype=int)[..., None]
alpha[np.sum(volume, axis=-1) == 0] = 0
print(volume.shape, alpha.shape)
volume = np.hstack([volume, alpha])
print(volume.shape)

# vtk_array = numpy_support.numpy_to_vtk(volume)
output.PointData.append(volume, "RGB")
output.PointData.SetActiveScalars("RGB")

