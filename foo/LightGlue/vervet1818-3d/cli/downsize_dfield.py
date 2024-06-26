import os.path
from glob import glob
import click

import SimpleITK as sitk

from vervet1818_3d.utils.transforms import downscale_image


@click.command()
@click.option("-i", "--input", type=str)
@click.option("-o", "--out_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default="./", help="Directory to store converted files.")
@click.option("-d", "--downscale", type=float)
def cli(input, out_dir, downscale):

    # Distributed
    from atlasmpi import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.size

    for i, f in enumerate(glob(input)):
        if i % size == rank:
            print(f"Load {f}")
            field_image = sitk.ReadImage(f)
            small_image = downscale_image(field_image, downscale, sitk.sitkLinear)
            small_transform = sitk.DisplacementFieldTransform(small_image)
            out_fname = os.path.splitext(os.path.basename(f))[0] + ".hdf"
            out_path = os.path.join(out_dir, out_fname)
            print(f"Save {out_path}")
            sitk.WriteTransform(small_transform, out_path)

            del field_image
            del small_image
            del small_transform


if __name__ == "__main__":
    cli()
