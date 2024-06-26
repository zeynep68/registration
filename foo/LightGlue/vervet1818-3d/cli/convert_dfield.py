import os.path
from glob import glob
import click

import SimpleITK as sitk


@click.command()
@click.option("-i", "--input", type=str)
@click.option("-o", "--out_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default="./", help="Directory to store converted files.")
def cli(input, out_dir):

    # Distributed
    from atlasmpi import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.size

    for i, f in enumerate(glob(input)):
        if i % size == rank:
            print(f"Load {f}")

            tf = sitk.DisplacementFieldTransform(sitk.ReadTransform(f))
            out_fname = os.path.splitext(os.path.basename(f))[0] + ".mat"
            out_path = os.path.join(out_dir, out_fname)
            print(f"Save {out_path}")
            sitk.WriteTransform(tf, out_path)

            del tf


if __name__ == "__main__":
    cli()
