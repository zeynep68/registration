import pli
from tqdm import tqdm
import os
import re
import click

from atlasmpi import MPI
comm = MPI.COMM_WORLD


@click.command()
@click.argument("files", type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=-1)
@click.option("-o", "--out_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default="./")
@click.option("--chunk_size", default=256, type=int)
@click.option("--num_levels", default=11, type=int)
def convert_folder(files, out_dir, chunk_size, num_levels):
    rank = comm.Get_rank()
    size = comm.size

    file_list = [f for i, f in enumerate(files) if (f.endswith(".nii.gz") and i % size == rank)]

    for f in tqdm(file_list, desc=f"Rank {rank}"):
        h5_file = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(f))[0]}.h5")
        try:
            section = pli.data.Section(f)
        except RuntimeError:
            print(f, "could not be loaded. Does it exist?")
            continue
        p = re.compile('Vervet1818([a-z]+)_.*_s([0-9]+)_.*_([A-Za-z]+).nii.gz')
        g = p.match(os.path.basename(f))
        section.roi = g.group(1)
        section.id = int(g.group(2))
        section.modality = g.group(3)
        section.brain_id = 'PE-2012-00102-V'
        section.source = ["PM", "LMP1"]
        section.to_hdf5(h5_file, pyramid=True, chunk_size=chunk_size, num_levels=num_levels)


if __name__ == "__main__":
    convert_folder()
