"""
Parallel implementation of marching cubes algorithm.
"""

import click
import sys

# Add code path
sys.path.insert(0, "code/")

import marching


# noinspection PyShadowingNames
@click.command()
@click.argument("volume", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("mesh", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--volume-prefix", default="volume", help="Defaults to \"volume\".")
@click.option("--affine-prefix", default="affine", help=" Defaults to \"affine\".")
@click.option("--chunk-size", default=256, type=int, help="Chunk size to use during parallel processing.")
@click.option("--level", default=1, type=int, help="Level to extract. Default is 1.")
@click.option("--pad", default=False, type=bool, help="If to pad the volume for closed surfaces")
def marching_cubes(volume, mesh, volume_prefix, affine_prefix, chunk_size, level, pad):
    return marching.marching_cubes(
        volume,
        mesh,
        volume_prefix,
        affine_prefix,
        chunk_size,
        level,
        pad
    )


if __name__ == "__main__":
    marching_cubes()
