from collections import namedtuple

import math
from tqdm import tqdm
from typing import Tuple
from glob import glob
import os
import re
import click

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import h5py as h5

from pli.data import Section

from pli_styles.modality.inclination import corr_factor_transmittance_weighted, inclination_from_retardation
from pli_styles.modality.fom import hsv_fom

from vervet1818_3d.resnet_wider import resnet50x1

# Distributed
from atlasmpi import MPI

comm = MPI.COMM_WORLD


Coord = namedtuple("Coord", ('x', 'y'))


def generate_fom(trans, dir, ret):
    corr = corr_factor_transmittance_weighted(
        trans,
        t_M=0.32,  # 0.23
        t_c=0.65,  # 0.65
        r_ref_wm=0.96, # 0.96
        r_ref_gm=0.16, # 0.16
        median_kernel_size=3
    )
    incl = inclination_from_retardation(
        ret,
        corr
    )
    fom = hsv_fom(
        np.rad2deg(dir),
        incl,
        saturation_min=0,
        inclination_scale='cosinus'
    )
    return fom


class SectionDataset(torch.utils.data.Dataset):

    def __init__(self, trans_file, dir_file, ret_file, patch_shape, out_shape, ram=True, norm_trans=None, norm_ret=None):
        # Expands the dataset to size input by repeating the provided ROIs
        # rois is a list of dicts with entries 'mask', 'ntrans', 'ret' and 'dir'
        super().__init__()

        # Scale to size that was used in Imagenet training
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
        ])

        self.ram = ram
        self.trans_section_mod = Section(path=trans_file)
        self.dir_section_mod = Section(path=dir_file)
        self.ret_section_mod = Section(path=ret_file)
        if ram:
            print("Load sections to RAM...")
            self.trans_section = np.array(self.trans_section_mod.image)
            self.dir_section = np.array(self.dir_section_mod.image)
            self.ret_section = np.array(self.ret_section_mod.image)
            print("All sections loaded to RAM")
        else:
            print("Do not load sections to RAM")
            self.trans_section = self.trans_section_mod.image
            self.dir_section = self.dir_section_mod.image
            self.ret_section = self.ret_section_mod.image

        if norm_trans is None:
            if self.trans_section_mod.norm_value is not None:
                self.norm_trans = self.trans_section_mod.norm_value
            else:
                print("[WARNING] Did not find a normalization value for Transmittance")
                self.norm_trans = 1.0
        else:
            self.norm_trans = norm_trans
            print(f"Normalize Transmittance by value of {self.norm_trans}")
        if norm_ret is None:
            self.norm_ret = 1.0
        else:
            self.norm_ret = norm_ret
            print(f"Normalize Retardation by value of {self.norm_ret}")
        self.brain_id = self.trans_section_mod.brain_id
        self.section_id = self.trans_section_mod.id
        self.section_roi = self.trans_section_mod.roi

        assert (patch_shape[0] - out_shape[0]) % 2 == 0  # Border symmetric
        assert (patch_shape[1] - out_shape[1]) % 2 == 0  # Border symmetric
        self.patch_shape = patch_shape
        self.out_shape = out_shape
        self.border = ((patch_shape[0] - out_shape[0]) // 2, (patch_shape[1] - out_shape[1]) // 2)
        self.shape = self.trans_section.shape

        self.coords = [Coord(x=x, y=y) for x in np.arange(0, self.shape[1], out_shape[1]) for y in
                       np.arange(0, self.shape[0], out_shape[0])]

    def __getitem__(self, i):
        x = self.coords[i].x
        y = self.coords[i].y

        b_y = self.border[0]
        b_x = self.border[1]

        pad_y_0 = max(b_y - y, 0)
        pad_x_0 = max(b_x - x, 0)
        pad_y_1 = max(y + (self.patch_shape[0] - b_y) - self.shape[0], 0)
        pad_x_1 = max(x + (self.patch_shape[1] - b_x) - self.shape[1], 0)

        trans_crop = np.array(
            self.trans_section[max(0, y - b_y):min(self.shape[0], y + self.patch_shape[0] - b_y),
            max(0, x - b_x):min(self.shape[1], x + self.patch_shape[1] - b_x)],
            dtype=np.float32
        ) / self.norm_trans
        ret_crop = np.array(
            self.ret_section[max(0, y - b_y):min(self.shape[0], y + self.patch_shape[0] - b_y),
            max(0, x - b_x):min(self.shape[1], x + self.patch_shape[1] - b_x)],
            dtype=np.float32
        ) / self.norm_ret
        dir_crop = np.deg2rad(
            self.dir_section[max(0, y - b_y):min(self.shape[0], y + self.patch_shape[0] - b_y),
            max(0, x - b_x):min(self.shape[1], x + self.patch_shape[1] - b_x)],
            dtype=np.float32
        )

        fom_crop = generate_fom(trans_crop, dir_crop, ret_crop)

        fom_crop = np.pad(fom_crop, ((pad_y_0, pad_y_1), (pad_x_0, pad_x_1), (0, 0)), mode='constant', constant_values=0.0)
        
        fom_crop = self.transforms(fom_crop)
       
        return {'x': x, 'y': y, 'crop': fom_crop}

    def __len__(self):
        return len(self.coords)


def get_files(
        trans: str,
        dir: str,
        ret: str,
        out: str,
        rank: int = 0,
        size: int = 1
):
    print(trans)
    trans_files = sorted(glob(trans))
    dir_files = sorted(glob(dir))
    ret_files = sorted(glob(ret))

    if os.path.isdir(out):
        ft_files = []
        for d_f in dir_files:
            d_fname = os.path.splitext(os.path.basename(d_f))[0]
            d_base = os.path.splitext(d_fname)[0]
            ft_file = re.sub("direction", "Features", d_base, flags=re.IGNORECASE)
            if "Features" not in ft_file:
                ft_file += "_Features.h5"
            else:
                ft_file += ".h5"
            ft_files.append(os.path.join(out, ft_file))
    else:
        ft_files = [out]

    for i, (trans_file, dir_file, ret_file, ft_file) \
            in enumerate(zip(trans_files, dir_files, ret_files, ft_files)):
        if i % size == rank:
            if not os.path.isfile(ft_file):
                yield trans_file, dir_file, ret_file, ft_file
            else:
                print(f"{ft_file} already exists. Skip.")


def create_features(
        encoder: torch.nn.Module,
        section_loader: DataLoader,
        h_size: int,
        out_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        rank: int
):
    print("Initialize output featuremaps...")
    h_features = np.zeros((*out_size, h_size), dtype=np.float32)
    
    def get_outputs(batch, network):
        with torch.no_grad():
            network.eval()
            h = network(
                batch['crop'].to(network.device),
            )
        return {'x': batch['x'], 'y': batch['y'], 'h': h}

    def transfer(batch, network):
        b = get_outputs(batch, network)
        for x, y, h in zip(b['x'], b['y'], b['h']):
            try:
                h_features[y // stride[0], x // stride[1]] = h.cpu().numpy()
            except:
                raise Exception(f"ERROR creating mask at x={x}, y={y}, shape={h_features.shape}")

    print("Start feature generation...")
    for batch in tqdm(section_loader, desc=f"Rank {rank}"):
        transfer(batch, encoder)

    return h_features


def save_features(
        h_features: np.ndarray,
        ft_file: str,
        spacing: Tuple[float, ...] = (1.0, 1.0),
        origin: Tuple[float, ...] = (0.0, 0.0),
        dtype: str = None,
):
    print("Save features...")
    with h5.File(ft_file, "w") as f:
        feature_group = f.create_group("Features")
        dset_h = feature_group.create_dataset(f"{h_features.shape[-1]}", data=h_features.transpose(2, 0, 1), dtype=dtype)
        dset_h.attrs['spacing'] = spacing
        dset_h.attrs['origin'] = origin
    print(f"Featuremaps created at {ft_file}")


@click.command()
@click.option("--ckpt", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--trans", type=str)
@click.option("--dir", type=str)
@click.option("--ret", type=str)
@click.option("--out", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--norm_trans", type=float, default=None)
@click.option("--num_workers", type=int, default=0)
@click.option("--batch_size", type=int, default=1)
@click.option("--patch_size", type=int, default=128)
@click.option("--overlap", type=float, default=0.0)
@click.option("--ram", default=False, is_flag=True)
@click.option("--dtype", type=str, default=None)
def cli(ckpt, trans, dir, ret, out, norm_trans, num_workers, batch_size, patch_size, overlap, ram, dtype):
    rank = comm.Get_rank()
    size = comm.size

    if torch.cuda.is_available():
        available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        print(f"Found {len(available_gpus)} GPUs")
        device = available_gpus[rank % len(available_gpus)]
    else:
        device = 'cpu'
    print(f"Use device {device} on rank {rank}")

    # Create model
    encoder = resnet50x1()
    ckpt = torch.load(ckpt, map_location='cpu')
    encoder.load_state_dict(ckpt['state_dict'])

    encoder.device = device
    encoder.to(device)
    print(f"Model loaded on rank {rank}")

    patch_shape = (patch_size, patch_size)
    stride = (int((1 - overlap) * patch_size), int((1 - overlap) * patch_size))

    h_size = encoder.fc.in_features

    for trans_file, dir_file, ret_file, ft_file in get_files(trans, dir, ret, out, rank, size):
        print(f"Initialize DataLoader for {trans_file}, {dir_file}, {ret_file}")

        section_dataset = SectionDataset(
            trans_file=trans_file,
            dir_file=dir_file,
            ret_file=ret_file,
            patch_shape=patch_shape,
            out_shape=stride,
            ram=ram,
            norm_trans=norm_trans,
        )
        section_loader = DataLoader(
            section_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        out_size = tuple(math.ceil(s / stride[i]) for i, s in enumerate(section_dataset.shape))

        h_features = create_features(encoder, section_loader, h_size, out_size, stride, rank)

        spacing = tuple(stride[i] * s for i, s in enumerate(section_dataset.trans_section_mod.spacing))
        origin = section_dataset.trans_section_mod.origin

        save_features(h_features, ft_file, spacing, origin, dtype)


if __name__ == '__main__':
    cli()
