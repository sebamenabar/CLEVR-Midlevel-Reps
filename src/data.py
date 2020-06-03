import os
import glob
import Imath
import numpy as np
from PIL import Image
import OpenEXR as exr
from skimage.transform import resize
from easydict import EasyDict as edict

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset


import glob
from torch.utils import data
import OpenEXR as exr
import Imath
from skimage.transform import resize


def readEXR(filename):
    """Read color + depth data from EXR image file.
    
    Parameters
    ----------
    filename : str
        File path.
        
    Returns
    -------
    img : Float matrix
    Z : Depth buffer in float32 format or None if the EXR file has no Z channel.
    """

    exrfile = exr.InputFile(filename)
    header = exrfile.header()

    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    # convert all channels in the image to numpy arrays
    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.frombuffer(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    colorChannels = (
        ["R", "G", "B", "A"] if "A" in header["channels"] else ["R", "G", "B"]
    )
    img = np.concatenate(
        [channelData[c][..., np.newaxis] for c in colorChannels], axis=2
    )

    # linear to standard RGB
    # img[..., :3] = np.where(img[..., :3] <= 0.0031308,
    #                       12.92 * img[..., :3],
    #                        1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055)

    # sanitize image to be in range [0, 1]
    # img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))

    Z = None if "Z" not in header["channels"] else channelData["Z"]

    return img, Z


def default_loader(path):
    return Image.open(path).convert("RGB")


def exr_loader(path):
    return readEXR(path)[0]


def get_depth(img_fp):
    return 1 - get_exr_replace(img_fp, "depths")[:, :, [0]]


def get_normal(img_fp):
    return get_exr_replace(img_fp, "normals")


def get_shadeless(img_fp):
    return default_loader(img_fp.replace("images", "shadeless"))


def get_exr_replace(img_fp, folder_replace):
    return exr_loader(
        img_fp.replace("images", folder_replace).replace(".png", "_0001.exr")
    )


class CLEVRMidrepsDataset(data.Dataset):
    get_fns = {
        "depths": get_depth,
        "normals": get_normal,
        "shadeless": get_shadeless,
        "autoencoder": lambda x: None,
    }
    std_img_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    std_midreps_transforms = T.Compose(
        [
            lambda img: resize(img, (224, 224), preserve_range=True).transpose(2, 0, 1),
            torch.from_numpy,
        ]
    )

    def __init__(
        self,
        base_dir,
        split="train",
        match="*.png",
        transform=None,
        loader=default_loader,
        midreps=[],
        midreps_transform=None,
    ):
        self.midreps = midreps
        self.transform = transform
        self.midreps_transform = midreps_transform
        self.loader = loader
        self.base_dir = base_dir
        self.files = sorted(glob.glob(os.path.join(base_dir, "images", split, match)))

    def __getitem__(self, index, midreps=None):
        midreps = midreps if midreps is not None else self.midreps
        img_fp = self.files[index]
        img = self.loader(img_fp)
        data = {midrep: self.get_fns[midrep](img_fp) for midrep in midreps}
        if "autoencoder" in data:
            del data["autoencoder"]

        if self.transform is not None:
            img = self.transform(img)
        if self.midreps_transform is not None:
            if isinstance(self.midreps_transform, dict):
                default_transform = self.midreps_transform.get("default", None)
                data = {
                    k: self.midreps_transform.get(k, default_transform)(v)
                    for k, v in data.items()
                }
            else:
                data = {k: self.midreps_transform(v) for k, v in data.items()}

        return img, edict(data)

    def __len__(self):
        return len(self.files)
