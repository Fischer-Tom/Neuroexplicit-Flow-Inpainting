from os.path import join
import os
import torch
from imagelib.inout import read
from torchvision import transforms
import numpy as np
import cv2


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]
    return input_string


class KITTIDataset:
    def __init__(
        self,
        img_dir,
        density,
        presmooth,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                transforms.CenterCrop((374, 1248)),
            ]
        ),
        mode="train",
        type="IP",
    ):
        self.mode = mode
        self.type = type
        self.transform = transform
        self.flow_transform = transforms.CenterCrop((374, 1248))
        self.density = density
        self.root_path = f"/share_chairilg/data/KITTI/depth_completion/{mode}/"
        self.root_image_path = "/share_chairilg/data/KITTI/raw/"
        self.data = {"image0": [], "sparse": [], "depth": []}
        self.read_data()

    def read_data(self):
        for drive in os.listdir(self.root_path):
            if os.path.isdir(os.path.join(self.root_path, drive)):
                current_path = os.path.join(
                    self.root_path, drive, "proj_depth/groundtruth/image_02"
                )
                current_path2 = os.path.join(
                    self.root_path, drive, "proj_depth/groundtruth/image_03"
                )
                depth_maps = [
                    os.path.join(current_path, im) for im in os.listdir(current_path)
                ]
                depth_maps += [
                    os.path.join(current_path2, im) for im in os.listdir(current_path2)
                ]
                image_path = os.path.join(
                    self.root_image_path, drive[:10], drive, "image_02/data"
                )
                images = [
                    os.path.join(image_path, im) for im in os.listdir(current_path)
                ]
                image_path2 = os.path.join(
                    self.root_image_path, drive[:10], drive, "image_03/data"
                )
                images += [
                    os.path.join(image_path2, im) for im in os.listdir(current_path)
                ]
                sparse_depth = [
                    path.replace("groundtruth", "velodyne_raw") for path in depth_maps
                ]
                self.data["depth"] += depth_maps
                self.data["image0"] += images
                self.data["sparse"] += sparse_depth

    def __len__(self):
        return len(self.data["image0"])

    def __getitem__(self, idx):
        im0 = cv2.imread(self.data["image0"][idx])
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        sparse_depth = cv2.imread(self.data["sparse"][idx], cv2.IMREAD_UNCHANGED)
        sparse_depth = sparse_depth.astype(np.float32) / 256.0
        depth = cv2.imread(self.data["depth"][idx], cv2.IMREAD_UNCHANGED)
        depth = (depth.astype(np.float32)) / 256.0
        occ_mask = (depth > 0).astype(np.float32)
        sparse_mask = (sparse_depth > 0).astype(np.float32)
        if self.transform:
            im0 = self.transform(np.array(im0[:, :, :3]))
            depth = torch.Tensor(np.array(depth))
            occ_mask = torch.Tensor(np.array(occ_mask))
            sparse_mask = torch.Tensor(np.array(sparse_mask))
            sparse_depth = torch.Tensor(np.array(sparse_depth))
        if self.flow_transform:
            depth = self.flow_transform(depth)
            occ_mask = self.flow_transform(occ_mask)
            sparse_mask = self.flow_transform(sparse_mask)
            sparse_depth = self.flow_transform(sparse_depth)
        depth = depth[None, 118:, :]
        im0 = im0[..., 118:, :]
        occ_mask = occ_mask[None,118:, :]
        sparse_mask = sparse_mask[None,118:, :]
        sparse_depth = sparse_depth[None,118:, :]
        return (
            im0,
            im0,
            sparse_mask,
            depth,
            sparse_depth,
            occ_mask,
        )
