import numpy as np
import torch
import h5py
import trimesh
import sys
import os
from os import path as osp

from torch_geometric.transforms import Compose, FaceToEdge
from tqdm import tqdm

from torch_geometric.data import Data

from mesh_segmentation.data.dataset import NormalizeUnitSphere
from mesh_segmentation.utils.segmentation_utils import get_best_model


class Segmentor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_params = dict(
            in_features=3,
            encoder_features=16,
            conv_channels=[32, 64, 128, 64],
            encoder_channels=[16],
            decoder_channels=[32],
            num_classes=14,
            num_heads=14,
            apply_batch_norm=True,
        )
        self.model, self.epoch_id = get_best_model(self.model_params, self.device)
        self.pre_transform = Compose([FaceToEdge(remove_faces=False), NormalizeUnitSphere()])

    def predict(self, mesh):
        vertices = torch.from_numpy(mesh.vertices).to(torch.float)
        faces = torch.from_numpy(mesh.faces)
        faces = faces.t().to(torch.long).contiguous()
        data = Data(x=vertices, face=faces).to(self.device)
        predictions = self.model(self.pre_transform(data))
        predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)
        return np.squeeze(predicted_seg_labels.detach().cpu().numpy())

if __name__ == '__main__':
    data_path = r'/mnt/DS_SHARED/users/talb/data/AMASS/h5'
    data = 'test'
    s = Segmentor()

    with h5py.File(osp.join(data_path, 'segmentation', f'{data}.h5'), "r") as f:
        pcs = np.array(f['point_cloud'])
        faces = np.array(f['faces'])
        skeletons = np.array(f['skeleton'])
        segments = np.array(f['segmentations'])

    for i, (pc, face, skeleton, org_segments) in tqdm(enumerate(zip(pcs, faces, skeletons, segments))):
        mesh = trimesh.Trimesh(vertices=pc, faces=faces)
        predicted_segments = s.predict(mesh)
