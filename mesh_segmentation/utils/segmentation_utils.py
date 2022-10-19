from os import path as osp
import os

import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.transforms import Compose, FaceToEdge
from torch_geometric.data import DataLoader

from mesh_segmentation.data.dataset import NormalizeUnitSphere, SegmentationFaust
from mesh_segmentation.model.segmentation_model import MeshSeg


def load_model(model_params, path_to_checkpoint, device):
    try:
        model = MeshSeg(**model_params)
        model.load_state_dict(
            torch.load(str(path_to_checkpoint)),
            strict=True,
        )
        model.to(device)
        return model
    except RuntimeError as err_msg:
        raise ValueError(
            f"Given checkpoint {str(path_to_checkpoint)} could"
            f" not be loaded. {err_msg}"
        )

def get_best_model(checkpoints_dir, model_params, device):
    checkpoints = [int(osp.splitext(f)[0]) for f in os.listdir(checkpoints_dir)]
    i = max(checkpoints)
    model_path = osp.join(checkpoints_dir, f'{i}.pt')
    model = load_model(
        model_params,
        model_path,
        device,
    )
    return model, i

@torch.no_grad()
def visualize_prediction(net, data, device, map_seg_id_to_color):
    """Visualization of predicted segmentation mask."""
    def _map_seg_label_to_color(seg_ids, map_seg_id_to_color):
        return torch.tensor([map_seg_id_to_color[int(seg_ids[idx])] for idx in range(seg_ids.shape[0])])
        # return torch.vstack(
        #     [map_seg_id_to_color[int(seg_ids[idx])] for idx in range(seg_ids.shape[0])]
        # )

    data = data.to(device)
    predictions = net(data)
    predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)
    mesh_colors = _map_seg_label_to_color(torch.squeeze(predicted_seg_labels), map_seg_id_to_color)
    segmented_mesh = trimesh.base.Trimesh(
        vertices=data.x.cpu().numpy(),
        faces=data.face.t().cpu().numpy(),
        process=False,
    )
    segmented_mesh.visual.vertex_colors = mesh_colors.cpu().numpy()
    return segmented_mesh

def plot(mesh):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.faces, Z=mesh.vertices[:, 2])
    p.set_fc(mesh.visual.face_colors / 255)
    plt.show()

def load_data(root, return_dataset=False):
    pre_transform = Compose([FaceToEdge(remove_faces=False), NormalizeUnitSphere()])

    train_data = SegmentationFaust(
        root=root,
        mode='train',
        pre_transform=pre_transform,
    )
    val_data = SegmentationFaust(
        root=root,
        mode='val',
        pre_transform=pre_transform,
    )
    test_data = SegmentationFaust(
        root=root,
        mode='test',
        pre_transform=pre_transform,
    )

    train_loader = DataLoader(train_data, shuffle=True)
    val_loader = DataLoader(val_data, shuffle=False)
    test_loader = DataLoader(test_data, shuffle=False)
    if return_dataset:
        return train_loader, val_loader, test_loader, train_data, val_data, test_data
    else:
        return train_loader, val_loader, test_loader