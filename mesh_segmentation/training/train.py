from time import sleep

import numpy as np
from tqdm import tqdm
from os import path as osp

import torch

from torch_geometric.data import DataLoader

import sys

from mesh_segmentation.model.segmentation_model import MeshSeg
from mesh_segmentation.utils.segmentation_utils import get_best_model, load_data

sys.path.append(r'/mnt/DS_SHARED/users/talb/projects/3d_avatar_generation')
sys.path.append(r'/mnt/DS_SHARED/users/talb/projects/3d_avatar_generation/src')

logger = init_logger('train_logger', log_path=r'/mnt/DS_SHARED/users/talb/projects/3d_avatar_generation')

def train(net, train_data, optimizer, loss_fn, device):
    """Train network on training dataset."""
    net.train()
    cumulative_loss = 0.0
    n = len(train_data)
    for i, data in enumerate(train_data):
        data = data.to(device)
        optimizer.zero_grad()
        out = net(data)
        loss = loss_fn(out, data.segmentation_labels.squeeze())
        loss.backward()
        cumulative_loss += loss.item()
        optimizer.step()
        if i % 100 == 0:
            logger.info(f'{i}/{n}')
    return cumulative_loss / n

def accuracy(predictions, gt_seg_labels):
    """Compute accuracy of predicted segmentation labels.

    Parameters
    ----------
    predictions: [|V|, num_classes]
        Soft predictions of segmentation labels.
    gt_seg_labels: [|V|]
        Ground truth segmentations labels.
    Returns
    -------
    float
        Accuracy of predicted segmentation labels.
    """
    predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)
    if predicted_seg_labels.shape != gt_seg_labels.shape:
        raise ValueError("Expected Shapes to be equivalent")
    correct_assignments = (predicted_seg_labels == gt_seg_labels).sum()
    num_assignemnts = predicted_seg_labels.shape[0]
    return float(correct_assignments / num_assignemnts)


def evaluate_performance(dataset, net, device):
    """Evaluate network performance on given dataset.

    Parameters
    ----------
    dataset: DataLoader
        Dataset on which the network is evaluated on.
    net: torch.nn.Module
        Trained network.
    device: str
        Device on which the network is located.

    Returns
    -------
    float:
        Mean accuracy of the network's prediction on
        the provided dataset.
    """
    prediction_accuracies = []
    n = len(dataset)
    for i, data in enumerate(dataset):
        data = data.to(device)
        predictions = net(data)
        prediction_accuracies.append(accuracy(predictions, data.segmentation_labels))
        if i % 100 == 0:
            logger.info(f'{i}/{n}')
    return sum(prediction_accuracies) / len(prediction_accuracies)

@torch.no_grad()
def test(net, train_data, val_data, test_data, device):
    net.eval()
    logger.info(f'Train evaluation')
    train_acc = evaluate_performance(train_data, net, device)
    logger.info(f'Validation evaluation')
    val_acc = evaluate_performance(val_data, net, device)
    logger.info(f'Test evaluation')
    test_acc = evaluate_performance(test_data, net, device)
    return train_acc, val_acc, test_acc


if __name__ == '__main__':
    checkpoints_dir = r'/mnt/DS_SHARED/users/talb/projects/3d_avatar_generation/checkpoints'
    data_dir = '/mnt/DS_SHARED/users/talb/data/3d_segmentation/AMASS'
    resume = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_params = dict(
        in_features=3,
        encoder_features=16,
        conv_channels=[32, 64, 128, 64],
        encoder_channels=[16],
        decoder_channels=[32],
        num_classes=14,
        num_heads=14,
        apply_batch_norm=True,
    )
    net, i = get_best_model(checkpoints_dir, model_params, device) if resume else MeshSeg(**model_params).to(device), 0
    train_loader, val_loader, test_loader = load_data(data_dir)

    lr = 0.001
    num_epochs = 100
    best_acc = 0.0

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    with tqdm(range(i, num_epochs + i), unit="Epoch") as tepochs:
        for epoch in tepochs:
            train_loss = train(net, train_loader, optimizer, loss_fn, device)
            train_acc, val_acc, test_acc = test(net, train_loader, val_loader, test_loader, device)

            tepochs.set_postfix(
                train_loss=train_loss,
                train_accuracy=100 * train_acc,
                val_accuracy=100 * val_acc,
                test_accuracy=100 * test_acc,
            )
            sleep(0.1)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(net.state_dict(), osp.join(checkpoints_dir, f"{epoch}.pt"))

            logger.info(f'Epoch {epoch} - Train Acc: {np.round(100 * train_acc, 3)}, '
                        f'Val Acc: {np.round(100 * val_acc, 3)}, '
                        f'Test Acc: {np.round(100 * test_acc, 3)}, '
                        f'Best Acc: {np.round(100 * best_acc, 3)}')