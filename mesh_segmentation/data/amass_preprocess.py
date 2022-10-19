import sys
from argparse import ArgumentParser
from os import path as osp
from pathlib import Path

import h5py
import trimesh
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


if __name__ == '__main__':
    dtype = 'val'
    with h5py.File(osp.join(DataPaths.POINT_CLOUDS_DB, 'segmentation', f'test.h5'), "r") as f:
        segmentation = np.array(f['segmentations'])[2]
    with h5py.File(osp.join(DataPaths.POINT_CLOUDS_DB, 'smpl_like', f'{dtype}.h5'), "r") as f:
        pcs = np.array(f['point_cloud'])
        faces = np.array(f['faces'])
    n = len(str(len(pcs)))

    for i, pc in tqdm(enumerate(pcs)):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_point_cloud(pc.T[0], pc.T[1], pc.T[2], segmentation, axis=ax, show=True, title=f'{dtype} {i}')