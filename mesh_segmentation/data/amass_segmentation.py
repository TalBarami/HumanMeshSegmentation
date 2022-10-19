from argparse import ArgumentParser

import h5py
import numpy as np
from os import path as osp
import trimesh
import torch
from pathlib import Path

from tqdm import tqdm


class Skeleton2Segments:
    def __init__(self, layout):
        self.layout = layout

    def _closest_point_and_distance(self, p, a, b):
        s = b - a
        w = p - a
        ps = np.dot(w, s)
        if ps <= 0:
            return a, np.linalg.norm(w)
        l2 = np.dot(s, s)
        if ps >= l2:
            closest = b
        else:
            closest = a + ps / l2 * s
        return closest, np.linalg.norm(p - closest)

    def _knn_classify(self, mesh, segments):
        _segments = []
        for segment, neighbors_d1 in zip(segments, mesh.vertex_neighbors):
            top = segment
            if segment != self.layout._parts['Body']:
                neighbors_d2 = [x for s in [mesh.vertex_neighbors[n] for n in neighbors_d1] for x in s if x not in neighbors_d1]
                neighbors = [(1, n) for n in neighbors_d1] + [(0.5, n) for n in neighbors_d2]
                votes = {}
                for s, n in neighbors:
                    if segments[n] not in votes.keys():
                        votes[segments[n]] = 0
                    votes[segments[n]] += s
                top = max(votes, key=votes.get)
            _segments.append(top)
        return np.array(segments)

    def segment_points(self, mesh, skeleton):
        segments = []
        for p, normal in zip(mesh.vertices, mesh.vertex_normals):
            candidates, parts = list(zip(*[(self._closest_point_and_distance(p, skeleton[a], skeleton[b]), n) for (a, b), n in self.layout.limbs().items()]))
            pts, distances = list(zip(*candidates))
            dots = [np.dot(normal, p - x) for x in pts]
            top_k = np.argsort(np.array(distances))
            argmin = top_k[0]
            if not self.layout.is_edge_limb(parts[argmin]):
                for i in top_k:
                    if dots[i] > 0:
                        argmin = i
                        break
            segments.append(parts[argmin])
        return self._knn_classify(mesh, np.array(segments))

def h5_to_ply(h5_root, out_root, dtype):
    out_dir = osp.join(out_root)
    init_directories(out_dir)

    with h5py.File(osp.join(h5_root, f'{dtype}.h5'), "r") as f:
        pcs = np.array(f['point_cloud'])
        faces = np.array(f['faces'])
    n = len(str(len(pcs)))

    for i, pc in tqdm(enumerate(pcs)):
        mesh = trimesh.Trimesh(vertices=pc, faces=faces)
        out_file = trimesh.exchange.ply.export_ply(mesh, encoding='ascii')
        out_path = osp.join(out_dir, f'{dtype}_{i:0{n}d}.ply')
        with open(out_path, 'wb+') as f:
            f.write(out_file)

def prepare_labels(segmentation_dir, out_path):
    labels = {}
    for dtype in ['train', 'val', 'test']:
        with h5py.File(osp.join(segmentation_dir, f'{dtype}.h5'), "r") as f:
            segmentation = np.array(f['segmentations'])
            labels[dtype] = segmentation
    with open(out_path, 'wb') as f:
        pickle.dump(labels, f)

def assign_segmentation_to_all():
    dtype = 'val'
    selected = 2
    with h5py.File(osp.join(DataPaths.POINT_CLOUDS_DB, 'segmentation', f'test.h5'), "r") as f:
        segmentation = np.array(f['segmentations'])[selected]
    with h5py.File(osp.join(DataPaths.POINT_CLOUDS_DB, 'smpl_like', f'{dtype}.h5'), "r") as f:
        pcs = np.array(f['point_cloud'])
        faces = np.array(f['faces'])
    n = len(str(len(pcs)))

def segment_all():
    parser = ArgumentParser()
    parser.add_argument("-data", "--data")
    args = vars(parser.parse_args())
    data = args['data']

    layout = SkeletonLayout()
    seg = Skeleton2Segments(layout)
    out_dir = osp.join(DataPaths.POINT_CLOUDS_DB, 'segmentation')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log.info(f'Preparing {data}')
    with h5py.File(osp.join(DataPaths.POINT_CLOUDS_DB, 'smpl_like', f'{data}.h5'), "r") as f:
        out = {k: np.array(v) for k,v in f.items()}
        pcs = out['point_cloud']
        faces = out['faces']
        skeletons = out['skeletons']
    out['segmentations'] = []
    for i in range(pcs.shape[0]):
        log.info(f'\t{i}/{pcs.shape[0]}')
        pc, skeleton = pcs[i], skeletons[i]
        mesh = trimesh.Trimesh(vertices=pc, faces=faces)
        segments = seg.segment_points(mesh, skeleton)
        out['segmentations'].append(segments)
        # fig = plt.figure(figsize=(15, 15))
        # ax = fig.add_subplot(221, projection='3d')
        # plot_3d_point_cloud(pc.T[0], pc.T[1], pc.T[2], segments, axis=ax, show=False, title='Before')
        # ax = fig.add_subplot(222, projection='3d')
        # plot_3d_point_cloud(pc.T[0], pc.T[1], pc.T[2], knn_segments, axis=ax, show=False, title='After')
        # ax = fig.add_subplot(223, projection='3d')
        # plot_3d_point_cloud(skeletons.T[0], skeletons.T[1], skeletons.T[2], axis=ax, show=False, title='Skeleton')
        # ax = fig.add_subplot(224, projection='3d')
        # plot_3d_point_cloud(reconstructed_skeleton.T[0], reconstructed_skeleton.T[1], reconstructed_skeleton.T[2], axis=ax, show=True, title='Reconstruction')
    out['segmentations'] = torch.tensor(out['segmentations'])
    with h5py.File(osp.join(out_dir, f'{data}.h5'), 'w') as f:
        log.info(f'Saving {data}')
        for k, v in out.items():
            log.info(f'Dataset {k}: {v.shape}')
            f.create_dataset(k, data=v)

# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument("-data", "--data")
#     args = vars(parser.parse_args())
#     data = args['data']
#
#     r = Reconstructor()
#     seg = Skeleton2Segments()
#     layout = GraphLayout()
#     out_dir = osp.join(DataPaths.POINT_CLOUDS_DB, 'segmentation')
#     Path(out_dir).mkdir(parents=True, exist_ok=True)
#     log.info(f'Preparing {data}')
#     with h5py.File(osp.join(DataPaths.POINT_CLOUDS_DB, 'smpl_like', f'{data}.h5'), "r") as f:
#         out = {k: np.array(v) for k,v in f.items()}
#         pcs = out['point_cloud']
#         faces = out['faces']
#         skeletons = out['skeletons']
#     out['segmentations'] = []
#     out['reconstructions'] = []
#     for i in range(pcs.shape[0]):
#         log.info(f'\t{i}/{pcs.shape[0]}')
#         pc, skeleton = pcs[i], skeletons[i]
#         mesh = trimesh.Trimesh(vertices=pc, faces=faces)
#         segments = seg.segment_points(mesh, skeleton)
#         knn_segments = seg.knn_classify(mesh, segments)
#         reconstructed_skeleton = r.reconstruct(mesh, knn_segments)
#         out['segmentations'].append(knn_segments)
#         out['reconstructions'].append(reconstructed_skeleton)
#         # fig = plt.figure(figsize=(15, 15))
#         # ax = fig.add_subplot(221, projection='3d')
#         # plot_3d_point_cloud(pc.T[0], pc.T[1], pc.T[2], segments, axis=ax, show=False, title='Before')
#         # ax = fig.add_subplot(222, projection='3d')
#         # plot_3d_point_cloud(pc.T[0], pc.T[1], pc.T[2], knn_segments, axis=ax, show=False, title='After')
#         # ax = fig.add_subplot(223, projection='3d')
#         # plot_3d_point_cloud(skeletons.T[0], skeletons.T[1], skeletons.T[2], axis=ax, show=False, title='Skeleton')
#         # ax = fig.add_subplot(224, projection='3d')
#         # plot_3d_point_cloud(reconstructed_skeleton.T[0], reconstructed_skeleton.T[1], reconstructed_skeleton.T[2], axis=ax, show=True, title='Reconstruction')
#     out['segmentations'] = torch.tensor(out['segmentations'])
#     out['reconstructions'] = torch.tensor(out['reconstructions'])
#     with h5py.File(osp.join(out_dir, f'{data}.h5'), 'w') as f:
#         log.info(f'Saving {data}')
#         for k, v in out.items():
#             log.info(f'Dataset {k}: {v.shape}')
#             f.create_dataset(k, data=v)
