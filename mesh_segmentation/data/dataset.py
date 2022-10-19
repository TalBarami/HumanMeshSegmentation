from pathlib import Path
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm


def load_mesh(mesh_filename: Path):
    """Extract vertices and faces from raw mesh file.

    Parameters
    ----------
    mesh_filename: PathLike
        Path to mesh `.ply` file.

    Returns
    -------
    vertices: torch.tensor
        Float tensor of size (|V|, 3), where each row
        specifies the spatial position of a vertex in 3D space.
    faces: torch.tensor
        Intger tensor of size (|M|, 3), where each row
        defines a traingular face.
    """
    mesh = trimesh.load_mesh(mesh_filename, process=False)
    vertices = torch.from_numpy(mesh.vertices).to(torch.float)
    faces = torch.from_numpy(mesh.faces)
    faces = faces.t().to(torch.long).contiguous()
    return vertices, faces

class SegmentationFaust(InMemoryDataset):
    # map_seg_label_to_id = GraphLayout()._parts
    # map_seg_label_to_id = dict(
    #     head=0,
    #     torso=1,
    #     left_arm=2,
    #     left_hand=3,
    #     right_arm=4,
    #     right_hand=5,
    #     left_upper_leg=6,
    #     left_lower_leg=7,
    #     left_foot=8,
    #     right_upper_leg=9,
    #     right_lower_leg=10,
    #     right_foot=11,
    # )

    def __init__(self, root, mode, segment_ids, pre_transform=None):
        """
        Parameters
        ----------
        root: PathLike
            Root directory where the dataset should be saved.
        mode: str
            Type of the dataset (train/val/test)
        part_ids: dict
        pre_transform: Optional[Callable]
            A function that takes in a torch_geometric.data.Data object
            and outputs a transformed version. Note that the transformed
            data object will be saved to disk.

        """
        super().__init__(root, pre_transform)
        self.mode = mode
        self.segment_ids = segment_ids
        modes = {'train': 0, 'val': 1, 'test': 2}
        path = self.processed_paths[modes[mode]]
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self) -> list:
        return ["train.pt", "val.pt", "test.pt"]

    @property
    @lru_cache(maxsize=32)
    def _segmentation_labels(self):
        """Extract segmentation labels."""
        path_to_labels = Path(self.root) / 'segmentation.npy'
        with open(path_to_labels, 'rb') as f:
            seg_labels = np.expand_dims(np.load(f), 1)
        return torch.from_numpy(seg_labels).type(torch.int64)

    def _mesh_filenames(self):
        """Extract all mesh filenames."""
        path_to_meshes = Path(self.root) / "meshes"
        return path_to_meshes.glob("*.ply")

    # def _unzip_dataset(self):
    #     """Extract dataset from zip."""
    #     path_to_zip = Path(self.root) / "MPI-FAUST.zip"
    #     extract_zip(str(path_to_zip), self.root, log=False)

    def process(self):
        """Process the raw meshes files and their corresponding class labels."""
        # self._unzip_dataset()

        dataset = {
            'train': [],
            'val': [],
            'test': []
        }
        paths = {
            'train': self.processed_paths[0],
            'val': self.processed_paths[1],
            'test': self.processed_paths[2]
        }
        for mesh_filename in tqdm(sorted(self._mesh_filenames())):
            dtype = mesh_filename.name.split('_')[0]
            vertices, faces = load_mesh(mesh_filename)
            data = Data(x=vertices, face=faces)
            data.segmentation_labels = self._segmentation_labels
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            dataset[dtype].append(data)

        for k, v in dataset.items():
            torch.save(self.collate(v), paths[k])

class InferenceDataset(InMemoryDataset):
    def __init__(self, transform, data):
        super().__init__(transform=transform)
        self.dataset = []
        if type(data) == list:
            for mesh in data:
                d = self.init_data(mesh)
                self.dataset.append(d)
        else:
            d = self.init_data(data)
            self.dataset.append(d)
        self.collate(self.dataset)

    def init_data(self, mesh):
        vertices = torch.from_numpy(mesh.vertices).to(torch.float)
        faces = torch.from_numpy(mesh.faces)
        faces = faces.t().to(torch.long).contiguous()
        return self.transform(Data(x=vertices, face=faces))

class NormalizeUnitSphere(BaseTransform):
    """Center and normalize node-level features to unit length."""

    @staticmethod
    def _re_center(x):
        """Recenter node-level features onto feature centroid."""
        centroid = torch.mean(x, dim=0)
        return x - centroid

    @staticmethod
    def _re_scale_to_unit_length(x):
        """Rescale node-level features to unit-length."""
        max_dist = torch.max(torch.norm(x, dim=1))
        return x / max_dist

    def __call__(self, data: Data):
        if data.x is not None:
            data.x = self._re_scale_to_unit_length(self._re_center(data.x))

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)