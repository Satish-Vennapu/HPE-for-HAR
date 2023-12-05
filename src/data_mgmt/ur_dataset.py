import torch
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.data import Dataset, Batch

# from torch_geometric.loader import DataLoader

from .dataloader import CustomDataLoader

from typing import Dict


def get_label(file_name: str) -> int:
    if "adl" in file_name:
        return 0
    return 1


def is_valid_file(file_name: str, skip: int = 11) -> bool:
    """
    Checks if the file is a valid file

    Parameters
    ----------
    file_name : str
        Name of the file
    skip : int, optional
        Number of frames to skip, by default 11

    Returns
    -------
    bool
        True if the file is valid, False otherwise
    """
    npy_file = file_name.endswith(".npy")
    cam0 = "cam0" in file_name
    skip_frame_num = file_name.split("/")[-1].split("-")[-2] == str(skip)

    return npy_file and cam0 and skip_frame_num


def get_edge_index():
    POSE_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),  # Head to left shoulder
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),  # Head to right shoulder
        (9, 10),
        (11, 12),  # Left and right shoulder
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),  # Left arm
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),  # Right arm
        (11, 23),
        (12, 24),
        (23, 24),  # Torso
        (23, 25),
        (25, 27),
        (27, 29),
        (29, 31),  # Left leg
        (24, 26),
        (26, 28),
        (28, 30),
        (30, 32),  # Right leg
    ]
    edge_index = torch.tensor(POSE_CONNECTIONS, dtype=torch.long).t().contiguous()

    return edge_index


class PoseGraphDataset(Dataset):
    """
    Dataset class for the keypoint dataset
    """

    def __init__(self, dataset_folder: str, skip: int = 11) -> None:
        super().__init__(None, None, None)
        self.dataset_folder = dataset_folder
        self.edge_index = get_edge_index()

        self.poses = []
        self.labels = []
        self.file_names = []

        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if is_valid_file(file, skip):
                    file_path = os.path.join(root, file)

                    kps = np.load(file_path)
                    kps = kps[:, :, :3]
                    pose_graphs = self._create_pose_graph(kps)

                    self.poses.append(pose_graphs)
                    self.labels.append(get_label(file_path))
                    self.file_names.append(file_path)

    def _create_pose_graph(self, keypoints: torch.Tensor) -> Data:
        """
        Creates a Pose Graph from the given keypoints and edge index

        Parameters
        ----------
        keypoints : torch.Tensor
            Keypoints of the pose
        edge_index : torch.Tensor
            Edge index of the pose

        Returns
        -------
        Data
            Pose Graph
        """
        pose_graphs = []
        for t in range(keypoints.shape[0]):
            pose_graph = Data(
                x=torch.tensor(keypoints[t, :, :], dtype=torch.float),
                edge_index=self.edge_index,
            )
            pose_graphs.append(pose_graph)

        return pose_graphs

    def len(self) -> int:
        """
        Returns the number of samples in the dataset

        Returns
        -------
        int : len
            Number of samples in the dataset
        """
        return len(self.poses)

    def get(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns the sample at the given index

        Returns
        -------
        dict : {kps, label, file_name}
            A dictionary containing the keypoint array, label and file name
        """
        poses = self.poses[index]
        label = self.labels[index]
        return poses, label


if __name__ == "__main__":
    dataset = PoseGraphDataset("../../data/")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataloader = CustomDataLoader(train_dataset, batch_size=4, shuffle=False)

    print(len(val_dataset))
    # val_dataloader = CustomDataLoader(val_dataset, batch_size=4, shuffle=True)

