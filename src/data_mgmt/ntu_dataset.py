import torch
import numpy as np
import os
import regex as re

from torch_geometric.data import Data
from torch_geometric.data import Dataset, Batch

# from torch_geometric.loader import DataLoader

from .dataloader import CustomDataLoader

from typing import Dict
from collections import Counter

label_action =[
    {"id" : 0, "A001" : "drink water"},
    {"id" : 1, "A002" : "eating"},
    # {"id" : 2, "A003" : "brushing teeth"},
    # {"id" : 3, "A004" : "brushing hair"},
    # {"id" : 4, "A005" : "drop"},
    # {"id" : 5, "A006" : "pickup"},
    # {"id" : 6, "A007" : "throw"},
    # {"id" : 7, "A008" : "sitting down"},
    # {"id" : 8, "A009" : "standing up"},
    # {"id" : 9, "A010" : "clapping"},
    # {"id" : 10, "A011" : "reading"},
    # {"id" : 11, "A012" : "writing"},
    # {"id" : 12, "A013" : "tear up paper"},
    # {"id" : 13, "A014" : "wear jacket"},
    # {"id" : 14, "A015" : "take off jacket"},
    # {"id" : 15, "A016" : "wear a shoe"},
    # {"id" : 16, "A017" : "take off a shoe"},
    # {"id" : 17, "A018" : "wear on glasses"},
    # {"id" : 18, "A019" : "take off glasses"},
    # {"id" : 19, "A020" : "put on a hat/cap"},
    # {"id" : 20, "A021" : "take off a hat/cap"},
    # {"id" : 21, "A022" : "cheer up"},
    # {"id" : 22, "A023" : "hand waving"},
    # {"id" : 23, "A024" : "kicking something"},
    # {"id" : 24, "A025" : "reach into pocket"},
    # {"id" : 25, "A026" : "hopping (one foot jumping)"},
    # {"id" : 26, "A027" : "jump up"},
    # {"id" : 27, "A028" : "make a phone call/answer phone"},
    # {"id" : 28, "A029" : "playing with phone/tablet"},
    # {"id" : 29, "A030" : "typing on a keyboard"},
    # {"id" : 30, "A031" : "pointing to something with finger"},
    # {"id" : 31, "A032" : "taking a selfie"},
    # {"id" : 32, "A033" : "check time (from watch)"},
    # {"id" : 33, "A034" : "rub two hands together"},
    # {"id" : 34, "A035" : "nod head/bow"},
    # {"id" : 35, "A036" : "shake head"},
    # {"id" : 36, "A037" : "wipe face"},
    # {"id" : 37, "A038" : "salute"},
    # {"id" : 38, "A039" : "put the palms together"},
    # {"id" : 39, "A040" : "cross hands in front (say stop)"},
    # {"id" : 40, "A041" : "sneeze/cough"},
    # {"id" : 41, "A042" : "staggering"},
    # {"id" : 42, "A043" : "falling"},
    # {"id" : 43, "A044" : "touch head"},
    # {"id" : 44, "A045" : "touch chest"},
    # {"id" : 45, "A046" : "touch back"},
    # {"id" : 46, "A047" : "touch neck"},
    # {"id" : 47, "A048" : "nausea or vomiting condition"},
    # {"id" : 48, "A049" : "feeling warm"}
]

file_name_regex = r"S(\d{3})C001P(\d{3})R(\d{3})A(\d{3})"
file_name_regex = re.compile(file_name_regex)

def get_label(file_name: str) -> int:
    label = file_name[-4:]
    for i in label_action:
        if label in i:
            return i["id"]
    return -1


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
    file_name = file_name.split("/")[-1].split(".")[0]

    if file_name_regex.match(file_name) is None or get_label(file_name) == -1:
        return False

    return npy_file 

def get_edge_index():
    POSE_CONNECTIONS = [
    (3, 2),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (0, 1), (1, 20),
    (0, 16), (0, 12),
    (16, 17), (17, 18), (18, 19),
    (12, 13), (13, 14), (14, 15)
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

                    file_data = np.load(file_path, allow_pickle=True).item()
                    kps = file_data["skel_body0"]
                    pose_graphs = self._create_pose_graph(kps)

                    self.poses.append(pose_graphs)
                    self.labels.append(get_label(file_data["file_name"]))
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
    dataset = PoseGraphDataset("../../../dataset/Python/raw_npy/")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataloader = CustomDataLoader(train_dataset, batch_size=4, shuffle=False)
    
    label_counts = Counter(dataset.labels)

    # get unique labels
    unique_labels = len(list(set(dataset.labels)))
    print("Unique labels:", unique_labels)

    for label, count in label_counts.items():
        print(f"Label {label}: {count}")
