import torch
import numpy as np
import os

from typing import Dict

def get_label(file_name : str) -> int:
    if 'adl' in file_name:
        return 0
    return 1

def is_valid_file(file_name : str, skip : int = 11) -> bool:
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
    npy_file = file_name.endswith('.npy')
    cam0 = 'cam0' in file_name
    skip_frame_num = file_name.split('/')[-1].split('-')[-2] == str(skip)

    return npy_file and cam0 and skip_frame_num

class KeyPointsDataset(torch.utils.data.Dataset):
    """
    Dataset class for the keypoint dataset
    """
    def __init__(self, dataset_folder : str, skip : int = 11) -> None:
        self.dataset_folder = dataset_folder
        self.labels = []

        self.kps = []
        self.file_names = []

        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if is_valid_file(file, skip):
                    file_path = os.path.join(root, file)
                    kps = np.load(file_path)
                    kps = kps[:, :, :3]
                    self.kps.append(kps)
                    self.labels.append(get_label(file_path))
                    self.file_names.append(file_path)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset

        Returns
        -------
        int : len
            Number of samples in the dataset
        """
        return len(self.kps)

    def __getitem__(self, index : int) -> Dict[str, torch.Tensor]:
        """
        Returns the sample at the given index

        Returns
        -------
        dict : {kps, label, file_name}
            A dictionary containing the keypoint array, label and file name
        """
        kps = self.kps[index]
        label = self.labels[index]
        return {'kps' : torch.from_numpy(kps), 'label' : torch.tensor(label), 'file_name' : self.file_names[index]}
    
if __name__ == "__main__":

    dataset = KeyPointsDataset('./data/')
    if len(dataset) > 0:
        print("Dataset loaded successfully")
        print("Number of samples:", len(dataset))
        print("First sample:")
        print("Keypoints shape:", dataset[0]['kps'].shape)
        print("Label:", dataset[0]['label'].item())
        print("File name:", dataset[0]['file_name'])
    else:
        print("Dataset not loaded")

    # check if dataset[index]['kps'].shape == (24, 33, 3)
    print("Checking if dataset[index]['kps'].shape == (24, 33, 3)")
    for i in range(len(dataset)):
        print(f"Checking sample {i}...")
        print(f"dataset[{i}]['kps'].shape:", dataset[i]['kps'].shape)
        # assert dataset[i]['kps'].shape == (24, 33, 3), f"dataset[{i}]['kps'].shape != (24, 33, 3)"