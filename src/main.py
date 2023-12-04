from action_recognizer import Solver
from dataset import KeyPointsDataset

import torch

def load_dataset(dataset_folder):
    dataset = KeyPointsDataset(dataset_folder, skip=11)

    if len(dataset) > 0:
        print("Dataset loaded successfully")
        print("Number of samples:", len(dataset))
    else:
        print("Dataset not loaded. Please check the dataset folder.")
        exit()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

def get_edge_index():
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Head to left shoulder
        (0, 4), (4, 5), (5, 6), (6, 8),  # Head to right shoulder
        (9, 10), (11, 12),               # Left and right shoulder
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm
        (11, 23), (12, 24), (23, 24),    # Torso
        (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
        (24, 26), (26, 28), (28, 30), (30, 32)   # Right leg
    ]
    edge_index = torch.tensor(POSE_CONNECTIONS, dtype=torch.long).t().contiguous()

    return edge_index

def main():
    train_dataset, val_dataset = load_dataset('./data/')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    solver = Solver(
        gcn_num_features=3,
        gcn_hidden_dim1=32,
        gcn_hidden_dim2=64,
        gcn_output_dim=128,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_num_layers=6,
        transformer_num_features=128,
        transformer_dropout=0.1,
        transformer_dim_feedforward=2048,
        edge_index=get_edge_index()
    )
    solver.train(train_dataloader, val_dataloader, epochs=50)

if __name__ == "__main__":
    main()
