from action_recognizer import Solver
from dataset import PoseGraphDataset

import torch
from torch_geometric.data import Batch
from dataloader import CustomDataLoader

def load_dataset(dataset_folder):
    dataset = PoseGraphDataset(dataset_folder, skip=11)

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

def main():
    train_dataset, val_dataset = load_dataset('./data/')
    train_dataloader = CustomDataLoader(train_dataset, batch_size=4, shuffle=False)
    val_dataloader = CustomDataLoader(val_dataset, batch_size=4, shuffle=False)
    
    solver = Solver(
        gcn_num_features=3,
        gcn_hidden_dim1=32,
        gcn_hidden_dim2=64,
        gcn_output_dim=128,
        transformer_d_model=128,
        transformer_nhead=16,
        transformer_num_layers=8,
        transformer_num_features=128,
        transformer_dropout=0.1,
        transformer_dim_feedforward=2048,
    )
    solver.train(train_dataloader, val_dataloader, epochs=100)

if __name__ == "__main__":
    main()
