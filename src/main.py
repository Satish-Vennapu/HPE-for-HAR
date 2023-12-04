from action_recognizer import Solver
from data_mgmt.dataset import PoseGraphDataset

import torch
from data_mgmt.dataloader import CustomDataLoader
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument(
        "--gcn_num_features", 
        type=int, 
        default=3, 
        help="Number of features in the GCN"
    )
    parser.add_argument(
        "--gcn_hidden_dim1",
        type=int,
        default=32,
        help="Dimension of the first hidden layer of the GCN",
    )
    parser.add_argument(
        "--gcn_hidden_dim2",
        type=int,
        default=64,
        help="Dimension of the second hidden layer of the GCN",
    )
    parser.add_argument(
        "--gcn_output_dim",
        type=int,
        default=128,
        help="Dimension of the output layer of the GCN",
    )
    parser.add_argument(
        "--transformer_d_model",
        type=int,
        default=128,
        help="Dimension of the input embedding",
    )
    parser.add_argument(
        "--transformer_nhead", 
        type=int, 
        default=32, 
        help="Number of attention heads"
    )
    parser.add_argument(
        "--transformer_num_layers",
        type=int,
        default=16,
        help="Number of transformer encoder layers",
    )
    parser.add_argument(
        "--transformer_num_features",
        type=int,
        default=128,
        help="Number of features in the input sequence",
    )
    parser.add_argument(
        "--transformer_dropout", 
        type=float, 
        default=0.1, 
        help="Dropout rate"
    )
    parser.add_argument(
        "--transformer_dim_feedforward",
        type=int,
        default=2048,
        help="Dimension of the feedforward network",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001, 
        help="Learning rate"
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="../data/",
        help="Path to the dataset folder",
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=20, 
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size"
    )
    parser.add_argument(
        "--shuffle", 
        type=bool, 
        default=False, 
        help="Shuffle the dataset"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="../output/",
        help="Path to the output folder",
    )

    args = parser.parse_args()
    return args


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
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    return train_dataset, val_dataset


def main():
    args = parse_args()

    train_dataset, val_dataset = load_dataset(args.dataset_folder)
    train_dataloader = CustomDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=args.shuffle
    )
    val_dataloader = CustomDataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=args.shuffle
    )

    solver = Solver(
        gcn_num_features=args.gcn_num_features,
        gcn_hidden_dim1=args.gcn_hidden_dim1,
        gcn_hidden_dim2=args.gcn_hidden_dim2,
        gcn_output_dim=args.gcn_output_dim,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_num_layers=args.transformer_num_layers,
        transformer_num_features=args.transformer_num_features,
        transformer_dropout=args.transformer_dropout,
        transformer_dim_feedforward=args.transformer_dim_feedforward,
        lr=args.lr,
    )
    
    solver.train(
        train_dataloader,
        val_dataloader,
        epochs=args.epochs,
        output_folder=args.output_folder,
    )


if __name__ == "__main__":
    main()
