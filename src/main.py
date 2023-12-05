import torch
import argparse
from collections import Counter

from solver import Solver
from utils.logger import Logger
from data_mgmt.datasets.ntu_dataset import PoseGraphDataset
from data_mgmt.dataloaders.multi_dataloader import DataLoader as MultiDataLoader
from data_mgmt.dataloaders.single_dataloader import DataLoader as SingleDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument(
        "--gcn_num_features", type=int, default=3, help="Number of features in the GCN"
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
        "--transformer_nhead", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument(
        "--transformer_num_layers",
        type=int,
        default=8,
        help="Number of transformer encoder layers",
    )
    parser.add_argument(
        "--transformer_num_features",
        type=int,
        default=128,
        help="Number of features in the input sequence",
    )
    parser.add_argument(
        "--transformer_dropout", type=float, default=0.1, help="Dropout rate"
    )
    parser.add_argument(
        "--transformer_dim_feedforward",
        type=int,
        default=2048,
        help="Dimension of the feedforward network",
    )
    parser.add_argument(
        "--transformer_num_classes",
        type=int,
        default=3,
        help="Dimension of the feedforward network",
    )
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="../../dataset/Python/raw_npy/",
        help="Path to the dataset folder",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="Shuffle the dataset"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="../output/",
        help="Path to the output folder",
    )
    parser.add_argument(
        "--single_view",
        action="store_true",
        help="Use single view",
    )
    args = parser.parse_args()
    return args


def load_dataset(dataset_folder, logger):
    dataset = PoseGraphDataset(dataset_folder, skip=11)

    if len(dataset) > 0:
        logger.info("Dataset loaded successfully.")
        logger.info(f"Dataset size: {len(dataset)}")
    else:
        logger.error("Dataset loading failed.")
        logger.info("Check if the dataset folder is correct.")
        exit()

    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    test_size = int(0.25 * len(val_dataset))
    val_dataset, test_dataset = torch.utils.data.random_split(
        val_dataset, [len(val_dataset) - test_size, test_size], generator=generator
    )

    label_counts = Counter(dataset.labels)
    unique_labels = len(list(set(dataset.labels)))
    logger.info(f"Number of unique labels: {unique_labels}")

    for label, count in label_counts.items():
        logger.info(f"Label: {label}, Count: {count}")

    return train_dataset, val_dataset, test_dataset


def main():
    args = parse_args()
    logger = Logger("../config/logger.ini").get_logger()

    logger.info("\n")
    logger.info("Loading the dataset...")
    train_dataset, val_dataset, test_dataset = load_dataset(args.dataset_folder, logger)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Testing dataset size: {len(test_dataset)}")

    logger.info(f"Model type: {'Single View' if args.single_view else 'Multi View'}")
    if not args.single_view:
        train_dataloader = MultiDataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=args.shuffle
        )
        val_dataloader = MultiDataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=args.shuffle
        )
        test_dataloader = MultiDataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=args.shuffle
        )
    else:
        train_dataloader = SingleDataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=args.shuffle
        )
        val_dataloader = SingleDataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=args.shuffle
        )
        test_dataloader = SingleDataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=args.shuffle
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
        transformer_num_classes=args.transformer_num_classes,
        lr=args.lr,
        is_single_view=args.single_view,
        logger=logger,
    )

    logger.info("")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")

    logger.info("Training the model. Please wait...")
    solver.train(
        train_dataloader,
        val_dataloader,
        epochs=args.epochs,
        output_path=args.output_folder,
    )

    logger.info("")
    logger.info("Testing model on the test dataset...")
    solver.test(test_dataloader)


if __name__ == "__main__":
    main()
