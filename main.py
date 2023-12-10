import torch
import numpy as np
import argparse
from collections import Counter

from solver import Solver
from model import get_multi_view, get_single_view
from utils.logger import Logger
from utils.model_config import ModelConfig
from data_mgmt.datasets.ntu_dataset import PoseGraphDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument(
        "--aggregator",
        type=str,
        default="average",
        help="Aggregator for the GCN output - Options: average, linear, self_attn",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--dataset",
        type=str,
        default="../dataset/Python/raw_npy/",
        help="Path to the dataset folder",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="Shuffle the dataset"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output/",
        help="Path to the output folder",
    )
    parser.add_argument(
        "--single_view",
        action="store_true",
        help="Use single view",
    )
    parser.add_argument(
        "--logging_config",
        type=str,
        default="./config/logger.ini",
        help="Path to the logging config file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./config/model.json",
        help="Path to the model config file",
    )
    parser.add_argument(
        "--occlude",
        action="store_true",
        help="Augment the dataset",
    )
    args = parser.parse_args()

    if args.aggregator not in ["average", "linear", "self_attn"]:
        raise ValueError("Invalid aggregator must be one of average, linear, self_attn")

    return args


def load_dataset(dataset_folder, logger, occlude=False):
    np.random.seed(42)
    dataset = PoseGraphDataset(dataset_folder, skip=11, occlude=occlude)

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
    logger = Logger(args.logging_config).get_logger()

    logger.info("\n")
    logger.info("Loading the dataset...")
    train_dataset, val_dataset, test_dataset = load_dataset(args.dataset, logger, args.occlude)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Testing dataset size: {len(test_dataset)}")

    logger.info(f"Model type: {'Single View' if args.single_view else 'Multi View'}")
    model_config = ModelConfig(args.model_config).get_config()
    if args.single_view:
        model, (train_dataloader, val_dataloader, test_dataloader) = get_single_view(
            model_config, args, (train_dataset, val_dataset, test_dataset)
        )
    else:
        logger.info(f"Aggregator: {args.aggregator}")
        model, (train_dataloader, val_dataloader, test_dataloader) = get_multi_view(
            model_config, args, (train_dataset, val_dataset, test_dataset)
        )

    solver = Solver(model, lr=args.lr, logger=logger)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")

    logger.info("Training the model. Please wait...")
    solver.train(
        train_dataloader,
        val_dataloader,
        epochs=args.epochs,
        output_path=args.output_folder,
        save_model=True,
    )

    logger.info("")
    logger.info("Testing model on the test dataset...")
    solver.test(test_dataloader, output_path=args.output_folder, aggregator=args.aggregator)

if __name__ == "__main__":
    main()
