import argparse
from torch_geometric.data import Dataset

from models.multi_view import MultiViewActionRecognizer
from models.single_view import SingleViewActionRecognizer

from data_mgmt.dataloaders.multi_dataloader import DataLoader as MultiDataLoader
from data_mgmt.dataloaders.single_dataloader import DataLoader as SingleDataLoader

from typing import Tuple


def get_multi_view(
    args: argparse.Namespace,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
) -> Tuple[
    MultiViewActionRecognizer, Tuple[MultiDataLoader, MultiDataLoader, MultiDataLoader]
]:
    """
    Returns the model and the dataloaders for the multi view case

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed to the program
    train_dataset : Dataset
        Training dataset
    val_dataset : Dataset
        Validation dataset
    test_dataset : Dataset
        Testing dataset

    Returns
    -------
    Tuple[MultiViewActionRecognizer, Tuple[MultiDataLoader, MultiDataLoader, MultiDataLoader]]
        Tuple containing the model and the dataloaders for the training, validation and testing datasets
    """
    train_loader = MultiDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = MultiDataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = MultiDataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )

    return MultiViewActionRecognizer(
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
        aggregator=args.aggregator,
    ), (train_loader, val_loader, test_loader)


def get_single_view(
    args: argparse.Namespace,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
) -> Tuple[
    SingleViewActionRecognizer, Tuple[MultiDataLoader, MultiDataLoader, MultiDataLoader]
]:
    """
    Returns the model and the dataloaders for the single view case

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed to the program
    train_dataset : Dataset
        Training dataset
    val_dataset : Dataset
        Validation dataset

    Returns
    -------
    Tuple[SingleViewActionRecognizer, Tuple[MultiDataLoader, MultiDataLoader, MultiDataLoader]]
        Tuple containing the model and the dataloaders for the training, validation and testing datasets
    """
    train_loader = SingleDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = SingleDataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = SingleDataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )

    return SingleViewActionRecognizer(
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
    ), (train_loader, val_loader, test_loader)
