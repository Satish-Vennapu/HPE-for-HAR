import torch
from torch_geometric.data import Batch
from torch.utils.data.dataloader import default_collate
from typing import Any, List, Mapping, Sequence, Tuple


class Collater:
    """
    Collates the batch of data

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to collate
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch) -> Any:
        """
        Collates the batch of data

        Parameters
        ----------
        batch : List[Any]
            Batch of data

        Returns
        -------
        Any
            Collated batch of data
        """
        elem = batch[0]

        if isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

    def collate_fn(self, batch: List[Any]) -> Any:
        """
        Collates the batch of data

        Parameters
        ----------
        batch : List[Any]
            Batch of data

        Returns
        -------
        Any
            Collated batch of data
        """
        view1 = [item["view1"] for item in batch]
        view2 = [item["view2"] for item in batch]
        view3 = [item["view3"] for item in batch]
        labels = [item["label"] for item in batch]

        batched_graphs = []
        for i in range(len(view1)):
            min_len = min(len(view1[i]), len(view2[i]), len(view3[i]))
            view1[i] = view1[i][:min_len]
            view2[i] = view2[i][:min_len]
            view3[i] = view3[i][:min_len]

            batched_view1 = Batch.from_data_list(view1[i])
            batched_view2 = Batch.from_data_list(view2[i])
            batched_view3 = Batch.from_data_list(view3[i])

            batched_graphs.append([batched_view1, batched_view2, batched_view3])

        labels = torch.tensor(labels, dtype=torch.long)

        return batched_graphs, labels


class DataLoader(torch.utils.data.DataLoader):
    """
    Dataloader for the dataset

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to load
    batch_size : int, optional
        Batch size, by default 1
    shuffle : bool, optional
        Whether to shuffle the dataset, by default False
    """

    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False, **kwargs):
        self.collator = Collater(dataset)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collator.collate_fn,
            **kwargs,
        )
