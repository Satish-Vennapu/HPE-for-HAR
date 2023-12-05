import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from itertools import cycle

from torch_geometric.data import Batch

from models.transformer import TransformerBinaryClassifier
from models.gcn import PoseGCN

from typing import Tuple

from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import seaborn as sns

class ActionRecogniser(nn.Module):
    def __init__(
        self,
        gcn_num_features: int,
        gcn_hidden_dim1: int,
        gcn_hidden_dim2: int,
        gcn_output_dim: int,
        transformer_d_model: int,
        transformer_nhead: int,
        transformer_num_layers: int,
        transformer_num_features: int,
        transformer_dropout: float = 0.1,
        transformer_dim_feedforward: int = 2048,
        transformer_num_classes: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        gcn_num_features : int
            Number of features in the input sequence
        gcn_hidden_dim1 : int
            Dimension of the first hidden layer of the GCN
        gcn_hidden_dim2 : int
            Dimension of the second hidden layer of the GCN
        gcn_output_dim : int
            Dimension of the output layer of the GCN
        transformer_d_model : int
            Dimension of the input embedding
        transformer_nhead : int
            Number of attention heads
        transformer_num_layers : int
            Number of transformer encoder layers
        transformer_num_features : int
            Number of features in the input sequence
        transformer_dropout : float, optional
            Dropout rate, by default 0.1
        transformer_dim_feedforward : int, optional
            Dimension of the feedforward network, by default 2048
        """
        super(ActionRecogniser, self).__init__()
        self.gcn1 = PoseGCN(
            gcn_num_features, gcn_hidden_dim1, gcn_hidden_dim2, gcn_output_dim
        )
        self.gcn2 = PoseGCN(
            gcn_num_features, gcn_hidden_dim1, gcn_hidden_dim2, gcn_output_dim
        )
        self.gcn3 = PoseGCN(
            gcn_num_features, gcn_hidden_dim1, gcn_hidden_dim2, gcn_output_dim
        )
        self.transformer = TransformerBinaryClassifier(
            transformer_d_model,
            transformer_nhead,
            transformer_num_layers,
            transformer_num_features,
            transformer_dropout,
            transformer_dim_feedforward,
            num_classes=transformer_num_classes
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        kps : torch.Tensor
            Input sequence of keypoints

        Returns
        -------
        torch.Tensor
            Classification of the input sequence of keypoints
        """
        batch_view1 = [Batch.from_data_list(item[0]) for item in batch]
        batch_view2 = [Batch.from_data_list(item[1]) for item in batch]
        batch_view3 = [Batch.from_data_list(item[2]) for item in batch]

        outputs = []
        for i in range(len(batch_view1)):
            view1_embedding = self.gcn1(batch_view1[i])
            view2_embedding = self.gcn2(batch_view2[i])
            view3_embedding = self.gcn3(batch_view3[i])

            min_length = min(len(view1_embedding), len(view2_embedding), len(view3_embedding))
            view1_embedding = view1_embedding[:min_length]
            view2_embedding = view2_embedding[:min_length]
            view3_embedding = view3_embedding[:min_length]

            aggregated_embeddings = (view1_embedding + view2_embedding + view3_embedding) / 3

            output = self.transformer(aggregated_embeddings.unsqueeze(0).to("cuda"))
            outputs.append(output)

        return torch.stack(outputs).squeeze(1)

class Solver:
    def __init__(
        self,
        gcn_num_features: int,
        gcn_hidden_dim1: int,
        gcn_hidden_dim2: int,
        gcn_output_dim: int,
        transformer_d_model: int,
        transformer_nhead: int,
        transformer_num_layers: int,
        transformer_num_features: int,
        transformer_dropout: float = 0.1,
        transformer_dim_feedforward: int = 2048,
        transformer_num_classes: int = 2,
        lr: float = 0.001,
    ) -> None:
        """
        Parameters
        ----------
        gcn_num_features : int
            Number of features in the input sequence
        gcn_hidden_dim1 : int
            Dimension of the first hidden layer of the GCN
        gcn_hidden_dim2 : int
            Dimension of the second hidden layer of the GCN
        gcn_output_dim : int
            Dimension of the output layer of the GCN
        transformer_d_model : int
            Dimension of the input embedding
        transformer_nhead : int
            Number of attention heads
        transformer_num_layers : int
            Number of transformer encoder layers
        transformer_num_features : int
            Number of features in the input sequence
        transformer_dropout : float, optional
            Dropout rate, by default 0.1
        transformer_dim_feedforward : int, optional
            Dimension of the feedforward network, by default 2048
        lr : float, optional
            Learning rate, by default 0.001
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.model = ActionRecogniser(
            gcn_num_features,
            gcn_hidden_dim1,
            gcn_hidden_dim2,
            gcn_output_dim,
            transformer_d_model,
            transformer_nhead,
            transformer_num_layers,
            transformer_num_features,
            transformer_dropout,
            transformer_dim_feedforward,
            transformer_num_classes
        ).to(self.device)

        print("Number of parameters :", sum(p.numel() for p in self.model.parameters()))
        self.lr = lr

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

        self.train_loss = []
        self.val_loss = []

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 20,
        output_path: str = "../output",
    ) -> None:
        """
        Trains the model

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset
        epochs : int, optional
            Number of epochs to train, by default 20

        Returns
        -------
        None
        """
        torch.manual_seed(0)
        best_val_loss = float("inf")

        print("\nTraining started. Please wait...")
        for epoch in range(epochs):
            epoch_loss, epoch_correct, epoch_count = self.train_one_epoch(train_loader)
            self.train_loss.append(epoch_loss)
            print(
                f"\nepoch: {epoch} | epoch loss: {epoch_loss}        | epoch accuracy: {epoch_correct / epoch_count}"
            )

            epoch_val_loss, epoch_val_correct, epoch_val_count = self.evaluate(
                val_loader
            )
            self.val_loss.append(epoch_val_loss)
            print(
                f"epoch: {epoch} | epoch val loss: {epoch_val_loss}   | epoch val accuracy: {epoch_val_correct / epoch_val_count}"
            )
            # self.scheduler.step()

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(self.model.state_dict(), output_path+"best_model.pt")

        self._plot_losses()

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, int, int]:
        """
        Trains the model for one epoch

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset

        Returns
        -------
        Tuple[float, int, int]
            Tuple containing the epoch loss, epoch correct and epoch count
        """
        self.model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_count = 0
        for idx, batch in enumerate(iter(train_loader)):
            predictions = self.model(batch[0])
            labels = batch[1].to(self.device)

            loss = self.criterion(predictions, labels)

            correct = predictions.argmax(axis=1) == labels
            epoch_correct += correct.sum().item()
            epoch_count += correct.size(0)

            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.optimizer.step()

        return epoch_loss, epoch_correct, epoch_count

    def evaluate(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, int, int]:
        """
        Evaluates the model

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset

        Returns
        -------
        Tuple[float, int, int]
            Tuple containing the validation epoch loss, validation epoch correct
            and validation epoch count
        """
        self.model.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_correct = 0
            val_epoch_count = 0

            for idx, batch in enumerate(iter(val_loader)):
                predictions = self.model(batch[0])
                labels = batch[1].to(self.device)

                val_loss = self.criterion(predictions, labels)

                correct = predictions.argmax(axis=1) == labels

                val_epoch_correct += correct.sum().item()
                val_epoch_count += correct.size(0)
                val_epoch_loss += val_loss.item()

        return val_epoch_loss, val_epoch_correct, val_epoch_count
    
    def test(self, test_loader: torch.utils.data.DataLoader) -> Tuple[float, int, int]:
        """
        Evaluates the model

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            DataLoader for the test dataset

        Returns
        -------
        Tuple[float, int, int]
            Tuple containing the test epoch loss, test epoch correct
            and test epoch count
        """
        self.model.eval()
        with torch.no_grad():
            predictions = []
            labels = []
            for idx, batch in enumerate(iter(test_loader)):
                predictions.extend(self.model(batch[0]).argmax(axis=1).tolist())
                labels.extend(batch[1].tolist())

            print("Predictions:", predictions)
            print("Labels:", labels)
            
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions, average=None)

            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1_score}")

            plt.figure(figsize=(8, 6))
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(3), colors):
                fpr, tpr, _ = roc_curve(labels, predictions, pos_label=i)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic for Multi-class')
            plt.legend(loc="lower right")
            plt.show()

            cm = confusion_matrix(labels, predictions)
            ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
            ax.set_xlabel("Predicted labels")
            ax.set_ylabel("True labels")
            ax.set_title("Confusion Matrix")
            plt.show()        

    def _plot_losses(self) -> None:
        """
        Plots the training and validation losses

        Returns
        -------
        None
        """
        plt.plot(self.train_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")

        plt.plot(self.val_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend(["Training Loss", "Validation Loss"])
        plt.show()    