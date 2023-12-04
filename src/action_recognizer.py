import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformer import TransformerBinaryClassifier
from gcn import PoseGCN, create_pose_graph

from typing import Tuple

class ActionRecogniser(nn.Module):
    def __init__(
            self,
            gcn_num_features : int,
            gcn_hidden_dim1 : int,
            gcn_hidden_dim2 : int,
            gcn_output_dim : int,
            transformer_d_model : int,
            transformer_nhead : int,
            transformer_num_layers : int,
            transformer_num_features : int,
            transformer_dropout : float = 0.1,
            transformer_dim_feedforward : int = 2048,
            edge_index : torch.Tensor = None
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
        self.edge_index = edge_index
        self.gcn = PoseGCN(
            gcn_num_features,
            gcn_hidden_dim1,
            gcn_hidden_dim2,
            gcn_output_dim
        )
        self.transformer = TransformerBinaryClassifier(
            transformer_d_model,
            transformer_nhead,
            transformer_num_layers,
            transformer_num_features,
            transformer_dropout,
            transformer_dim_feedforward
        )

    def forward(self, kps : torch.Tensor) -> torch.Tensor:
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
        seq_length = kps.shape[1]

        frame_embeddings = []
        for t in range(seq_length):
            pose_graph = create_pose_graph(kps[0, t, :, :], self.edge_index).to("cuda")
            embedding = self.gcn(pose_graph)
            frame_embeddings.append(embedding)

        sequence_embeddings = torch.stack(frame_embeddings)
        classification = self.transformer(sequence_embeddings.unsqueeze(0))

        return classification
    
class Solver:

    def __init__(
            self,
            gcn_num_features : int,
            gcn_hidden_dim1 : int,
            gcn_hidden_dim2 : int,
            gcn_output_dim : int,
            transformer_d_model : int,
            transformer_nhead : int,
            transformer_num_layers : int,
            transformer_num_features : int,
            transformer_dropout : float = 0.1,
            transformer_dim_feedforward : int = 2048,
            edge_index : torch.Tensor = None,
            lr : float = 0.001
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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            edge_index
        ).to(self.device)

        self.lr = lr

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

        self.train_loss = []
        self.val_loss = []

    def train(self, train_loader : torch.utils.data.DataLoader, val_loader : torch.utils.data.DataLoader, epochs : int = 20) -> None:
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

        best_val_loss = float('inf')

        for epoch in range(epochs):
            epoch_loss, epoch_correct, epoch_count = self.train_one_epoch(train_loader)
            self.train_loss.append(epoch_loss)
            print(f"\nepoch: {epoch} | epoch loss: {epoch_loss}        | epoch accuracy: {epoch_correct / epoch_count}")
            
            epoch_val_loss, epoch_val_correct, epoch_val_count = self.evaluate(val_loader)
            self.val_loss.append(epoch_val_loss)
            print(f"epoch: {epoch} | epoch val loss: {epoch_val_loss}   | epoch val accuracy: {epoch_val_correct / epoch_val_count}")
            # self.scheduler.step()

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')

        self._plot_losses()

    def train_one_epoch(self, train_loader : torch.utils.data.DataLoader) -> Tuple[float, int, int]:
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
            predictions = self.model(batch['kps'].to(self.device))
            labels = batch['label'].to(self.device)

            loss = self.criterion(predictions, labels)

            correct = predictions.argmax(axis=1) == labels

            epoch_correct += correct.sum().item()
            epoch_count += correct.size(0)

            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.optimizer.step()

        return epoch_loss, epoch_correct, epoch_count

    def evaluate(self, val_loader : torch.utils.data.DataLoader) -> Tuple[float, int, int]:
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
                predictions = self.model(batch['kps'].to(self.device).long())
                labels = batch['label'].to(self.device)
                
                val_loss = self.criterion(predictions, labels)
                
                correct = predictions.argmax(axis=1) == labels

                val_epoch_correct += correct.sum().item()
                val_epoch_count += correct.size(0)
                val_epoch_loss += val_loss.item()

        return val_epoch_loss, val_epoch_correct, val_epoch_count
    
    def _plot_losses(self) -> None:
        """
        Plots the training and validation losses

        Returns
        -------
        None
        """
        plt.plot(self.train_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')

        plt.plot(self.val_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.show()  