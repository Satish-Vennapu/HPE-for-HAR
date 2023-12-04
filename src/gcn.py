import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class PoseGCN(torch.nn.Module):
    def __init__(
            self,
            num_features : int,
            hidden_dim1 : int,
            hidden_dim2 : int,
            output_dim : int
        ) -> None:
        """
        Parameters
        ----------
        num_features : int
            Number of features in the input sequence
        hidden_dim1 : int
            Dimension of the first hidden layer of the GCN
        hidden_dim2 : int
            Dimension of the second hidden layer of the GCN
        output_dim : int
            Dimension of the output layer of the GCN
        """
        super(PoseGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, output_dim)

    def forward(self, data : Data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : Data
            Pose Graph  
            
        Returns
        -------
        torch.Tensor
            Output of the GCN of shape (batch_size, output_dim)
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.mean(x, dim=0)
        return x