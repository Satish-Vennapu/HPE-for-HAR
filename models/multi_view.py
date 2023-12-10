import torch
import torch.nn as nn
from torch_geometric.data import Batch

from models.transformer import Transformer
from models.gcn import PoseGCN


class MultiViewActionRecognizer(nn.Module):
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
        aggregator: str = "average",
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
        super(MultiViewActionRecognizer, self).__init__()
        self.gcn1 = PoseGCN(
            gcn_num_features, gcn_hidden_dim1, gcn_hidden_dim2, gcn_output_dim
        )
        self.gcn2 = PoseGCN(
            gcn_num_features, gcn_hidden_dim1, gcn_hidden_dim2, gcn_output_dim
        )
        self.gcn3 = PoseGCN(
            gcn_num_features, gcn_hidden_dim1, gcn_hidden_dim2, gcn_output_dim
        )
        self.transformer = Transformer(
            transformer_d_model,
            transformer_nhead,
            transformer_num_layers,
            transformer_num_features,
            transformer_dropout,
            transformer_dim_feedforward,
            num_classes=transformer_num_classes,
        )

        self.aggregator = aggregator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.aggregator == "self_attn":
            self.self_attention = nn.MultiheadAttention(
                embed_dim=gcn_output_dim * 3, num_heads=1
            )
            self.projection = nn.Linear(gcn_output_dim * 3, gcn_output_dim)

        elif self.aggregator == "linear":
            self.linear = nn.Linear(gcn_output_dim * 3, gcn_output_dim)

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
        outputs = []

        for item in batch:
            view1_embedding = self.gcn1(item[0])
            view2_embedding = self.gcn2(item[1])
            view3_embedding = self.gcn3(item[2])

            if self.aggregator == "linear":
                output = torch.cat(
                    (view1_embedding, view2_embedding, view3_embedding), dim=-1
                )
                output = self.linear(output)
            elif self.aggregator == "self_attn":
                concat_emb = torch.cat(
                    [view1_embedding, view2_embedding, view3_embedding], dim=-1
                )
                attn_output, _ = self.self_attention(
                    concat_emb.unsqueeze(0),
                    concat_emb.unsqueeze(0),
                    concat_emb.unsqueeze(0),
                )
                output = self.projection(attn_output.squeeze(0))
            else:
                output = (view1_embedding + view2_embedding + view3_embedding) / 3

            output = self.transformer(output.unsqueeze(0).to(self.device))
            outputs.append(output)

        return torch.stack(outputs).squeeze(1)
