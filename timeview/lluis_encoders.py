import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

def is_dynamic_bias_enabled(config):
    if hasattr(config, 'dynamic_bias'):
        return config.dynamic_bias
    else:
        return False

class GNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_features = config.n_features
        self.n_basis = config.n_basis
        self.hidden_sizes = config.encoder.hidden_sizes
        self.dropout_p = config.encoder.dropout_p

        assert len(self.hidden_sizes) > 0

        self.latent_size = self.n_basis + 1 if is_dynamic_bias_enabled(config) else self.n_basis
        self.edge_index = self._build_edge_index(self.n_features)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.n_features, self.hidden_sizes[0]))
        for i in range(len(self.hidden_sizes) - 1):
            self.convs.append(GCNConv(self.hidden_sizes[i], self.hidden_sizes[i + 1]))

        self.fc = nn.Linear(self.hidden_sizes[-1], self.latent_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_p)

    def _build_edge_index(self, num_features):
        row = []
        col = []
        for i in range(num_features):
            for j in range(num_features):
                if i != j:
                    row.append(i)
                    col.append(j)
        return torch.tensor([row, col], dtype=torch.long)

    def forward(self, x):
        device = x.device
        edge_index = self.edge_index.to(device)
        batch_size, n_features = x.size()
        assert n_features == self.n_features
        x = x.repeat_interleave(self.n_features, dim=0)
        batch = torch.arange(batch_size, device=device).repeat_interleave(self.n_features)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x