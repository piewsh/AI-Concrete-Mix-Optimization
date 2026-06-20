import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import scatter

class EdgeModel(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim, dropout=0.2):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim)
        )
    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim, dropout=0.2):
        super(NodeModel, self).__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )
    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        agg = scatter(edge_attr, col, dim=0, reduce='mean', dim_size=x.size(0))
        out = torch.cat([x, agg, u[batch]], dim=1)
        return self.node_mlp(out)

class GlobalModel(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim, dropout=0.2):
        super(GlobalModel, self).__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, global_dim)
        )
    def forward(self, x, edge_index, edge_attr, u, batch):
        node_agg = scatter(x, batch, dim=0, reduce='mean')
        edge_agg = scatter(edge_attr, batch[edge_index[0]], dim=0, reduce='mean', dim_size=u.size(0))
        out = torch.cat([node_agg, edge_agg, u], dim=1)
        return self.global_mlp(out)

class ProcessorBlock(nn.Module):
    def __init__(self, latent_dim, hidden_dim, dropout=0.2):
        super(ProcessorBlock, self).__init__()
        self.meta_layer = MetaLayer(
            edge_model=EdgeModel(latent_dim, latent_dim, latent_dim, hidden_dim, dropout),
            node_model=NodeModel(latent_dim, latent_dim, latent_dim, hidden_dim, dropout),
            global_model=GlobalModel(latent_dim, latent_dim, latent_dim, hidden_dim, dropout)
        )
    def forward(self, x, edge_index, edge_attr, u, batch):
        dx, de, du = self.meta_layer(x, edge_index, edge_attr, u, batch)
        return x + dx, edge_attr + de, u + du

class Encoder(nn.Module):
    def __init__(self, node_in, node_latent, edge_in, edge_latent, global_latent, dropout=0.2):
        super(Encoder, self).__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, node_latent),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_latent, node_latent)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in, edge_latent),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_latent, edge_latent)
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(1, global_latent)
        )
    def forward(self, x, edge_attr):
        x_enc = self.node_encoder(x)
        e_enc = self.edge_encoder(edge_attr)
        u_enc = self.global_encoder(torch.zeros((1, 1), device=x.device))
        return x_enc, e_enc, u_enc

class Decoder(nn.Module):
    def __init__(self, latent_dim, dropout=0.2):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 1)
        )
    def forward(self, x):
        return self.mlp(x)

class ConcreteGNN(nn.Module):
    def __init__(self,
                 node_in=8,
                 node_latent=128,
                 edge_in=1,
                 edge_latent=128,
                 global_latent=128,
                 num_processor=6,
                 hidden_dim=256,
                 dropout=0.2):
        super(ConcreteGNN, self).__init__()
        self.encoder = Encoder(node_in, node_latent, edge_in, edge_latent, global_latent, dropout)
        self.processors = nn.ModuleList([
            ProcessorBlock(node_latent, hidden_dim, dropout) for _ in range(num_processor)
        ])
        self.decoder = Decoder(node_latent, dropout)
    def forward(self, data):
        x, edge_attr, u = self.encoder(data.x, data.edge_attr)
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        for processor in self.processors:
            x, edge_attr, u = processor(x, data.edge_index, edge_attr, u, batch)
        out = self.decoder(x)
        return out.squeeze()
