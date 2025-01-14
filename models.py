import torch
import fast_pytorch_kmeans as fpk
import numpy as np

from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, GCN

from coarsening import coarsen_graph


class MLPGraphHead(torch.nn.Module):
    def __init__(self, cfg):
        """
        Initialize an MLP-based graph prediction head.

        Args:
            hidden_channels (int): Input dimension.
            out_channels (int): Output dimension.
        """
        super().__init__()
        self.pooling_fun = global_mean_pool
        dropout = cfg["mlp"]["dropout"]
        L = cfg["mlp"]["L"]
        hidden_channels = cfg["model"]["hidden_channels"]
        out_channels = cfg["model"]["out_channels"]

        layers = []
        for _ in range(L - 1):
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_channels, out_channels))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x, batch):
        """
        Forward pass through the MLP head.

        Args:
            x (Tensor): Node features.
            batch (Tensor): Batch indices.

        Returns:
            Tensor: Predictions.
        """
        x = self.pooling_fun(x, batch)
        return self.mlp(x)


class newGCN(torch.nn.Module):
    def __init__(self, in_channels, cfg):
        super(newGCN, self).__init__()

        # Define GCN layers with edge attributes
        self.gcn = GCN(
            in_channels=in_channels,
            hidden_channels=cfg["model"]["hidden_channels"],
            out_channels=cfg["model"]["hidden_channels"],
            num_layers=cfg["model"]["num_layers"],
            act=cfg["model"]["act"],
            dropout=cfg["model"]["dropout"],
            norm=cfg["model"]["norm"],
            norm_kwargs={'track_running_stats': False}
        )

        # Replace the prediction head with MLPGraphHead
        self.head = MLPGraphHead(cfg)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Apply GCN layers with edge attributes
        x = self.gcn(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return self.head(x, batch)


class Clustering:
    def __init__(self, cfg):
        """
        Initialize clustering model (KMeans or GMM).

        Args:
            clustering_type (str): Clustering type ('KMeans' or 'GMM').
            n_clusters (int): Number of clusters.
            random_state (int): Random seed for reproducibility.
        """
        self.type = cfg["clustering"]["clustering_type"]
        self.n_clusters = cfg["clustering"]["n_clusters"]
        self.random_state = cfg["clustering"]["random_state"]
        self.model = None

        if cfg["clustering"]["clustering_type"] == 'KMeans':
            self.model = fpk.KMeans(n_clusters=cfg["clustering"]["n_clusters"])
        elif cfg["clustering"]["clustering_type"] == 'GMM':
            self.model = GaussianMixture(n_components=cfg["clustering"]["n_clusters"])
        else:
            raise ValueError("Invalid clustering type. Choose 'KMeans' or 'GMM'.")

    def fit(self, features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Fit the clustering model and assign clusters to nodes.

        Args:
            features (torch.Tensor): Node features.
            batch (torch.Tensor): Batch indices.

        Returns:
            torch.Tensor: Cluster assignments.
        """
        features_np = features.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        unique_batches = np.unique(batch_np)

        def process_batch(b):
            mask = batch_np == b
            features_tensor = torch.tensor(features_np[mask], dtype=torch.float32)
            return self.model.fit_predict(features_tensor)

        clusters = Parallel(n_jobs=-1)(delayed(process_batch)(b) for b in unique_batches)

        combined_clusters = np.zeros(features_np.shape[0], dtype=int)
        offset = 0
        for b, cluster in zip(unique_batches, clusters):
            mask = batch_np == b
            combined_clusters[mask] = cluster + offset
            offset += torch.max(cluster) + 1

        return torch.tensor(combined_clusters, dtype=torch.long, device=features.device)


class GCNWithCoarsening(torch.nn.Module):
    def __init__(self, cfg):
        """
        Initialize a GCN model with graph coarsening.

        Args:
            in_channels (int): Input feature dimension.
            hidden_channels (int): Hidden feature dimension.
            out_channels (int): Output feature dimension.
            clustering_type (str): Clustering method ('KMeans' or 'GMM').
            n_clusters (int): Number of clusters.
        """
        super().__init__()
        self.gcn_conv_layers = GCN(
            in_channels=cfg["model"]["in_channels"],
            hidden_channels=cfg["model"]["hidden_channels"],
            out_channels=cfg["model"]["hidden_channels"],
            num_layers=cfg["model"]["num_layers_before"],
            act=cfg["model"]["act"],
            dropout=cfg["model"]["dropout"],
            norm=cfg["model"]["norm"],
            norm_kwargs={'track_running_stats': False}
        )
        
        self.clustering = Clustering(cfg=cfg)
        self.coarsen_projection = torch.nn.Linear(hidden_channels=cfg["model"]["hidden_channels"], hidden_channels=cfg["model"]["hidden_channels"])
        self.gcn_post_coarsen = GCN(
            in_channels=cfg["model"]["in_channels"],
            hidden_channels=cfg["model"]["hidden_channels"],
            out_channels=cfg["model"]["hidden_channels"],
            num_layers=cfg["coarsening"]["num_layers_after"],
            act=cfg["model"]["act"],
            dropout=cfg["model"]["dropout"],
            norm=cfg["model"]["norm"],
            norm_kwargs={'track_running_stats': False}
        )
        self.head = MLPGraphHead(cfg=cfg)

    def forward(self, data):
        """
        Forward pass for the GCN with coarsening.

        Args:
            data (Data): Input graph data.

        Returns:
            Tensor: Predictions.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        x = self.gcn_conv_layers(x=x, edge_index=edge_index)

        cluster = self.clustering.fit(x, batch)
        coarsened_data = coarsen_graph(cluster, Data(x=x, edge_index=edge_index, batch=batch), reduce='mean') # by setting reduce ='sample' we perfomr sampling of a super node instead of using mean (default)
        coarsened_data.x = self.coarsen_projection(coarsened_data.x)
        x = self.gcn_post_coarsen(coarsened_data.x, coarsened_data.edge_index)

        return self.head(x, coarsened_data.batch)