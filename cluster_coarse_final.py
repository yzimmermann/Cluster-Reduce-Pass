# Import required libraries
import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from torch_geometric.nn import GCN
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch, pool_edge, pool_pos
from torch_geometric.nn import max_pool
from torch_geometric.utils import scatter
from sklearn.mixture import GaussianMixture
import fast_pytorch_kmeans as fpk
import json
import os
from joblib import Parallel, delayed
from torchmetrics.classification import MultilabelAveragePrecision
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.transforms import AddRandomWalkPE

# Set dataset name
LRGB_dat = 'Peptides-struct'  # Options: 'Peptides-func' or 'Peptides-struct'
LRGB_sp = LRGB_dat.split('-')[1]

# Set positional encoding (PE)
# control how to load dataset based on PE
pos_encoding = True
pos_encoding_type = 'LAPE' # can be LAPE or RWPE
k = 4 # number of eigenvectors
walk_length = 25 # walking length needed for RWPE

#Coarsening param
coarsen_graph = 'yes' #either 'yes' or 'no', anything that is not a 'yes' will be treated as a 'no'
num_layers_before = 2 #num layers of GCN before clustering and coarsening
num_layers_after = 3 #num layers of GCN after clustering and coarsening

if pos_encoding:
    if pos_encoding_type == 'LAPE':
        tranformer = AddLaplacianEigenvectorPE(k=k,attr_name=None)
        # Load dataset and splits
        dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat, pre_transform = tranformer)
        train_dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat, split='train', pre_transform = tranformer)
        val_dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat, split='val', pre_transform = tranformer)
        test_dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat, split='test', pre_transform = tranformer)
    else:
        tranformer = AddRandomWalkPE(walk_length=walk_length, attr_name=None)
        # Load dataset and splits
        dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat, pre_transform=tranformer)
        train_dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat, split='train', pre_transform=tranformer)
        val_dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat, split='val', pre_transform=tranformer)
        test_dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat, split='test', pre_transform=tranformer)
else:
    # Load dataset and splits
    dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat)
    train_dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat, split='train')
    val_dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat, split='val')
    test_dataset = LRGBDataset(root='data/LRGBDataset', name=LRGB_dat, split='test')


print(LRGB_dat)
if pos_encoding: print('Positional Encoding type:',pos_encoding_type)
print('num eigenvectors: ', k)
print('num features = ', dataset.num_node_features)
print('num classes = ', dataset.num_classes)


# Clustering class for KMeans or GMM
class Clustering:
    def __init__(self, clustering_type: str, n_clusters: int, random_state: int = 0):
        """
        Initialize clustering model (KMeans or GMM).

        Args:
            clustering_type (str): Clustering type ('KMeans' or 'GMM').
            n_clusters (int): Number of clusters.
            random_state (int): Random seed for reproducibility.
        """
        self.type = clustering_type
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

        if clustering_type == 'KMeans':
            self.model = fpk.KMeans(n_clusters=n_clusters)
        elif clustering_type == 'GMM':
            self.model = GaussianMixture(n_components=n_clusters)
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


def coarsen_graph_proportional(cluster: torch.Tensor, data: Data, sampling_ratio: float = 0.5,
                               min_samples: int = 1, max_samples: int = 10) -> Data:
    """
    Coarsens a graph by sampling nodes proportionally to cluster sizes while maintaining
    all valid inter-node connections.

    Args:
        cluster (Tensor): Tensor assigning each node to a specific cluster
        data (Data): Input graph data object
        sampling_ratio (float): Fraction of nodes to sample from each cluster (default: 0.5)
        min_samples (int): Minimum number of nodes to sample per cluster (default: 1)
        max_samples (int): Maximum number of nodes to sample per cluster (default: 10)

    Returns:
        Data: Coarsened graph data object with proportional samples per cluster
    """
    # Map nodes to clusters
    cluster = cluster.to(device)
    data = data.to(device)
    unique_clusters, perm = torch.unique(cluster, return_inverse=True)
    num_clusters = unique_clusters.size(0)

    # Sample nodes proportionally from each cluster
    sampled_indices = []
    num_samples_per_cluster = []  # Keep track of how many samples per cluster

    for cluster_id in range(num_clusters):
        # Get nodes in this cluster
        cluster_mask = perm == cluster_id
        cluster_indices = torch.nonzero(cluster_mask, as_tuple=True)[0]
        cluster_features = data.x[cluster_indices]

        # Calculate number of samples for this cluster
        cluster_size = len(cluster_indices)
        num_samples = int(max(min(cluster_size * sampling_ratio, max_samples), min_samples))
        num_samples_per_cluster.append(num_samples)

        # Sample nodes
        if num_samples >= cluster_size:
            # Take all nodes if we want more samples than available
            sampled_idx = cluster_indices
        else:
            # Random sampling without replacement
            weights = torch.norm(cluster_features, dim =1)
            sampled_idx = cluster_indices[torch.multinomial(weights,num_samples,replacement = False)]

        sampled_indices.append(sampled_idx)

    # Concatenate all sampled indices
    sampled_indices = torch.cat(sampled_indices)

    # Create new x features from sampled nodes
    x = data.x[sampled_indices]

    # Create a mapping for new node indices
    old_to_new = torch.empty(data.x.size(0), dtype=torch.long, device=device)
    current_idx = 0
    for cluster_id, num_samples in enumerate(num_samples_per_cluster):
        cluster_mask = perm == cluster_id
        old_indices = torch.nonzero(cluster_mask, as_tuple=True)[0]
        new_indices = torch.arange(current_idx, current_idx + num_samples, device=device)
        # Map each old index to its corresponding new index if sampled, or the first new index if not
        mapped_indices = torch.ones_like(old_indices, device=device) * new_indices[0]
        sampled_mask = torch.isin(old_indices, sampled_indices)
        mapped_indices[sampled_mask] = new_indices
        old_to_new[old_indices] = mapped_indices
        current_idx += num_samples

    # Remap edges to new indices
    edge_index = old_to_new[data.edge_index]
    edge_attr = data.edge_attr

    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    # Create new cluster assignments for sampled nodes
    new_cluster = torch.zeros(len(sampled_indices), dtype=torch.long, device=device)
    current_idx = 0
    for cluster_id, num_samples in enumerate(num_samples_per_cluster):
        new_cluster[current_idx:current_idx + num_samples] = cluster_id
        current_idx += num_samples

    # Update batch information for sampled nodes
    batch = scatter(data.batch, perm, dim=0, reduce='max')
    batch = batch[new_cluster]
    # print(edge_index.shape)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                batch=batch, cluster=new_cluster)
    # return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
    # batch=batch)


# Coarsen a graph based on clustering
def coarsen_graph(cluster: torch.Tensor, data: Data, reduce: str = 'mean') -> Data:
    """
    Coarsen the graph by pooling nodes based on cluster assignments.

    Args:
        cluster (Tensor): Cluster assignments.
        data (Data): Input graph data.
        reduce (str): Aggregation method for node features ('mean', 'max', 'sample', etc.).

    Returns:
        Data: Coarsened graph data.
    """
    if reduce == 'sample':
        return coarsen_graph_proportional(cluster,data)
    else:
        cluster, perm = torch.unique(cluster, return_inverse=True)

        x = scatter(data.x, perm, dim=0, reduce=reduce)
        edge_index = perm[data.edge_index]
        edge_attr = data.edge_attr

        # Remove self-loops and duplicate edges
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]

        batch = scatter(data.batch, perm, dim=0, reduce='max')

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

# MLP-based graph prediction head
class MLPGraphHead(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        """
        Initialize an MLP-based graph prediction head.

        Args:
            hidden_channels (int): Input dimension.
            out_channels (int): Output dimension.
        """
        super().__init__()
        self.pooling_fun = global_mean_pool
        dropout = 0.1
        L = 3

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

# GCN model with graph coarsening
class GCNWithCoarsening(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, clustering_type='KMeans', n_clusters=5):
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
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers_before,
            act='gelu',
            dropout=0.1,
            norm='batch',
            norm_kwargs={'track_running_stats': False}
        )
        if coarsen_graph == 'yes':
            self.clustering = Clustering(clustering_type=clustering_type, n_clusters=n_clusters)
            self.coarsen_projection = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gcn_post_coarsen = GCN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers_after,
            act='gelu',
            dropout=0.1,
            norm='batch',
            norm_kwargs={'track_running_stats': False}
        )
        self.head = MLPGraphHead(hidden_channels, out_channels)

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
        if coarsen_graph == 'yes':
            cluster = self.clustering.fit(x, batch)
            coarsened_data = coarsen_graph(cluster, Data(x=x, edge_index=edge_index, batch=batch), reduce='sample') # by setting reduce ='sample' we perfomr sampling of a super node instead of using mean (default)
            coarsened_data.x = self.coarsen_projection(coarsened_data.x)
            x = self.gcn_post_coarsen(coarsened_data.x, coarsened_data.edge_index)
            return self.head(x, coarsened_data.batch)
        else:
            x = self.gcn_post_coarsen(x=x, edge_index=edge_index)

        return self.head(x,batch)

# Rest of the script contains dataset preparation, training, and evaluation.




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model
model = GCNWithCoarsening(in_channels=dataset.num_features,
                          hidden_channels=235,
                          out_channels=dataset.num_classes,
                          n_clusters=22,
                          clustering_type='KMeans').to(device)
print(model)


# Learning rate scheduler
# Example usage
warmup_epochs = 5
total_epochs = 250
# Define the warmup and cosine decay schedule
def cosine_with_warmup(epoch):
    if epoch < warmup_epochs:
        # Linear warmup
        return epoch / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))



print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

if device == 'cuda':
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, pin_memory=True,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True,num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True,num_workers=2)
else:
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the loss criterion
if LRGB_dat == 'Peptides-struct':
    criterion = torch.nn.L1Loss()  # For MAE-based regression
else:
    criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = LambdaLR(optimizer, lr_lambda=cosine_with_warmup)
if LRGB_dat == 'Peptides-func':
    AP = MultilabelAveragePrecision(num_labels=train_loader.dataset.num_classes, average='macro')

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        data.x = data.x.to(device)
        optimizer.zero_grad()
        out = model(data)
        target = data.y.float().to(device)  # Ensure target is on the same device
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Evaluation Function
def test(loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0  # Track loss for the scheduler

    with torch.no_grad():
        for data in loader:
            # data = compute_laplacian_pe(data)
            data = data.to(device)
            data.x = data.x.float()
            out = model(data)
            loss = criterion(out, data.y.float())  # Compute loss
            total_loss += loss.item()
            if LRGB_dat == 'Peptides-func':
                pred = torch.sigmoid(out).cpu().numpy()
            else:
                pred = out.cpu().numpy()
            labels = data.y.cpu().numpy()  # Squeeze to remove single-dimensional entries
            all_preds.append(pred)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    if LRGB_dat == 'Peptides-func':
        AP.reset()
        AP.update(torch.tensor(all_preds), torch.tensor(all_labels).long())
        ap_score = AP.compute().item()
        return ap_score
    else:
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds, multioutput='uniform_average')  # Average across all tasks
        return mae, r2, total_loss / len(loader)




# Training loop with logging and saving results
seed = 42
log_directory = './training_logs'

if not os.path.exists(log_directory):
    os.makedirs(log_directory)

if LRGB_dat=='Peptides-func':
    second_sota_gcn_value = 0.6860  # SOTA GCN baseline value
    first_sota_gcn_value = 0.5930 #SOTA in LRGB paper
else:
    second_sota_gcn_value = 0.2460  # SOTA GCN baseline value
    first_sota_gcn_value = 0.3496 #SOTA in LRGB paper

logs = []

for epoch in range(1, total_epochs + 1):
    loss = train()
    if LRGB_dat=='Peptides-func':
        val_ap = test(val_loader)
        test_ap = test(test_loader)
        train_ap= test(train_loader)
        log_entry = {
            'epoch': int(epoch),
            'loss': float(loss),
            'val_metric': float(val_ap),
            'test_metric': float(test_ap)
        }
        logs.append(log_entry)
        print(f"Seed {seed}, Epoch {epoch:03d}, Loss: {loss:.4f}, Val AP: {val_ap:.4f}, Test AP: {test_ap:.4f}")
    else:
        val_mae, val_r2, val_loss = test(val_loader)
        test_mae, test_r2, _ = test(test_loader)
        train_mae, train_r2, _ = test(train_loader)
        log_entry = {
            'epoch': int(epoch),
            'loss': float(loss),
            'val_metric': float(val_mae),
            'val_r2': float(val_r2),
            'test_metric': float(test_mae),
            'test_r2': float(test_r2)
        }
        logs.append(log_entry)

        print(f"Seed {seed}, Epoch {epoch:03d}, Loss: {loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}")

        # Save logs to a JSON file
log_file_path = os.path.join(log_directory, f'seed_{seed}_logs_{LRGB_sp}.json')
with open(log_file_path, 'w') as f:
    json.dump(logs, f, indent=4)

        # Save the final plot
plot_file_path = os.path.join(log_directory, f'seed_{seed}_plot_{LRGB_sp}.png')
plt.figure(figsize=(10, 6))
plt.plot([log['epoch'] for log in logs], [log['val_metric'] for log in logs], label='Validation AP')
plt.plot([log['epoch'] for log in logs], [log['test_metric'] for log in logs], label='Test AP')
plt.axhline(y=second_sota_gcn_value, color='black', linestyle='--', label='2nd SOTA GCN')
plt.axhline(y=first_sota_gcn_value, color='black', linestyle='--', label='1st SOTA GCN')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title(f'Seed {seed} Training Progress')
plt.legend()
plt.savefig(plot_file_path)
plt.close()
