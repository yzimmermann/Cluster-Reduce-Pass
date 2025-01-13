# Import required libraries
import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.nn import global_mean_pool, GCN
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.utils import scatter
from sklearn.mixture import GaussianMixture
import fast_pytorch_kmeans as fpk
import json
import os
from joblib import Parallel, delayed
from torchmetrics.classification import MultilabelAveragePrecision
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set dataset name and parameters
LRGB_dat = 'Peptides-struct'
pos_encoding = True
pos_encoding_type = 'LAPE'  # can be LAPE or RWPE
k = 4  # number of eigenvectors
walk_length = 25
coarsen_graph_flag = True
num_layers_before = 2
num_layers_after = 3


# Helper function to load dataset
def load_dataset(root, name, split=None, transformer=None):
    return LRGBDataset(root=root, name=name, split=split, pre_transform=transformer)


# Load datasets
transformer = None
if pos_encoding:
    transformer = AddLaplacianEigenvectorPE(k=k) if pos_encoding_type == 'LAPE' else AddRandomWalkPE(walk_length=walk_length)

train_dataset = load_dataset('data/LRGBDataset', LRGB_dat, split='train', transformer=transformer)
val_dataset = load_dataset('data/LRGBDataset', LRGB_dat, split='val', transformer=transformer)
test_dataset = load_dataset('data/LRGBDataset', LRGB_dat, split='test', transformer=transformer)


# Clustering class for KMeans or GMM
class Clustering:
    def __init__(self, clustering_type: str, n_clusters: int, random_state: int = 0):
        self.type = clustering_type
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = fpk.KMeans(n_clusters=n_clusters) if clustering_type == 'KMeans' else GaussianMixture(n_components=n_clusters)

    def fit(self, features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        features_np = features.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        unique_batches = np.unique(batch_np)

        def process_batch(b):
            mask = batch_np == b
            return self.model.fit_predict(features_np[mask])

        clusters = Parallel(n_jobs=-1)(delayed(process_batch)(b) for b in unique_batches)
        combined_clusters = np.concatenate(clusters)
        return torch.tensor(combined_clusters, dtype=torch.long, device=features.device)


# Coarsen a graph based on clustering
def coarsen_graph(cluster: torch.Tensor, data: Data, reduce: str = 'mean') -> Data:
    cluster, perm = torch.unique(cluster, return_inverse=True)
    x = scatter(data.x, perm, dim=0, reduce=reduce)
    edge_index = perm[data.edge_index]
    edge_attr = data.edge_attr
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    batch = scatter(data.batch, perm, dim=0, reduce='max')
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


# MLP-based graph prediction head
class MLPGraphHead(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
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
        x = self.pooling_fun(x, batch)
        return self.mlp(x)


# GCN model with graph coarsening
class GCNWithCoarsening(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, clustering_type='KMeans', n_clusters=5):
        super().__init__()
        self.gcn_conv_layers = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers_before,
            act='gelu',
            dropout=0.1,
            norm='batch'
        )
        self.clustering = Clustering(clustering_type=clustering_type, n_clusters=n_clusters) if coarsen_graph_flag else None
        self.coarsen_projection = torch.nn.Linear(hidden_channels, hidden_channels) if coarsen_graph_flag else None
        self.gcn_post_coarsen = GCN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers_after,
            act='gelu',
            dropout=0.1,
            norm='batch'
        )
        self.head = MLPGraphHead(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.gcn_conv_layers(x=x, edge_index=edge_index)
        if coarsen_graph_flag:
            cluster = self.clustering.fit(x, batch)
            coarsened_data = coarsen_graph(cluster, Data(x=x, edge_index=edge_index, batch=batch), reduce='mean')
            coarsened_data.x = self.coarsen_projection(coarsened_data.x)
            x = self.gcn_post_coarsen(coarsened_data.x, coarsened_data.edge_index)
        else:
            x = self.gcn_post_coarsen(x=x, edge_index=edge_index)
        return self.head(x, batch)


# Create optimizer and scheduler
def create_optimizer_and_scheduler(model, lr=0.001, warmup_epochs=5, total_epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))))
    return optimizer, scheduler


# Initialize model, optimizer, and scheduler
model = GCNWithCoarsening(
    in_channels=train_dataset.num_node_features,
    hidden_channels=235,
    out_channels=train_dataset.num_classes,
    n_clusters=22,
    clustering_type='KMeans'
).to(device)
optimizer, scheduler = create_optimizer_and_scheduler(model)


# Define the loss criterion
criterion = torch.nn.L1Loss() if LRGB_dat == 'Peptides-struct' else torch.nn.BCEWithLogitsLoss()
AP = MultilabelAveragePrecision(num_labels=train_dataset.num_classes, average='macro') if LRGB_dat == 'Peptides-func' else None


# Training function
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        target = data.y.float().to(device)
        loss = criterion(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Evaluation function
def test(loader):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.float())
            total_loss += loss.item()
            preds = torch.sigmoid(out).cpu().numpy() if LRGB_dat == 'Peptides-func' else out.cpu().numpy()
            labels = data.y.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)
    all_preds, all_labels = np.concatenate(all_preds), np.concatenate(all_labels)
    if LRGB_dat == 'Peptides-func':
        AP.reset()
        AP.update(torch.tensor(all_preds), torch.tensor(all_labels).long())
        return AP.compute().item()
    return mean_absolute_error(all_labels, all_preds), r2_score(all_labels, all_preds), total_loss / len(loader)


# Prepare data loaders
num_workers = os.cpu_count() // 2
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, pin_memory=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=num_workers)


# Training loop
seed = 42
log_directory = './training_logs'
os.makedirs(log_directory, exist_ok=True)
logs = []

for epoch in range(1, total_epochs + 1):
    loss = train()
    if LRGB_dat == 'Peptides-func':
        val_ap = test(val_loader)
        test_ap = test(test_loader)
        logs.append({'epoch': epoch, 'loss': loss, 'val_metric': val_ap, 'test_metric': test_ap})
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val AP: {val_ap:.4f}, Test AP: {test_ap:.4f}")
    else:
        val_mae, val_r2, val_loss = test(val_loader)
        test_mae, test_r2, _ = test(test_loader)
        logs.append({'epoch': epoch, 'loss': loss, 'val_metric': val_mae, 'val_r2': val_r2, 'test_metric': test_mae, 'test_r2': test_r2})
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}")

# Save logs
log_file_path = os.path.join(log_directory, f'seed_{seed}_logs.json')
with open(log_file_path, 'w') as f:
    json.dump(logs, f, indent=4)

# Plot results
plot_file_path = os.path.join(log_directory, f'seed_{seed}_plot.png')
plt.figure(figsize=(10, 6))
plt.plot([log['epoch'] for log in logs], [log['val_metric'] for log in logs], label='Validation Metric')
plt.plot([log['epoch'] for log in logs], [log['test_metric'] for log in logs], label='Test Metric')
plt.xlabel('Epoch')
plt.ylabel('MAE' if LRGB_dat == 'Peptides-struct' else 'AP')
plt.title(f'Seed {seed} Training Progress')
plt.legend()
plt.savefig(plot_file_path)
plt.close()
