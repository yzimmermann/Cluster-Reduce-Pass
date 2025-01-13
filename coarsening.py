import torch

from torch_geometric.data import Data
from torch_geometric.utils import scatter


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            perm_indices = torch.randperm(cluster_size, device=device)[:num_samples]
            sampled_idx = cluster_indices[perm_indices]

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