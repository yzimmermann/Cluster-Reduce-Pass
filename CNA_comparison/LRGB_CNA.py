import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import (
    ARMAConv, ChebConv, CuGraphSAGEConv, GATConv, GCNConv, GraphConv, LEConv,
    Linear as TGLinear, MFConv, SAGEConv, Sequential, SGConv, SSGConv, TransformerConv,
    BatchNorm
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.model_params import LayerType, ModelParams as mp, ActivationType, ReclusterOption
from clustering.rationals_on_clusters import RationalOnCluster

from torchmetrics.classification import MultilabelAveragePrecision
from sklearn.metrics import r2_score, mean_absolute_error

#################################################################
#                     Data and Transform Setup
#################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = "Peptides-struct"

class FloatTransform:
    """
    Ensures the node features (x) and labels (y) are converted to float.
    This prevents repeated casting in the training loop.
    """

    def __call__(self, data):
        if data.x is not None:
            data.x = data.x.float()
        if data.y is not None:
            data.y = data.y.float()
        return data


# Load datasets with transform
train_dataset = LRGBDataset(
    root='data/LRGBDataset',
    name=dataset_name,
    split='train',
    transform=FloatTransform()
)
val_dataset = LRGBDataset(
    root='data/LRGBDataset',
    name=dataset_name,
    split='val',
    transform=FloatTransform()
)
test_dataset = LRGBDataset(
    root='data/LRGBDataset',
    name=dataset_name,
    split='test',
    transform=FloatTransform()
)

# Print dataset sizes
print(f"Dataset name: {dataset_name}")
print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of validation graphs: {len(val_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")
print('====================')


#################################################################
#                       Auxiliary Classes
#################################################################
class BatchData:
    """
    Simple container class for node features and batch information.
    """

    def __init__(self, x, batch):
        self.x = x
        self.batch = batch


class MLPGraphHead(torch.nn.Module):
    """
    MLP prediction head for graph-level tasks.
    Pools node embeddings and passes them through MLP layers.

    Args:
        hidden_channels (int): Dimensionality of node embeddings.
        out_channels (int): Dimensionality of output.
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.pooling_fun = global_mean_pool
        dropout = 0.1
        L = 3  # Number of MLP layers

        layers = []
        for _ in range(L - 1):
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=True))
            layers.append(torch.nn.GELU())

        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_channels, out_channels, bias=True))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, batch):
        x = self.pooling_fun(batch.x, batch.batch)
        return self.mlp(x)


#################################################################
#                    Loss Function Definitions
#################################################################
def multilabel_weighted_bce_loss(output, target, weights=None):
    """
    Weighted Binary Cross Entropy for Multilabel Classification.

    Args:
        output (Tensor): Model logits of shape [batch_size, n_classes].
        target (Tensor): Ground truth labels of shape [batch_size, n_classes].
        weights (Tensor, optional): Per-class weight.
    """
    if weights is None:
        pos_freq = target.float().mean(dim=0)
        weights = 1.0 / (pos_freq + 1e-8)

    bce_loss = F.binary_cross_entropy_with_logits(
        output,
        target,
        pos_weight=weights.to(output.device),
        reduction='none'
    )
    return bce_loss.mean()


#################################################################
#                        Model Definition
#################################################################
class Net(torch.nn.Module):
    """
    General Graph Neural Network with a sequential stack of message-passing layers,
    followed by a graph-level MLP head.
    """

    def __init__(self, activation, hidden_features, num_layer, layer_type, out_channels):
        """
        Args:
            activation (torch.nn.Module): Activation function (e.g., GELU).
            hidden_features (int): Number of hidden (embedding) dimensions.
            num_layer (int): Number of graph-conv layers.
            layer_type (LayerType): Type of GNN layer (GCN, SAGE, GAT, etc.).
            out_channels (int): Number of output classes or final regression dimension.
        """
        super(Net, self).__init__()
        self.layer_type = layer_type
        self.activation = activation
        self.num_layer = num_layer

        # Build GNN backbone and MLP head
        self.model_ = self._build_sequential_container(
            input_features=train_dataset.num_features,
            hidden_features=hidden_features
        )
        self.head = MLPGraphHead(hidden_channels=hidden_features, out_channels=out_channels)

    def forward(self, x, edge_index, batch):
        """
        Args:
            x (Tensor): Node features of shape [num_nodes, in_channels].
            edge_index (LongTensor): Graph connectivity.
            batch (LongTensor): Batch vector, which assigns each node to a specific example.

        Returns:
            Tensor: Graph-level predictions of shape [batch_size, out_channels].
        """
        # Compute node-level embeddings
        h = self.model_(x, edge_index)

        # Global pooling and MLP head
        batch_data = BatchData(x=h, batch=batch)
        return self.head(batch_data)

    def _build_sequential_container(self, input_features, hidden_features):
        """
        Build a container of GNN layers and batch norms in a Sequential style.

        Args:
            input_features (int): Input feature dimension.
            hidden_features (int): Hidden dimension for layers.

        Returns:
            Sequential: Torch Geometric-style sequential container.
        """
        dropout_rate = 0.1
        conv_list = [
            (self._get_conv_layer(input_features, hidden_features), "x, edge_index -> x0"),
            (self.activation, "x0 -> x0_act"),
            (BatchNorm(hidden_features), "x0_act -> x0_bn"),
            (Dropout(p=dropout_rate), "x0_bn -> x0_do"),
        ]
        # Add subsequent layers
        for i in range(1, self.num_layer - 1):
            conv_list.extend([
                (self._get_conv_layer(hidden_features, hidden_features), f"x{i - 1}_do, edge_index -> x{i}"),
                (self.activation, f"x{i} -> x{i}_act"),
                (BatchNorm(hidden_features), f"x{i}_act -> x{i}_bn"),
                (Dropout(p=dropout_rate), f"x{i}_bn -> x{i}_do"),
            ])
        return Sequential("x, edge_index", conv_list)

    def _get_conv_layer(self, input_features, output_features):
        """
        Obtain the GNN layer module based on the layer_type.

        Args:
            input_features (int): Number of input features.
            output_features (int): Number of output features for the layer.

        Returns:
            torch.nn.Module: Chosen GNN layer.
        """
        if self.layer_type == LayerType.LINEAR:
            return TGLinear(input_features, output_features)
        elif self.layer_type == LayerType.GCNCONV:
            return GCNConv(input_features, output_features)
        elif self.layer_type == LayerType.SAGECONV:
            return SAGEConv(input_features, output_features)
        elif self.layer_type == LayerType.GATCONV:
            return GATConv(
                in_channels=input_features,
                out_channels=output_features,
                heads=3,
                edge_dim=train_dataset.num_edge_features,
                concat=False
            )
        elif self.layer_type == LayerType.TRANSFORMERCONV:
            return TransformerConv(input_features, output_features)
        elif self.layer_type == LayerType.CHEBCONV:
            return ChebConv(input_features, output_features, K=3)
        elif self.layer_type == LayerType.CUGRAPHSAGECONV:
            return CuGraphSAGEConv(input_features, output_features)
        elif self.layer_type == LayerType.GRAPHCONV:
            return GraphConv(input_features, output_features)
        elif self.layer_type == LayerType.ARMACONV:
            return ARMAConv(input_features, output_features)
        elif self.layer_type == LayerType.SGCONV:
            return SGConv(input_features, output_features)
        elif self.layer_type == LayerType.MFCONV:
            return MFConv(input_features, output_features)
        elif self.layer_type == LayerType.SSGCONV:
            return SSGConv(input_features, output_features, alpha=0.5)
        elif self.layer_type == LayerType.LECONV:
            return LEConv(input_features, output_features)
        else:
            raise ValueError(f"Unsupported layer_type: {self.layer_type}")


#################################################################
#                    Data Loaders and Model
#################################################################
# For reproducibility
torch.manual_seed(8)

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#################################################################
#                        Activation Setup
#################################################################
# You can switch to RationalOnCluster from the CNA paper by uncommenting the following lines:
"""
activation = RationalOnCluster(
    clusters=20,
    with_clusters=True,
    n=5,
    m=5,
    activation_type=ActivationType.RAT,
    mode=True,
    normalize=True,
    recluster_option=ReclusterOption.ITR,
)
"""
activation = torch.nn.GELU()

#################################################################
#               Model, Optimizer, and Scheduler Setup
#################################################################
model = Net(
    activation=activation,
    hidden_features=35,
    num_layer=20,
    layer_type=LayerType.GCNCONV,
    out_channels=train_dataset.num_classes
).to(device)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss function selection based on dataset name
is_peptides_func = (dataset_name == "Peptides-func")
if is_peptides_func:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.L1Loss()

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=20,
    min_lr=1e-5,
    verbose=True
)


#################################################################
#                  Training and Evaluation Loop
#################################################################
def train_and_evaluate(model, train_loader, val_loader, test_loader,
                       optimizer, criterion, scheduler, num_epochs=20):
    """
    Train and evaluate the model for a specified number of epochs.

    Args:
        model (torch.nn.Module): The GNN model.
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
        test_loader (DataLoader): DataLoader for test set.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (callable): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.

    Returns:
        tuple: (trained_model, all_loss, all_train_metrics, all_val_metrics, all_test_metrics)
    """
    # Initialize metric for multi-label classification if needed
    if is_peptides_func:
        AP = MultilabelAveragePrecision(num_labels=train_loader.dataset.num_classes, average='macro')

    all_loss = []
    all_train_metrics = []
    all_val_metrics = []
    all_test_metrics = []

    for epoch in range(num_epochs):
        # --------------------- Training ---------------------
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ------------------- Evaluation ---------------------
        model.eval()

        def compute_metrics(loader):
            all_preds = []
            all_targets = []
            total_eval_loss = 0.0

            with torch.no_grad():
                for b in loader:
                    b = b.to(device)
                    out = model(b.x, b.edge_index, b.batch)
                    eval_loss = criterion(out, b.y)
                    total_eval_loss += eval_loss.item()

                    if is_peptides_func:
                        preds = torch.sigmoid(out).cpu()
                    else:
                        preds = out.cpu()
                    targets = b.y.cpu()

                    all_preds.append(preds)
                    all_targets.append(targets)

            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Compute metrics
            if is_peptides_func:
                AP.reset()
                AP.update(all_preds, all_targets.long())
                metric_score = AP.compute().item()
            else:
                # Regression metrics: MAE, R2
                mae = mean_absolute_error(all_targets.numpy(), all_preds.numpy())
                r2 = r2_score(all_targets.numpy(), all_preds.numpy(), multioutput='uniform_average')
                metric_score = (mae, r2)

            return metric_score, total_eval_loss / len(loader)

        # Train set metrics
        train_metrics, train_loss = compute_metrics(train_loader)
        # Val set metrics
        val_metrics, val_loss = compute_metrics(val_loader)
        # Test set metrics
        test_metrics, test_loss = compute_metrics(test_loader)

        # Aggregate losses/metrics
        all_loss.append(total_loss / len(train_loader))
        all_train_metrics.append(train_metrics)
        all_val_metrics.append(val_metrics)
        all_test_metrics.append(test_metrics)

        if is_peptides_func:
            scheduler.step(val_metrics)
        else:
            scheduler.step(val_metrics[0])

        # ------------------- Logging ------------------------
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Training Loss: {total_loss:.4f}")
        if is_peptides_func:
            print(f"  Train AP: {train_metrics:.4f} | Val AP: {val_metrics:.4f} | Test AP: {test_metrics:.4f}")
        else:
            print(f"  Train MAE: {train_metrics[0]:.4f}, Train R2: {train_metrics[1]:.4f}")
            print(f"  Val   MAE: {val_metrics[0]:.4f}, Val   R2: {val_metrics[1]:.4f}")
            print(f"  Test  MAE: {test_metrics[0]:.4f}, Test  R2: {test_metrics[1]:.4f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.5e}")
        print("-" * 40)

    return model, all_loss, all_train_metrics, all_val_metrics, all_test_metrics


#################################################################
#                          Run Training
#################################################################
trained_model, all_loss, all_train_metrics, all_val_metrics, all_test_metrics = train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    scheduler,
    num_epochs=300
)
