import torch
import argparse
import json
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import MultilabelAveragePrecision

from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

from models import GCNWithCoarsening, newGCN


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    

def load_dataset(cfg):
    if cfg['dataset']['pos_encoding']:
        if cfg['pos_encoding']['type'] == 'LAPE':
            transformer = AddLaplacianEigenvectorPE(k=cfg['pos_encoding']['num_eigenvectors'])
        else:
            transformer = AddRandomWalkPE(walk_length=cfg['pos_encoding']['walk_length'])
        
        dataset = LRGBDataset(root='data/LRGBDataset', name=cfg['dataset']['name'], pre_transform=transformer)
        train_dataset = LRGBDataset(root='data/LRGBDataset', name=cfg['dataset']['name'], split='train', pre_transform=transformer)
        val_dataset = LRGBDataset(root='data/LRGBDataset', name=cfg['dataset']['name'], split='val', pre_transform=transformer)
        test_dataset = LRGBDataset(root='data/LRGBDataset', name=cfg['dataset']['name'], split='test', pre_transform=transformer)
    else:
        dataset = LRGBDataset(root='data/LRGBDataset', name=cfg['dataset']['name'])
        train_dataset = LRGBDataset(root='data/LRGBDataset', name=cfg['dataset']['name'], split='train')
        val_dataset = LRGBDataset(root='data/LRGBDataset', name=cfg['dataset']['name'], split='val')
        test_dataset = LRGBDataset(root='data/LRGBDataset', name=cfg['dataset']['name'], split='test')

    return dataset, train_dataset, val_dataset, test_dataset


def get_data_loader(device, train_dataset, val_dataset, test_dataset):
    if device == 'cuda':
        train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, pin_memory=True,num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True,num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True,num_workers=2)
    else:
        train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


def get_loss_criterion(dataset_name):
    if dataset_name == 'Peptides-struct':
        criterion = torch.nn.L1Loss()  # For MAE-based regression
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    return criterion


def cosine_with_warmup(epoch, warmup_epochs, total_epochs):
    """
    Defines the warmup and cosine decay schedule.
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return epoch / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    

def train(train_loader, model, device, optimizer, criterion, scheduler):
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


def test(model, loader, device, criterion, dataset_name, AP=None):
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
            if dataset_name == 'Peptides-func':
                pred = torch.sigmoid(out).cpu().numpy()
            else:
                pred = out.cpu().numpy()
            labels = data.y.cpu().numpy()  # Squeeze to remove single-dimensional entries
            all_preds.append(pred)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    if dataset_name == 'Peptides-func':
        AP.reset()
        AP.update(torch.tensor(all_preds), torch.tensor(all_labels).long())
        ap_score = AP.compute().item()
        return ap_score
    else:
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds, multioutput='uniform_average')  # Average across all tasks
        return mae, r2, total_loss / len(loader)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run GNN with coarsening")
    parser.add_argument('--cfg', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.cfg)
    warmup_epochs = cfg["training"]["warmup_epochs"]
    total_epochs = cfg["training"]["total_epochs"]

    # Load dataset
    dataset, train_dataset, val_dataset, test_dataset = load_dataset(cfg)
    dataset_name = cfg["dataset"]["name"]
    dataset_sp = dataset_name.split('-')[1]

    # Initialize the model
    if cfg["training"]["coarsening"]:
        model = GCNWithCoarsening(in_channels=dataset.num_features,
                                  out_channels=dataset.num_classes,
                                  cfg=cfg).to(device)
    else:
        model = newGCN(in_channels=dataset.num_features,
                       cfg=cfg).to(device)

    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["model"]["lr"])
    scheduler = LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: cosine_with_warmup(epoch, warmup_epochs, total_epochs)
    )

    # Define the loss criterion
    criterion = get_loss_criterion(dataset_name=dataset_name)
    if dataset_name == 'Peptides-func':
        AP = MultilabelAveragePrecision(num_labels=train_loader.dataset.num_classes, average='macro')
    else:
        AP = None

    # Prepare data loaders
    train_loader, val_loader, test_loader = get_data_loader(device, train_dataset, val_dataset, test_dataset)

    # Training Loop
    seed = cfg["training"]["seed"]
    log_directory = cfg["training"]["log_directory"]
    os.makedirs(log_directory, exist_ok=True)
    logs = []

    if dataset_name =='Peptides-func':
        second_sota_gcn_value = 0.6860  # SOTA GCN baseline value
        first_sota_gcn_value = 0.5930 #SOTA in LRGB paper
    else:
        second_sota_gcn_value = 0.2460  # SOTA GCN baseline value
        first_sota_gcn_value = 0.3496 #SOTA in LRGB paper
    
    for epoch in range(1, total_epochs + 1):
        loss = train(model, train_loader, device, optimizer, criterion, scheduler)
        if dataset_name =='Peptides-func':
            val_ap = test(val_loader, device, criterion, dataset_name, AP)
            test_ap = test(test_loader, device, criterion, dataset_name, AP)
            train_ap= test(train_loader, device, criterion, dataset_name, AP)
            log_entry = {
                'epoch': int(epoch),
                'loss': float(loss),
                'val_metric': float(val_ap),
                'test_metric': float(test_ap),
                'train_metric': float(train_ap)
            }
            logs.append(log_entry)
            print(f"Seed {seed}, Epoch {epoch:03d}, Loss: {loss:.4f}, Val AP: {val_ap:.4f}, Test AP: {test_ap:.4f}")
        else:
            val_mae, val_r2, val_loss = test(val_loader, device, criterion, dataset_name, AP)
            test_mae, test_r2, _ = test(test_loader, device, criterion, dataset_name, AP)
            train_mae, train_r2, _ = test(train_loader, device, criterion, dataset_name, AP)
            log_entry = {
                'epoch': int(epoch),
                'loss': float(loss),
                'val_metric': float(val_mae),
                'val_r2': float(val_r2),
                'val_loss': float(val_loss),
                'test_metric': float(test_mae),
                'test_r2': float(test_r2),
                'train_metric': float(train_mae),
                'train_r2': float(train_r2)
            }
            logs.append(log_entry)
            print(f"Seed {seed}, Epoch {epoch:03d}, Loss: {loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}")

    # save logs
    log_file_path = os.path.join(log_directory, f'seed_{seed}_logs_{dataset_sp}.json')
    with open(log_file_path, 'w') as f:
        json.dump(logs, f, indent=4)

    # plot results
    plot_file_path = os.path.join(log_directory, f'seed_{seed}_plot_{dataset_sp}.png')
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

if __name__ == "__main__":
    main()