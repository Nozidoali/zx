from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from src.gnn import GNNPolicy
from src.data import SupervisedDataset, collate_fn
from src.utils import log_verbose


def train_supervised(model: GNNPolicy,
                     dataset: SupervisedDataset,
                     config: Dict,
                     verbose: bool = False,
                     tensorboard_dir: str = None) -> Dict[str, List[float]]:
    val_split = config.get('val_split', 0.2)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                             batch_size=config.get('batch_size', 32),
                             shuffle=True,
                             collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset,
                           batch_size=config.get('batch_size', 32),
                           shuffle=False,
                           collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = config.get('patience', 10)
    patience_counter = 0

    for epoch in range(config.get('epochs', 100)):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for states, candidates_list, labels in train_loader:
            optimizer.zero_grad()

            batch_loss = 0.0
            batch_correct = 0

            for i, (state, candidates, label) in enumerate(zip(states, candidates_list, labels)):
                if len(candidates) == 0:
                    continue

                logits = model(state, candidates)
                loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                batch_loss += loss

                pred = torch.argmax(logits)
                if pred == label:
                    batch_correct += 1
                train_total += 1

            if train_total > 0:
                batch_loss = batch_loss / len(states)
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.item()
                train_correct += batch_correct

        train_loss /= len(train_loader)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for states, candidates_list, labels in val_loader:
                batch_loss = 0.0
                batch_correct = 0

                for state, candidates, label in zip(states, candidates_list, labels):
                    if len(candidates) == 0:
                        continue

                    logits = model(state, candidates)
                    loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                    batch_loss += loss

                    pred = torch.argmax(logits)
                    if pred == label:
                        batch_correct += 1
                    val_total += 1

                if val_total > 0:
                    val_loss += batch_loss.item() / len(states)
                    val_correct += batch_correct

        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        val_acc = val_correct / max(val_total, 1)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

        log_verbose(f"Epoch {epoch+1}/{config.get('epochs', 100)}: "
                   f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                   f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}", verbose)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_verbose(f"Early stopping at epoch {epoch+1}", verbose)
                break

    if writer:
        writer.close()

    return history


def evaluate_supervised(model: GNNPolicy, dataset: SupervisedDataset) -> Tuple[float, float]:
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for states, candidates_list, labels in loader:
            for state, candidates, label in zip(states, candidates_list, labels):
                if not candidates:
                    continue

                logits = model(state, candidates)
                loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                total_loss += loss.item()

                pred = torch.argmax(logits)
                if pred == label:
                    correct += 1
                total += 1

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy
