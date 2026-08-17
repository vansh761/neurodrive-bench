from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from neurodrive_bench.config import BenchmarkConfig
from neurodrive_bench.data.storage import load_dataset, split_dataset
from neurodrive_bench.models.neural.checkpoint import save_checkpoint
from neurodrive_bench.models.neural.dataset import TelemetrySequenceDataset
from neurodrive_bench.models.neural.lstm_net import LSTMNetwork
from neurodrive_bench.models.neural.transformer_net import TransformerNetwork
from neurodrive_bench.models.neural.lnn_net import LiquidNetwork


class NeuralTrainer:
    def __init__(self, config: BenchmarkConfig, model_type: str, dataset_path: str) -> None:
        self.config = config
        self.model_type = model_type
        self.dataset_path = dataset_path
        
        # Load training config
        train_cfg = getattr(config, "neural_training_config", {})
        self.epochs = int(train_cfg.get("epochs", 50))
        self.batch_size = int(train_cfg.get("batch_size", 64))
        self.learning_rate = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-5))
        # Weight on the uncertainty/adaptation auxiliary loss relative to the control loss.
        # Kept lower than 1.0 since control accuracy is the primary objective and the
        # auxiliary targets are heuristic proxies, not expert-labeled ground truth.
        self.aux_loss_weight = float(train_cfg.get("aux_loss_weight", 0.3))
        self.output_dir = Path(train_cfg.get("output_dir", "artifacts/neural_models"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train(self) -> Path:
        print(f"Loading dataset from {self.dataset_path}")
        records = load_dataset(self.dataset_path)
        
        # Train/val split
        train_records, val_records, _ = split_dataset(records, train_ratio=0.8, val_ratio=0.2, seed=self.config.benchmark_seed)
        
        # DataLoaders
        seq_len = self.config.telemetry_history_window
        train_dataset = TelemetrySequenceDataset(train_records, seq_len=seq_len)
        val_dataset = TelemetrySequenceDataset(val_records, seq_len=seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Network
        network = self._build_network().to(self.device)
        print(f"Built {self.model_type} network with {self._count_parameters(network):,} parameters")
        
        # Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Tensorboard
        log_dir = self.output_dir / "logs" / f"{self.model_type}_{int(time.time())}"
        writer = SummaryWriter(log_dir=str(log_dir))
        
        best_val_loss = float("inf")
        best_model_path = self.output_dir / f"{self.model_type}_best.pt"
        
        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(1, self.epochs + 1):
            network.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass (handle LNN separately because of delta_ts)
                if self.model_type == "lnn":
                    delta_ts = inputs[:, :, 6:7]
                    outputs, _ = network(inputs, delta_ts)
                else:
                    outputs = network(inputs)
                
                # Channels 0:3 are steering/throttle/brake, supervised against the expert.
                # Channels 3:5 are uncertainty/adaptation, supervised against heuristic
                # proxy targets built in neural/dataset.py (stress-level + tracking error,
                # and trend-based stability, respectively). Previously these two channels
                # received no gradient at all and were effectively untrained noise; see
                # project review notes on why that made GDI partially meaningless.
                control_loss = criterion(outputs[:, :3], targets[:, :3])
                aux_loss = criterion(outputs[:, 3:5], targets[:, 3:5])
                loss = control_loss + self.aux_loss_weight * aux_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
            train_loss /= max(1, len(train_loader.dataset))
            
            # Validation
            network.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    if self.model_type == "lnn":
                        delta_ts = inputs[:, :, 6:7]
                        outputs, _ = network(inputs, delta_ts)
                    else:
                        outputs = network(inputs)
                        
                    control_loss = criterion(outputs[:, :3], targets[:, :3])
                    aux_loss = criterion(outputs[:, 3:5], targets[:, 3:5])
                    loss = control_loss + self.aux_loss_weight * aux_loss
                    val_loss += loss.item() * inputs.size(0)
                    
            val_loss /= max(1, len(val_loader.dataset))
            
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                metadata = {
                    "model_type": self.model_type,
                    "history_window": seq_len,
                    "val_loss": best_val_loss,
                    "epoch": epoch
                }
                save_checkpoint(network, optimizer, epoch, val_loss, best_model_path, metadata)
                
        writer.close()
        print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        print(f"Saved best model to {best_model_path}")
        return best_model_path
        
    def _build_network(self) -> nn.Module:
        if self.model_type == "lstm":
            return LSTMNetwork()
        if self.model_type == "transformer":
            return TransformerNetwork()
        if self.model_type == "lnn":
            return LiquidNetwork()
        raise ValueError(f"Unknown neural model type: {self.model_type}")

    def _count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
