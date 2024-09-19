import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import mlflow
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Tuple, Dict, Any
import typer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define your model architecture here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define the forward pass
        pass

def setup_mlflow(experiment_name: str) -> mlflow.ActiveRun:
    """
    Set up MLflow experiment and start a new run.

    Args:
        experiment_name (str): Name of the MLflow experiment.

    Returns:
        mlflow.ActiveRun: Active MLflow run object.
    """
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()

def log_params(params: Dict[str, Any]) -> None:
    """
    Log parameters to MLflow.

    Args:
        params (Dict[str, Any]): Dictionary of parameters to log.
    """
    mlflow.log_params(params)

def log_model_architecture(model: nn.Module) -> None:
    """
    Log model architecture to MLflow.

    Args:
        model (nn.Module): PyTorch model to log.
    """
    mlflow.log_text(str(model), "model_architecture.txt")

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, 
                    epoch: int, best_val_loss: float, checkpoint_path: Path) -> None:
    """
    Save model checkpoint and log it to MLflow.

    Args:
        model (nn.Module): PyTorch model to save.
        optimizer (optim.Optimizer): Optimizer to save.
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler to save.
        epoch (int): Current epoch number.
        best_val_loss (float): Best validation loss so far.
        checkpoint_path (Path): Path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    mlflow.log_artifact(str(checkpoint_path), f"checkpoints/epoch_{epoch}")

def load_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer: optim.Optimizer, 
                    scheduler: optim.lr_scheduler._LRScheduler) -> Tuple[int, float]:
    """
    Load model checkpoint if it exists.

    Args:
        checkpoint_path (Path): Path to the checkpoint file.
        model (nn.Module): PyTorch model to load the state into.
        optimizer (optim.Optimizer): Optimizer to load the state into.
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler to load the state into.

    Returns:
        Tuple[int, float]: Start epoch and best validation loss.
    """
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        logging.info(f"Resuming training from epoch {start_epoch}")
        return start_epoch, best_val_loss
    return 0, float('inf')

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device) -> float:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to train on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """
    Validate the model.

    Args:
        model (nn.Module): PyTorch model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to validate on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, lr: float, 
          device: torch.device, checkpoint_dir: Path, mlflow_experiment_name: str, stage: str) -> None:
    """
    Train the model.

    Args:
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train for.
        lr (float): Learning rate.
        device (torch.device): Device to train on.
        checkpoint_dir (Path): Directory to save checkpoints.
        mlflow_experiment_name (str): Name of the MLflow experiment.
        stage (str): Training stage name.
    """
    run = setup_mlflow(mlflow_experiment_name)
    writer = SummaryWriter(f"runs/{stage}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    log_params({
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "stage": stage,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
    })
    log_model_architecture(model)

    checkpoint_path = checkpoint_dir / f"{stage}_latest.pth"
    start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        mlflow.log_metrics({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = checkpoint_dir / f"{stage}_best.pth"
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(str(best_model_path), "best_model")
            mlflow.pytorch.log_model(model, f"{stage}_best_model")

        scheduler.step()

    final_model_path = checkpoint_dir / f"{stage}_final.pth"
    torch.save(model.state_dict(), final_model_path)
    mlflow.log_artifact(str(final_model_path), "final_model")
    mlflow.pytorch.log_model(model, f"{stage}_final_model")

    writer.close()
    run.end()

def train_pdnorm_layers(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, 
                        lr: float, device: torch.device, checkpoint_dir: Path, mlflow_experiment_name: str) -> None:
    """
    Train only the PDNorm layers of the model.

    Args:
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train for.
        lr (float): Learning rate.
        device (torch.device): Device to train on.
        checkpoint_dir (Path): Directory to save checkpoints.
        mlflow_experiment_name (str): Name of the MLflow experiment.
    """
    for name, param in model.named_parameters():
        param.requires_grad = 'pdnorm' in name.lower()
    train(model, train_loader, val_loader, num_epochs, lr, device, checkpoint_dir, mlflow_experiment_name, "pdnorm")

def apply_lora(model: nn.Module) -> None:
    """
    Apply LoRA to the model.

    Args:
        model (nn.Module): PyTorch model to apply LoRA to.
    """
    # Implement your LoRA application logic here
    pass

def train_lora(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, 
               lr: float, device: torch.device, checkpoint_dir: Path, mlflow_experiment_name: str) -> None:
    """
    Train the LoRA-adapted model.

    Args:
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train for.
        lr (float): Learning rate.
        device (torch.device): Device to train on.
        checkpoint_dir (Path): Directory to save checkpoints.
        mlflow_experiment_name (str): Name of the MLflow experiment.
    """
    apply_lora(model)
    for name, param in model.named_parameters():
        param.requires_grad = 'lora' in name.lower()
    train(model, train_loader, val_loader, num_epochs, lr, device, checkpoint_dir, mlflow_experiment_name, "lora")

def setup_data() -> Tuple[DataLoader, DataLoader]:
    """
    Set up datasets and dataloaders.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    # Set up your datasets and dataloaders here
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    return train_loader, val_loader

app = typer.Typer()

@app.command()
def main(
    pdnorm_epochs: int = typer.Option(50, help="Number of epochs for PDNorm training"),
    pdnorm_lr: float = typer.Option(1e-3, help="Learning rate for PDNorm training"),
    lora_epochs: int = typer.Option(30, help="Number of epochs for LoRA training"),
    lora_lr: float = typer.Option(5e-4, help="Learning rate for LoRA training"),
    checkpoint_dir: Path = typer.Option(Path("checkpoints"), help="Directory to save checkpoints"),
    mlflow_experiment_name: str = typer.Option("YourExperimentName", help="Name of the MLflow experiment"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device to train on (cuda/cpu)")
):
    """
    Run the full training pipeline: PDNorm layer training followed by LoRA training.
    """
    device = torch.device(device)
    model = YourModel().to(device)
    train_loader, val_loader = setup_data()

    # Stage 1: Train PDNorm layers
    train_pdnorm_layers(model, train_loader, val_loader, pdnorm_epochs, pdnorm_lr, device, 
                        checkpoint_dir, mlflow_experiment_name)

    # Stage 2: LoRA training
    train_lora(model, train_loader, val_loader, lora_epochs, lora_lr, device, 
               checkpoint_dir, mlflow_experiment_name)

if __name__ == "__main__":
    app()
