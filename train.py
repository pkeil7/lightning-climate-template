"""
Training script for climate model.
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import LazyDataModule
from model import CNNLightningModule, CNNModel


def _load_config(config_path):
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping of keys to values")

    return data


def parse_args():
    """Parse command line arguments.
    Ignore this for jupyter notebook usage
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    config_args, remaining_args = config_parser.parse_known_args()
    try:
        config = _load_config(config_args.config)
    except (FileNotFoundError, ValueError) as exc:
        config_parser.error(str(exc))

    parser = argparse.ArgumentParser(description="Train climate model")
    parser.add_argument(
        "--config",
        type=str,
        default=config_args.config,
        help="Path to YAML config file",
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.get("data_dir"),
        help="Path or pattern to data files",
    )
    
    # Model arguments
    parser.add_argument(
        "--in_channels",
        type=int,
        default=config.get("in_channels", 10),
        help="Number of input channels",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=config.get("out_channels", 1),
        help="Number of output channels",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.get("batch_size", 32),
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config.get("learning_rate", 1e-3),
        help="Learning rate",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=config.get("max_epochs", 100),
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=config.get("num_workers", 4),
        help="Number of dataloader workers",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=config.get("seed", 42),
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.get("output_dir", "outputs"),
        help="Output directory for logs and checkpoints",
    )

    args = parser.parse_args(remaining_args)
    if args.data_dir is None:
        parser.error("--data_dir is required (or set it in the config file)")

    return args


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    datamodule = LazyDataModule(
        train_files=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    
    # Initialize model
    model = CNNModel(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
    
    # Wrap in Lightning module
    lightning_module = CNNLightningModule(
        model=model,
        learning_rate=args.learning_rate,
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="cnn-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
    ) # we could also use wandb: https://wandb.ai/home
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices="auto",
    )
    
    # Train
    trainer.fit(lightning_module, datamodule=datamodule)
    
    # Test
    trainer.test(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
