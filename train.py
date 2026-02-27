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
    parser = argparse.ArgumentParser(description="Train climate model")
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file",
    )

    args = parser.parse_args()
    try:
        config = _load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    defaults = {
        "data_dir": None,
        "in_channels": 10,
        "out_channels": 1,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "max_epochs": 100,
        "num_workers": 4,
        "seed": 42,
        "output_dir": "outputs",
    }
    merged = {**defaults, **config, "config": args.config}

    if merged["data_dir"] is None:
        parser.error("data_dir must be set in the config file")

    return argparse.Namespace(**merged)


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
