import argparse
import os
import warnings
import multiprocessing
import sys
from src.models.system import System
from src.models.fusion import Fusion
from src.data.datamodule import VoxCeleb2DataModule
from src.losses.complex_mse import PSA_MSE

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.audio import SignalNoiseRatio, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

import config


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


def get_latest_checkpoint_path(checkpoint_dir):
    checkpoints = os.listdir(checkpoint_dir)
    checkpoints = [x for x in checkpoints if 'ckpt' in x]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('=')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoints[-1])


def train(args, train_from_checkpoint=True):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    datamodule = VoxCeleb2DataModule(
        data_path=config.DATA_FOLDER_PATH,
        visual_encoder=args.visual_encoder,
        embedding_size=args.embedding_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    datamodule.setup(test_condition="one_interfering_speaker", test_snr=-5)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    logger = TensorBoardLogger(
        save_dir=args.logger_save_dir,
        name=args.logger_name,
        version=args.version_name
    )

    every_epoch_checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="epoch-{epoch:02d}-{step}",
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True
    )

    best_checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best-{epoch:02d}-{step}",
        monitor="val/loss",
        mode="min",
        save_top_k=5,
        save_last=True
    )

    fusion_network = Fusion(embedding_size=args.embedding_size)
    loss = PSA_MSE()
    metrics = {
        'snr': SignalNoiseRatio(),
        'pesq': PerceptualEvaluationSpeechQuality(16000, 'wb'),
        'sisdr': ScaleInvariantSignalDistortionRatio(),
        'estoi': ShortTimeObjectiveIntelligibility(16000, True)
    }

    model = System(fusion_network, loss, metrics)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        logger=logger,
        callbacks=[every_epoch_checkpoint, best_checkpoint],
        enable_progress_bar=True,
        val_check_interval=0.5
    )

    if train_from_checkpoint:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AV speech model with flexible configuration.")

    parser.add_argument("--visual_encoder", type=str, default=config.VISUAL_ENCODER,
                        choices=config.embedding_size_dict.keys(), help="Visual encoder to use")
    parser.add_argument("--embedding_size", type=int,
                        help="Embedding size (optional; will use encoder default if not provided)")
    parser.add_argument("--checkpoint_dir", type=str, 
                        help="Directory to save checkpoints")
    parser.add_argument("--ckpt_path", type=str, default=config.CKPT_PATH,
                        help="Path to resume training from checkpoint")
    parser.add_argument("--version_name", type=str, 
                        help="Version name for TensorBoard logger")
    parser.add_argument("--logger_save_dir", type=str, default="lightning_logs",
                        help="Root directory for saving TensorBoard logs")
    parser.add_argument("--logger_name", type=str, default="pretrained_encoders",
                        help="Experiment name for logger")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of training epochs")
    parser.add_argument("--train_from_checkpoint", action="store_true",
                    help="If set, training resumes from --ckpt_path. Otherwise, trains from scratch.")


    args = parser.parse_args()
    args.version_name = f"{args.visual_encoder}_5layer"
    args.checkpoint_dir = os.path.join(config.PROJECT_ROOT, f"src/checkpoints/{args.version_name}")
    if args.embedding_size is None:
        args.embedding_size = config.embedding_size_dict[args.visual_encoder]


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

    os.environ["GLOG_minloglevel"] = "3"
    multiprocessing.set_start_method('spawn', True)

    train(args, train_from_checkpoint=args.train_from_checkpoint)

