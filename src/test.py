import argparse
from src.models.system import System
from src.models.fusion import Fusion
from src.data.datamodule import VoxCeleb2DataModule
from src.losses.complex_mse import PSA_MSE
from pytorch_lightning import Trainer
from torchmetrics.audio import SignalNoiseRatio, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
import torch.nn as nn
import os
import multiprocessing
import warnings
import src.config as config


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


def test(visual_encoder, ckpt_path, test_condition, test_snr, embedding_size, batch_size, num_workers):
    datamodule = VoxCeleb2DataModule(
        data_path=config.DATA_FOLDER_PATH,
        visual_encoder=visual_encoder,
        embedding_size=embedding_size,
        batch_size=batch_size,
        num_workers=num_workers
    )
    datamodule.setup(test_condition=test_condition, test_snr=test_snr)
    test_loader = datamodule.test_dataloader()

    fusion_network = Fusion(embedding_size=embedding_size)
    loss = PSA_MSE()
    metrics = {
        'snr': SignalNoiseRatio(),
        'pesq': PerceptualEvaluationSpeechQuality(16000, 'wb'),
        'sisdr': ScaleInvariantSignalDistortionRatio(),
        'estoi': ShortTimeObjectiveIntelligibility(16000, True)
    }

    model = System(fusion_network, loss, metrics)
    trainer = Trainer()
    trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AV fusion model")

    parser.add_argument("--visual_encoder", type=str, required=True,
                        choices=config.TEST_VISUAL_ENCODERS,
                        help="Visual encoder to use")
    parser.add_argument("--ckpt_path", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--test_condition", type=str, default=config.TEST_CONDITION,
                        choices=config.TEST_ALL_CONDITIONS,
                        help="Test condition")
    parser.add_argument("--test_snr", type=int, default=config.TEST_SNR,
                        choices=config.TEST_ALL_SNRs,
                        help="Test SNR value")
    parser.add_argument("--embedding_size", type=int,
                        help="Embedding size; defaults to encoder-specific value")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader")

    args = parser.parse_args()

    embedding_size = args.embedding_size or config.embedding_size_dict[args.visual_encoder]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

    os.environ["GLOG_minloglevel"] = "3"
    multiprocessing.set_start_method('spawn', True)

    test(
        visual_encoder=args.visual_encoder,
        ckpt_path=args.ckpt_path,
        test_condition=args.test_condition,
        test_snr=args.test_snr,
        embedding_size=embedding_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
