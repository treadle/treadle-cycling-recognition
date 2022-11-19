import argparse
from email.policy import default

import numpy as np
import os
from callbacks.recontruction_vis import ReconstructionVis

import torch
from torchvision import transforms
from pytorch_lightning import Trainer, seed_everything

from datasets.lightning_datamodule import SensorDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from models.cae import Autoencoder
from utils.augmentations import init_transforms
from utils.utils import load_yaml_to_dict

from torch.utils.mobile_optimizer import optimize_for_mobile


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    # paths
    parser.add_argument('--data_path', required=True, help='path to sampled data')
    parser.add_argument('--experiment_config', required=True, help='path to experiment config')
    parser.add_argument('--model_weights_path', required=True, help='path for model parameters to be saved')

    # model
    parser.add_argument('--model', default='cae')
    parser.add_argument('--export_onnx', action='store_true', default=False)
    parser.add_argument('--as_ptl', action='store_true', default=False)

    # misc
    parser.add_argument('--num_workers', default=1, type=int)

    return parser.parse_args()


def train_test_autoencoder(args, config):
    batch_size = config['experiment']['batch_size']
    num_epochs = config['experiment']['num_epochs']

    cae = Autoencoder(config['model'][args.model]['kwargs']['in_channels'], config['model'][args.model]['kwargs']['out_channels'], config['model'][args.model]['kwargs']['latent_size'])


    train_transforms = None
    test_transforms = None

    mean = np.load(os.path.join(args.data_path, 'ovr_mean.npy'))
    std = np.load(os.path.join(args.data_path, 'ovr_std.npy'))
    train_transforms, test_transforms = init_transforms(mean, std)
    

    datamodule = SensorDataModule(
        os.path.join(args.data_path, 'train'),
        os.path.join(args.data_path, 'val'),
        os.path.join(args.data_path, 'test'),
        batch_size,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        num_workers=args.num_workers
    )

    recvis_callback = ReconstructionVis()
    checkpoint_callback = ModelCheckpoint(dirpath=args.model_weights_path, save_top_k=1, monitor="train_loss", mode='min')

    trainer = Trainer.from_argparse_args(args=args, gpus=1, deterministic=True, max_epochs=num_epochs, default_root_dir='logs', callbacks=[recvis_callback, checkpoint_callback])

    trainer.fit(cae, datamodule)
    trainer.test(cae, datamodule)

    if args.as_ptl:  
        scripted_model = torch.jit.script(cae)
        optimized_model = optimize_for_mobile(scripted_model)
        optimized_model._save_for_lite_interpreter("model_weights.ptl")
    if args.export_onnx:
        cae.eval()
        data_placeholder = torch.randn(1, config['model'][args.model]['kwargs']['in_channels'], config['model'][args.model]['kwargs']['max_len'], requires_grad=True)
        output = cae(data_placeholder)
        print(output.shape)
        torch.onnx.export(
            cae,
            data_placeholder,
            os.path.join(args.model_weights_path, f'{args.model}_model_19ch.onnx'),
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input' : {0 : 'batch_size'},  'output' : {0 : 'batch_size'}}
        )


def main():
    seed_everything(20734869)
    args = parse_arguments()
    config = load_yaml_to_dict(args.experiment_config)
    train_test_autoencoder(args, config)


if __name__ == '__main__':
    main()
