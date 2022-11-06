import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns

import pytorch_lightning as pl
from torch import nn
import torch



class ReconstructionVis(pl.Callback):
    """
    A callback which logs one or more classifier-specific metrics at the end of each
    validation and test epoch, to all available loggers.
    The available metrics are: accuracy, precision, recall, F1-score.
    """

    def __init__(self):
        self.epoch = 0
        self._reset_state()

    def _reset_state(self):
        self.inputs = []
        self.noised = []
        self.reconstructed = []
        self.gt = []
        self.pairwise_loss = []

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.epoch += 1
        self._reset_state()

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._reset_state()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx) -> None:
        self.inputs = outputs['x']
        self.noised = outputs['noised']
        self.reconstructed = outputs['reconstructed']
        self.gt = outputs['y']
    
    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx, dataloader_idx) -> None:
        self.inputs.extend(outputs['x'].cpu().detach().numpy())
        self.reconstructed.extend(outputs['reconstructed'].cpu().detach().numpy())
        self.gt.extend(outputs['y'].cpu().detach().numpy())
        self.pairwise_loss.extend(outputs['pairwise_loss'].cpu().detach().numpy())

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        idx = random.randint(0, len(self.inputs) - 1)
        inp = self.inputs[idx].cpu().detach().numpy()
        noised = self.noised[idx].cpu().detach().numpy()
        rec = self.reconstructed[idx].cpu().detach().numpy()
        visualize_reconstruction(inp, rec, f'epoch_{self.epoch}', noised, 'results/reconstruction/train')

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        gt = np.array(self.gt)
        pairwise_loss = np.array(self.pairwise_loss)
        visualize_loss_distribution(gt, pairwise_loss, 'loss_dist', save_path='results/reconstruction/test')
        
        idxs = np.random.randint(0, len(self.inputs) - 1, 10)
        for i, idx in enumerate(idxs): 
            inp = self.inputs[idx]
            rec = self.reconstructed[idx]
            gt = self.gt[idx]
            visualize_reconstruction(inp, rec, f'test_label_{gt}_{i}', save_path='results/reconstruction/test')


def visualize_loss_distribution(gt, pairwise_loss, name, save_path='results/reconstruction/test'):
    os.makedirs(save_path, exist_ok=True)
    idx_cycling = np.where(gt == 0)
    idx_scooter = np.where(gt == 1)
    idx_walking = np.where(gt == 2)
    loss_cycling = pairwise_loss[idx_cycling]
    loss_scooter = pairwise_loss[idx_scooter]
    loss_walking = pairwise_loss[idx_walking]
    df_list = [(loss, 'cycling') for loss in loss_cycling]
    df_list.extend([(loss, 'scooter') for loss in loss_scooter])
    df_list.extend([(loss, 'walking') for loss in loss_walking])
    df = pd.DataFrame(df_list, columns=['loss', 'label'])
    sns.displot(data=df, x='loss', hue='label', kind='kde')
    plt.show()
    plt.savefig(f'{save_path}/{name}.png')


def visualize_reconstruction(inp, rec, name, nsd=None, save_path='results/reconstruction/train'):
    os.makedirs(save_path, exist_ok=True)
    num_channels = inp.shape[0]
    fig, axes = plt.subplots(num_channels, 1, figsize=(6, 25))
    x = np.arange(0, inp.shape[1])
    for i in range(num_channels):
        ax = axes[i]
        if nsd is None:
            df = pd.DataFrame(data=[x, inp[i],rec[i]]).T
            df.columns = ['x', 'inp', 'rec']
        else:
            df = pd.DataFrame(data=[x, inp[i],rec[i], nsd[i]]).T
            df.columns = ['x', 'inp', 'rec', 'nsd']
        sns.lineplot(data=df, x='x', y='inp', ax=ax, label='input')
        sns.lineplot(data=df,  x='x', y='rec', ax=ax, label='reconstructed')
        if nsd is not None:
            sns.lineplot(data=df,  x='x', y='nsd', ax=ax, label='noised')
    fig.tight_layout()
    plt.show()
    plt.savefig(f'{save_path}/{name}.png')
        