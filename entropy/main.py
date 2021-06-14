import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
import torch.optim as optim
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F

from datasets import QProper_signals1, QProper_signals2, QImproper_signals1, QImproper_signals2

# from QVAE.base import *
# from QVAE.base import BaseVAE
# from QVAE.quaternion_layers import (QuaternionConv, QuaternionLinear,
#                                QuaternionTransposeConv)
from ..models.vanilla_vae_q import QuaternionFCVAE2

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num", type=int, default=0)
parser.add_argument("--track_signals", type=bool, default=False)
parser.add_argument(
    "--experiment", type=int, default=3, help="1, 2"
)
parser.add_argument("--len_train", type=int, default=1000)
parser.add_argument("--len_val", type=int, default=500)
parser.add_argument("--n", type=int, default=1)
parser.add_argument(
    "--signal", type=str, default="improper", help="Either proper or improper."
)
parser.add_argument(
    "--n_channels",
    type=int,
    default=4,
    help="3 for real-valued inputs, 4 for quaternion-valued inputs.",
)
parser.add_argument(
    "--latent_dim", type=int, default=100, help="Dimension of the latent space."
)
parser.add_argument("--n_epochs", type=int, default=5000, help="Training epochs.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate.")
parser.add_argument("--kld_weight", type=float, default=1, help="Weight KLD in loss.")
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument(
    "--num_samples", type=int, default=1000, help="Number of samples to generate."
)

opt = parser.parse_args()


print("#######")
print(
    "Running experiment %i on %s signals for %i epochs, with KLD weight %f"
    % (opt.experiment, opt.signal, opt.n_epochs, opt.kld_weight)
)
print("######")


def loss_function(recons, input, mu, log_var, kld_weight=opt.kld_weight):
    """Computes QVAE loss function.
    QVAEloss function is composed of BCE reconstruction loss and weighted KL divergence.
    Quaternion-valued KL divergence is defined according to the paper."""

    # recons_loss = F.binary_cross_entropy(recons, input)
    recons_loss = F.mse_loss(recons, input)
    kld_loss = torch.mean(
        0.5 * (torch.sum(log_var.exp() + mu ** 2 - 1, dim=2))
        - 2 * torch.sum(log_var, dim=2),
        dim=0,
    )
    kld_loss = torch.mean(kld_loss)

    loss = recons_loss + opt.kld_weight * kld_loss
    return loss, recons_loss, -kld_loss


###############################
# Select and setup experiment #
###############################

# Define model
qvae = QuaternionFCVAE2(
    in_channels=opt.n_channels, latent_dim=opt.latent_dim, n=opt.n
)

if opt.experiment == 1:
    # Define input signal
    if opt.signal == "proper":
        train_loader, val_loader, _ = QProper_signals1(
            opt.len_train, opt.len_val, 100, opt.n
        )
    elif opt.signal == "improper":
        train_loader, val_loader, _ = QImproper_signals1(
            opt.len_train, opt.len_val, 100, opt.n
        )
    else:
        print("Wrong kind of signal. Signal can be either 'proper' or 'improper'.")

elif opt.experiment == 2:
    # Define input signal
    if opt.signal == "proper":
        train_loader, val_loader, _ = QProper_signals2(
            opt.len_train, opt.len_val, 100, opt.n
        )
    elif opt.signal == "improper":
        train_loader, val_loader, _ = QImproper_signals2(
            opt.len_train, opt.len_val, 100, opt.n
        )
    else:
        print("Wrong kind of signal. Signal can be either 'proper' or 'improper'.")


print(
    "Number of parameters", sum(p.numel() for p in qvae.parameters() if p.requires_grad)
)
if opt.cuda:
    qvae.cuda()

optimizer = optim.Adam(qvae.parameters(), lr=opt.lr)
loss_train = []
loss_val = []
kld_train = []
kld_val = []

if opt.cuda:
    torch.cuda.set_device(opt.gpu_num)
if opt.cuda:
    device = "cuda:%i" % opt.gpu_num
else:
    device = "cpu"


###########################
# Define and run training #
###########################


def train(model):
    model.train()
    start = time.time()
    for epoch in range(opt.n_epochs):
        # Training
        temp_loss_train = []
        temp_kld_train = []
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            # Apply model
            recons, input, mu, log_var = model(data)
            # Compute loss function
            loss, _, kld = loss_function(recons, input, mu, log_var)
            kld_train.append(kld.detach().item())
            temp_kld_train.append(kld.detach().item())
            loss_train.append(loss.detach().item())
            temp_loss_train.append(loss.detach().item())
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            temp_loss_val = []
            temp_kld_val = []
            for val_data in val_loader:
                val_data = val_data.to(device)
                # Evaluate model
                val_recons, val_input, val_mu, val_log_var = model(val_data)
                # Compute loss function
                loss, _, kld = loss_function(val_recons, val_input, val_mu, val_log_var)
                kld_val.append(kld.detach().item())
                temp_kld_val.append(kld.detach().item())
                temp_loss_val.append(loss.item())
                loss_val.append(loss.item())

                if epoch % 10 == 0:
                    if opt.track_signals:
                        gen_samples = model.sample(
                            num_samples=opt.num_samples, current_device=opt.gpu_num
                        )
                        df = pd.DataFrame(gen_samples.detach().cpu().numpy())
                        df.to_csv(
                            "./signals/toy_example%i/track_signals/%s_samples_%i.csv"
                            % (opt.experiment, opt.signal, epoch),
                            index=False, header=None
                        )

        end = time.time()
        print(
            "[Epoch: %i][Train loss: %f][Train KLD: %f][Val loss: %f][Val KLD: %f][Time: %f]"
            % (
                epoch,
                np.mean(temp_loss_train),
                np.mean(temp_kld_train),
                np.mean(temp_loss_val),
                np.mean(temp_kld_val),
                end - start,
            )
        )
    #         scheduler.step()

    # # Store losses
    # torch.save(loss_train, 'checkpoints/toy_example2/loss_train_qproper')
    # torch.save(loss_val, 'checkpoints/toy_example2/loss_val_qproper')
    # torch.save(kld_train, 'checkpoints/toy_example2/kld_train_qproper')
    # torch.save(kld_val, 'checkpoints/toy_example2/kld_val_qproper')

    # Store model
    torch.save(
        model.state_dict(),
        "./%i/model_%s_epoch%i" % (opt.experiment, opt.signal, epoch),
    )


train(qvae)
