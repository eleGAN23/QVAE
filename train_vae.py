import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.optim as optim
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F

from models.vanilla_vae import VanillaVAE
#pylint:disable=E1101

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--latent_dim', type=int, default=20, help="Dimension of the latent space.")
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--n_channels', type=int, default=3, help="3 for real-valued inputs, 4 for quaternion-valued inputs.")
parser.add_argument('--kld_weight', type=float, default=0.00001)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--patience_epochs', type=int, default=10)
parser.add_argument('--epochs_no_improve', type=int, default=10)
parser.add_argument('--train_root_dir', type=str, default='../Datasets/img_align_celeba/train')
parser.add_argument('--val_root_dir', type=str, default='../Datasets/img_align_celeba/val')
parser.add_argument('--test_root_dir', type=str, default='../Datasets/img_align_celeba/test')

opt = parser.parse_args()

# Set parameters same as DFC-VAE
opt.batch_size = 64
opt.lr = 0.0005
opt.latent_dim = 100

if opt.cuda:
    torch.cuda.set_device(opt.gpu_num)
if opt.cuda:
    device = "cuda:%i" %opt.gpu_num
else:
    device = "cpu"

# Set seed
seed = 1656079
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def weights_init_normal(m):
    ''' Initialize weights.'''

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01) # according to kingma,2013
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)

def loss_function(recons, input, mu, log_var, kld_weight=opt.kld_weight, quaternion=False) -> dict:
    '''Computes the VAE loss function.
        VAEloss function is composed of BCE reconstruction loss and weighted KL divergence.'''
        
    recons_loss = F.binary_cross_entropy(recons, input)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

def early_stopping(val_loss, train_loss, min_val_loss, patience_epochs, epochs_no_improve):
    early_stop = False
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience_epochs and train_loss < val_loss:
            print("Early stopping...")
            early_stop = True
    return early_stop, min_val_loss, epochs_no_improve


##### DATASET #####
# Prepare CelebA dataset
class CelebaDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, im_name_list, resize_dim, transform=None):
        self.root_dir = root_dir
        self.im_list = im_name_list
        self.resize_dim = resize_dim
        self.transform = transform

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.root_dir, self.im_list[idx])).resize(self.resize_dim, resample=PIL.Image.NEAREST)
        im = np.array(im)
        im = im / 255

        if self.transform:
            im = self.transform(im)

        return im

# Define train set
train_celeba_dataset = CelebaDataset(opt.train_root_dir, os.listdir(opt.train_root_dir), (opt.image_size, opt.image_size),
                                torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_celeba_dataset, batch_size=opt.batch_size, shuffle=True)
# Define validation set
val_celeba_dataset = CelebaDataset(opt.val_root_dir, os.listdir(opt.val_root_dir), (opt.image_size, opt.image_size),
                                torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
val_loader = torch.utils.data.DataLoader(val_celeba_dataset, batch_size=opt.batch_size, shuffle=True)
# Define test set
test_celeba_dataset = CelebaDataset(opt.test_root_dir, os.listdir(opt.test_root_dir), (opt.image_size, opt.image_size),
                                torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_celeba_dataset, batch_size=opt.batch_size, shuffle=True)

##### MODEL SETUP #####
vae = VanillaVAE(in_channels=opt.n_channels, latent_dim=opt.latent_dim)
print("Number of parameters", sum(p.numel() for p in vae.parameters() if p.requires_grad))
if opt.cuda:
    vae.cuda()

optimizer = optim.Adam(vae.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
loss_train = []
loss_val = []

##### TRAINING #####
def train(model):
    model.train()
    start = time.time()
    for epoch in range(opt.n_epochs):
        min_val_loss = 1.0
        epochs_no_improve = 0
        # Training
        temp_loss_train = []
        for idx, data in enumerate(train_loader):
            data = data.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            # Apply model
            recons, input, mu, log_var = model(data)

            # Compute loss function
            loss = loss_function(recons, input, mu, log_var)['loss']
            loss_train.append(loss.detach().item())
            temp_loss_train.append(loss.detach().item())
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            temp_loss_val = []
            for val_data in val_loader:
                val_data = val_data.type(torch.FloatTensor).to(device)
                # Evaluate model
                val_recons, val_input, val_mu, val_log_var = model(val_data)
                # Compute loss function
                loss = loss_function(val_recons, val_input, val_mu, val_log_var)['loss']
                temp_loss_val.append(loss.item())
                loss_val.append(loss.item())
            
        end = time.time()
        print("[Epoch: %i][Train loss: %f][Val loss: %f][Time: %f]" % (epoch, np.mean(temp_loss_train), np.mean(temp_loss_val), end-start))        
        scheduler.step()
        # Store losses
        with open('checkpoints/loss_train_vae_epoch%i' %epoch, 'wb') as fp:
            pickle.dump(loss_train, fp)
        with open('checkpoints/loss_val_vae_epoch%i' %epoch, 'wb') as f:
            pickle.dump(loss_val, f)
        # Store model
        torch.save(model.state_dict(), "checkpoints/model_vae_epoch%i" %epoch)

        # Early stopping
        early_stop, min_val_loss, epochs_no_improve = early_stopping(np.mean(temp_loss_val), np.mean(temp_loss_train), min_val_loss, opt.patience_epochs, epochs_no_improve)
        
        if early_stop:
            print("Early stopped!")
            break

train(vae)