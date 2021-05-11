import argparse
import os
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

from models.vanilla_vae_q import QuaternionVanillaVAE
from models.vanilla_vae import VanillaVAE

#pylint:disable=E1101

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int, default=4)
parser.add_argument('--QVAE', type=bool, default=False, help="VAE or QVAE model to use for generation.")
parser.add_argument('--n_channels', type=int, default=3, help="3 for real-valued inputs, 4 for quaternion-valued inputs.")
parser.add_argument('--latent_dim', type=int, default=100, help="Dimension of the latent space.")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--num_samples', type=int, default=8, help="Number of samples to generate.")
parser.add_argument('--root_dir', type=str, default='../Datasets/img_align_celeba/test')

opt = parser.parse_args()
if opt.QVAE:
    opt.n_channels = 4

##### DATASET #####
# Perepare the CelebA dataset
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
        
        if opt.QVAE:
            # Manipulation for quaternion net
            npad  = ((1, 0), (0, 0), (0, 0))
            im = np.pad(im, pad_width=npad, mode='constant', constant_values=0)
        return im

celeba_dataset = CelebaDataset(opt.root_dir, os.listdir(opt.root_dir), (opt.image_size, opt.image_size),
                                torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(celeba_dataset, batch_size=len(celeba_dataset), shuffle=False)


def save_image_single(img, filename):
    '''Save image with given filename.'''

    img = img.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    plt.imsave(filename, img)

def generate_new_samples(model, num_samples):
    '''Sample from a Gaussian N(0,1) and then generates new samples.
        Take model and number of samples to generate in input and store generated images.'''

    model.eval()
    image = model.sample(num_samples=num_samples, current_device=opt.gpu_num)
    for idx, i in enumerate(image):
        if opt.QVAE:
            save_image_single(i[1:, :, :], "RESULTS_EVALUATION/gen_test_MidQVAE_linearlayers/%i.png" %idx)
        else:
            save_image_single(i, "RESULTS_EVALUATION/gen_test_VAE_kldi/%i.png" %idx)
    
def compare_reconstructions(model):
    '''Reconstruct and store images given in input from the test set.'''

    model.eval()
    for data in test_loader:
        data = data.type(torch.FloatTensor).to(device)
        image = model.generate(x=data)
        for idx, i in enumerate(image):
            if opt.QVAE:
                save_image_single(i[1:, :, :], "RESULTS_EVALUATION/recons_test_MidQVAE_linearlayers/%i.png" %idx)
            else:
                save_image_single(i, "RESULTS_EVALUATION/recons_test_VAE_kldi/%i.png" %idx)

if opt.QVAE:
    model = QuaternionVanillaVAE(in_channels=opt.n_channels, latent_dim=opt.latent_dim)
else:
    model = VanillaVAE(in_channels=opt.n_channels, latent_dim=opt.latent_dim)
if opt.cuda:
    torch.cuda.set_device(opt.gpu_num)
    device = "cuda:%i" %opt.gpu_num
    model.cuda()
else:
    device = "cpu"
    
# Load model state dictionary
if opt.QVAE:
    model.load_state_dict(torch.load("checkpoints/model_midqvae_newloss_epoch49"))
else:
    model.load_state_dict(torch.load("checkpoints/model_vae_nobn_kldi_epoch49"))

generate_new_samples(model, opt.num_samples)
compare_reconstructions(model)
