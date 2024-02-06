import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import config
from gaussian_diffusion_utils import GaussianDiffusion
from unet import Unet
import sys
sys.path.append(config.PRETRAINED_MODEL_FOLDER)
from vqvae import VQVAE



def generate_samples(model, gdf_util, num_samples, device):
    """Generate samples from DIffusion model.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to generate samples from.
    gdf_util : GaussianDiffusion
        The Gaussian Diffusion utility.
    num_samples : int
        The number of samples to generate.
    device : torch.device
        The device to use.
    
    Returns
    -------
    torch.Tensor
        The generated samples.
    """
    # 1. Randomly sample noise (starting point for reverse process)
    size = (num_samples, config.IMG_CHANNEL, config.IMG_DIM, config.IMG_DIM)
    samples = torch.randn(size=size).to(device)

    # 2. Sample from the model iteratively
    timesteps = gdf_util.num_timesteps

    with tqdm(reversed(range(timesteps)), unit='step') as bar:
        for t in bar:
            with torch.no_grad():
                tt = torch.full(size=(num_samples, ), fill_value=t, dtype=torch.int32).to(device)
                pred_noise = model((samples, tt))
                samples = gdf_util.p_sample(pred_noise, samples, tt).to(torch.float)

    return samples


def range_conversion(samples, max_pixel_value):
    """Convert samples to desired range.
    
    Parameters
    ----------
    samples : torch.Tensor
        The samples to convert.
    max_pixel_value : float
        The maximum pixel value.
    
    Returns
    -------
    torch.Tensor
        The converted samples.
    """
    samples = (samples + 1) * (max_pixel_value / 2)
    gen_imgs = torch.clamp(samples, min=0.0, max=max_pixel_value).to(torch.int64)
    return gen_imgs


# Load models: Unet and VQVAE
gdf_util = GaussianDiffusion.load(save_folder=config.SAVE_FOLDER)
unet = Unet.load_model(save_folder=config.SAVE_FOLDER).to(config.DEVICE)
unet.eval()
vqvae = VQVAE.load_model(save_folder=config.PRETRAINED_MODEL_FOLDER_PARAM).to(config.DEVICE)
vqvae.eval()

# Generate latent codes
gen_codes = generate_samples(model=unet, 
                             gdf_util=gdf_util, 
                             num_samples=config.NUM_IMG_TO_GENERATE,
                             device=config.DEVICE)
num_emb = vqvae.num_emb
gen_codes = range_conversion(gen_codes, max_pixel_value=num_emb-1)

# Generate images with the vqvae's Decoder
gen_imgs = vqvae.generate_from_codes(codes=gen_codes)


# Show generated images
fig, ax = plt.subplots(4, 5, figsize=(20, 10))
for j in range(2):
    for i in range(5):
        ax[j, i].imshow(gen_imgs[i + j*5].permute(1, 2, 0).to('cpu'))
        ax[j, i].axis('off')

