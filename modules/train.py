import torch.optim as optim
import config
from dataset import prepare_ImageDataset
from unet import Unet
from gaussian_diffusion_utils import GaussianDiffusion
from diffusion_trainer import LatentDiffusionTrainer
import sys
sys.path.append(config.PRETRAINED_MODEL_FOLDER)
from modules.vqvae import VQVAE


model_loaded = VQVAE.load_model(config.PRETRAINED_MODEL_FOLDER_PARAM)

# Load codebook dataset
train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_ImageDataset(img_dir=config.IMG_DIR,
                                                                                    batch_size=config.BATCH_SIZE,
                                                                                    validation_split=config.VALIDATION_SPLIT,
                                                                                    transform=config.TRANSFORM,
                                                                                    seed=123,
                                                                                    fraction=config.FRACTION)
# Create Unet and Gdf_util
gdf_util = GaussianDiffusion(beta_start=config.BETA_START,
                             beta_end=config.BETA_END,
                             timesteps=config.TIMESTEPS,
                             clip_max=config.CLIP_MAX,
                             clip_min=config.CLIP_MIN)

unet = Unet(input_channels=config.INPUT_CHANNELS,
            channels=config.CHANNELS,
            has_attention=config.HAS_ATTENTION,
            num_residual_blocks=config.NUM_RES_BLOCKS,
            norm_groups=config.NORM_GROUPS)

# Create optimizer
optimizer = optim.Adam(unet.parameters(), lr=config.LEARNING_RATE)


# Create trainer
trainer = LatentDiffusionTrainer(unet=unet,
                                 pretrained_module=model_loaded,
                                 gdf_util=gdf_util,
                                 optimizer=optimizer,
                                 device=config.DEVICE,
                                 verbose=True)

# Train
history = trainer.train(train_dataloader=train_dataloader,
                        num_epochs=config.NUM_EPOCHS,
                        val_dataloader=val_dataloader)

# Save Unet, Gdf_util and history
unet.save_model(config.SAVE_FOLDER)
gdf_util.save(config.SAVE_FOLDER)
Unet.save_history(history, config.SAVE_FOLDER)







