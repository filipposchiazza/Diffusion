import torch.optim as optim
import config
from dataset import prepare_ImageDataset
import unet as unet
import uvit as uvit
from gaussian_diffusion_utils import GaussianDiffusion
from diffusion_trainer import LatentDiffusionTrainer, DiffusionTrainer


# Load image dataset
train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_ImageDataset(img_dir=config.IMG_DIR,
                                                                                    batch_size=config.BATCH_SIZE,
                                                                                    validation_split=config.VALIDATION_SPLIT,
                                                                                    transform=config.TRANSFORM,
                                                                                    seed=123,
                                                                                    fraction=config.FRACTION,
                                                                                    normalization_mode=config.NORMALIZATION_MODE)
# Create Unet and Gdf_util
gdf_util = GaussianDiffusion(schedule='cosine_shifted',
                             timesteps=config.TIMESTEPS,
                             beta_start=config.BETA_START,
                             beta_end=config.BETA_END,
                             clip_min=config.CLIP_MIN,
                             clip_max=config.CLIP_MAX,
                             img_size=config.IMG_DIM)




# Train directly on image space 
if config.TRAIN_LATENT == False:

    model = uvit.UNet(input_channels=config.INPUT_CHANNELS,
                      base_channels=config.BASE_CHANNELS,
                      channel_multiplier=config.CHANNEL_MULTIPLIER,
                      temb_dim=config.TEMB_DIM,
                      num_resblocks=config.NUM_RES_BLOCKS,
                      has_attention=config.HAS_ATTENTION,
                      num_heads=config.NUM_HEADS,
                      dropout_from_resolution=config.DROPOUT_FROM_RESOLUTION,
                      dropout=config.DROPOUT,
                      downsampling_kernel_dim=config.DOWNSAMPLING_KERNEL_DIM)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Create trainer
    trainer = DiffusionTrainer(model=model,
                               gdf_util=gdf_util,
                               optimizer=optimizer,
                               device=config.DEVICE,
                               verbose=True)
    
    # Train
    history = trainer.train(train_dataloader=train_dataloader,
                            num_epochs=config.NUM_EPOCHS,
                            val_dataloader=val_dataloader)
    
    # Save model, Gdf_util and history
    model.save_model(config.SAVE_FOLDER)
    gdf_util.save(config.SAVE_FOLDER)
    model.save_history(history, config.SAVE_FOLDER)




# Train on latent space
else:
    import sys
    sys.path.append(config.PRETRAINED_MODEL_FOLDER)
    from modules.vqvae import VQVAE

    model_loaded = VQVAE.load_model(config.PRETRAINED_MODEL_FOLDER_PARAM)

    model = unet.Unet(input_channels=config.INPUT_CHANNELS,
                      channels=config.CHANNELS,
                      has_attention=config.HAS_ATTENTION,
                      num_residual_blocks=config.NUM_RES_BLOCKS,
                      norm_groups=config.NORM_GROUPS)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Create trainer
    trainer = LatentDiffusionTrainer(unet=model,
                                     pretrained_module=model_loaded,
                                     gdf_util=gdf_util,
                                     optimizer=optimizer,
                                     device=config.DEVICE,
                                     verbose=True)

    # Train
    history = trainer.train(train_dataloader=train_dataloader,
                            num_epochs=config.NUM_EPOCHS,
                            val_dataloader=val_dataloader)
    
    # 
    model.save_model(config.SAVE_FOLDER)
    gdf_util.save(config.SAVE_FOLDER)
    model.save_history(history, config.SAVE_FOLDER)










