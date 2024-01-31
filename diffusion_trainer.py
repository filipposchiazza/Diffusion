import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class DiffusionTrainer():

    def __init__(self, 
                 unet,
                 gdf_util,
                 optimizer,
                 device,
                 verbose=True):
        
        self.unet = unet.to(device)
        self.gdf_util = gdf_util
        self.timesteps = gdf_util.num_timesteps
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose



    def train(self,
              train_dataloader,
              num_epochs,
              val_dataloader=None):
        
        # Initialize history
        history = {'loss_train': [],
                   'loss_val': []}
        
        for epoch in range(num_epochs):
            
            # Training mode
            self.unet.train()

            # Train one epoch
            train_loss = self._train_one_epoch()

            # Update history
            history['loss_train'].append(train_loss)

            # Validation mode
            self.unet.eval()

            # Validate one epoch
            val_loss = self._validate()

            # Update history
            history['loss_val'].append(val_loss)
        
        return history



    def _train_one_epoch(self,
                         train_dataloader,
                         epoch):
        
        running_loss = 0.
        mean_loss = 0.

        with tqdm(train_dataloader, unit="batches") as tepoch:

            for batch_idx, imgs in enumerate(tepoch):

                # Update the progress bar description
                tepoch.set_description(f'Epoch {epoch+1}')

                # Load images on device
                imgs = imgs.to(device=self.device, dtype=torch.float)

                # 1. Get the batch size
                batch_size = imgs.shape[0]
        
                # 2. Sample timesteps uniformely
                t = torch.randint(low=0, high=self.timesteps, size=(batch_size, )).to(self.device)
        
                # 3. Sample random noise to be added to the images in the batch
                noise = torch.randn(size=imgs.shape, dtype=imgs.dtype).to(self.device)

                # 4. Diffuse the images with noise
                imgs_t = self.gdf_util.q_sample(imgs, t, noise).to(self.device, dtype=torch.float)
        
                # 5. Pass the diffused images and time steps to the unet
                pred_noise = self.unet((imgs_t, t))
                
                # 6. Calculate the loss and update the weights
                loss = F.mse_loss(noise, pred_noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
                # 7. Update running losses and mean losses
                running_loss += loss.item()
                mean_loss = running_loss / (batch_idx + 1)
            
                tepoch.set_postfix(loss="{:.6f}".format(mean_loss))
        
        return mean_loss
    


    def _validate(self, val_dataloader):

        running_val_loss = 0.

        with torch.no_grad():
            for batch_idx, imgs in enumerate(val_dataloader):

                # Load images on device
                imgs = imgs.to(device=self.device, dtype=torch.float)

                # 1. Get the batch size
                batch_size = imgs.shape[0]
        
                # 2. Sample timesteps uniformely
                t = torch.randint(low=0, high=self.timesteps, size=(batch_size, )).to(self.device)
        
                # 3. Sample random noise to be added to the images in the batch
                noise = torch.randn(size=imgs.shape, dtype=imgs.dtype).to(self.device)

                # 4. Diffuse the images with noise
                imgs_t = self.gdf_util.q_sample(imgs, t, noise).to(self.device, dtype=torch.float)
        
                # 5. Pass the diffused images and time steps to the unet
                pred_noise = self.unet((imgs_t, t))
                
                # 6. Calculate the loss 
                loss = F.mse_loss(noise, pred_noise)
            
                # 7. Update running losses and mean losses
                running_val_loss += loss.item()

        mean_val_loss = running_val_loss / len(val_dataloader)

        if self.verbose == True:
            print(f"Validation loss: {mean_val_loss:.6f}")

        return mean_val_loss

            
        
    





"""
def train_one_epoch(train_dataloader, 
                    unet, 
                    gdf_utils, 
                    epoch,
                    optimizer,
                    timesteps,
                    device):
    running_loss = 0.
    mean_loss = 0.
    
    unet.train()
    
    with tqdm(train_dataloader, unit="batches") as tepoch:
        
        for i, imgs in enumerate(tepoch):
            
            tepoch.set_description(f"Epoch {epoch+1}")
            
            # Load images on gpu
            imgs = imgs.to(device=device, dtype=torch.float)
            
            # Zero your gradients for every batch!
            optimizer.zero_grad()
        
            # 1. Get the batch size
            batch_size = imgs.shape[0]
        
            # 2. Sample timesteps uniformely
            t = torch.randint(low=0, high=timesteps, size=(batch_size, )).to(device)
        
            # 3. Sample random noise to be added to the images in the batch
            noise = torch.randn(size=imgs.shape, dtype=imgs.dtype).to(device)
            # 4. Diffuse the images with noise
            imgs_t = gdf_utils.q_sample(imgs, t, noise).to(device, dtype=torch.float)
        
            # 5. Pass the diffused images and time steps to the unet
            pred_noise = unet((imgs_t, t))
            # 6. Calculate the loss and its gradients
            loss = F.mse_loss(noise, pred_noise)
            loss.backward()
        
            # 7. Adjust learning weights
            optimizer.step()
            
            # 8. Update running losses and mean losses
            running_loss += loss.item()
            mean_loss = running_loss / (i + 1)
            
            tepoch.set_postfix(loss="{:.6f}".format(mean_loss))
    

    return mean_loss
"""
