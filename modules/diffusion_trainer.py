import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class DiffusionTrainer():

    def __init__(self, 
                 model,   # unet for low-res images, unet for high-res images, uvit for high-res images
                 gdf_util,
                 optimizer,
                 device,
                 verbose=True):
        
        self.model = model.to(device)
        self.gdf_util = gdf_util
        self.timesteps = gdf_util.timesteps
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
            self.model.train()

            # Train one epoch
            train_loss = self._train_one_epoch(train_dataloader=train_dataloader,
                                               epoch=epoch)

            # Update history
            history['loss_train'].append(train_loss)

            if val_dataloader is not None:
                # Validation mode
                self.model.eval()

                # Validate one epoch
                val_loss = self._validate(val_dataloader=val_dataloader)

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
                pred_noise = self.model((imgs_t, t))
                
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
                pred_noise = self.model((imgs_t, t))
                
                # 6. Calculate the loss 
                loss = F.mse_loss(noise, pred_noise)
            
                # 7. Update running losses and mean losses
                running_val_loss += loss.item()

        mean_val_loss = running_val_loss / len(val_dataloader)

        if self.verbose == True:
            print(f"Validation loss: {mean_val_loss:.6f}")

        return mean_val_loss

            
        
    





class LatentDiffusionTrainer(DiffusionTrainer):

    def __init__(self, 
                 unet, 
                 pretrained_module,
                 gdf_util, 
                 optimizer, 
                 device, 
                 verbose=True):
        super().__init__(unet, gdf_util, optimizer, device, verbose)
        self.pretrained_module = pretrained_module.to(device)
        self.pretrained_module.eval()



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

                # encode imgs and get codes
                with torch.no_grad():
                    e = self.pretrained_module.encoder(imgs)
                    _, _, _, _, codes = self.pretrained_module.vq_layer(e)
                    codes.unsqueeze_(1)  # add channel dimension
                    codes = codes * 2 / (self.pretrained_module.num_emb - 1) - 1.0    # from [0, num_emb - 1] to [-1, 1]

                # 1. Get the batch size
                batch_size = codes.shape[0]
        
                # 2. Sample timesteps uniformely
                t = torch.randint(low=0, high=self.timesteps, size=(batch_size, )).to(self.device)
        
                # 3. Sample random noise to be added to the images in the batch
                noise = torch.randn(size=codes.shape, dtype=imgs.dtype).to(self.device)

                # 4. Diffuse the images with noise
                imgs_t = self.gdf_util.q_sample(codes, t, noise).to(self.device, dtype=torch.float)
        
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

                # encode imgs and get codes
                e = self.pretrained_module.encoder(imgs)
                _, _, _, _, codes = self.pretrained_module.vq_layer(e)
                codes.unsqueeze_(1)  # add channel dimension
                codes = codes * 2 / (self.pretrained_module.num_emb - 1) - 1.0    # from [0, num_emb - 1] to [-1, 1]

                # 1. Get the batch size
                batch_size = codes.shape[0]
        
                # 2. Sample timesteps uniformely
                t = torch.randint(low=0, high=self.timesteps, size=(batch_size, )).to(self.device)
        
                # 3. Sample random noise to be added to the images in the batch
                noise = torch.randn(size=codes.shape, dtype=imgs.dtype).to(self.device)

                # 4. Diffuse the images with noise
                imgs_t = self.gdf_util.q_sample(codes, t, noise).to(self.device, dtype=torch.float)
        
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


