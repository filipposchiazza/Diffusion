import torch
import torch.nn as nn
import torch.nn.functional as F
from building_modules import ResidualBlock, DownSample, UpSample, TimeEmbedding, TimeMLP, AttentionBlock
import pickle
import os


class Unet(nn.Module):
    
    def __init__(self,
                 input_channels, # 1 for grayscale, 3 for RGB 
                 channels, # list of channels for each block
                 has_attention=[False, False, True, True],
                 num_residual_blocks=2,
                 norm_groups=8,
                 activation_fn=F.silu):
        
        """Unet model with GroupNormalization and time embedding.

        Parameters:
        ----------
        input_channels: int
            Number of input channels
        channels: list
            List of channels for each block
        has_attention: list
            List of booleans indicating if the block has an attention mechanism
        num_residual_blocks: int
            Number of residual blocks per block
        norm_groups: int    
            Number of groups to be used for GroupNormalization layers
        activation_fn: function
            Activation function to be used
        """
        
        super(Unet, self).__init__()
        self.img_channels = input_channels
        self.channels = channels
        self.has_attention = has_attention
        self.num_residual_blocks = num_residual_blocks
        self.norm_groups = norm_groups
        self.activation_fn = activation_fn
        
        self.temb_dim = self.channels[0] * 4
        self.time_emb = TimeEmbedding(dim=self.temb_dim)
        self.time_mlp = TimeMLP(input_dim=self.temb_dim, 
                                units=self.temb_dim,
                                activation_fn=activation_fn)
        
        self.conv0 = nn.Conv2d(in_channels=input_channels, 
                               out_channels=self.channels[0], 
                               kernel_size=1)
        
        
        # downblock
        self.downblock = nn.ModuleList()
        for i in range(len(self.channels)):
            for _ in range(num_residual_blocks):
                self.downblock.append(ResidualBlock(input_channels=self.channels[i], 
                                                    t_dim=self.temb_dim, 
                                                    num_channels=self.channels[i],
                                                    groups=norm_groups,
                                                    activation_fn=activation_fn))
                if has_attention[i] == True:
                    self.downblock.append(AttentionBlock(units=self.channels[i], 
                                                         groups=norm_groups))
               
                self.downblock.append(nn.Identity()) # placeholder for the skip
                
            if self.channels[i] != self.channels[-1]:
                self.downblock.append(DownSample(in_channels=self.channels[i], 
                                                 out_channels=self.channels[i+1]))
                self.downblock.append(nn.Identity()) # placeholder for the skip
        
        # middleblock
        self.middle_res1 = ResidualBlock(input_channels=self.channels[-1], 
                                         t_dim=self.temb_dim, 
                                         num_channels=self.channels[-1],
                                         groups=norm_groups,
                                         activation_fn=activation_fn)
        
        self.middle_att = AttentionBlock(units=self.channels[-1], 
                                         groups=norm_groups)
        
        self.middle_res2 = ResidualBlock(input_channels=self.channels[-1], 
                                         t_dim=self.temb_dim, 
                                         num_channels=self.channels[-1],
                                         groups=norm_groups,
                                         activation_fn=activation_fn)
        
        # upblock
        self.upblock = nn.ModuleList()
        for i in reversed(range(len(self.channels))):
            for _ in range(num_residual_blocks + 1):
                self.upblock.append(nn.Identity()) # placeholder for the concat
                self.upblock.append(ResidualBlock(input_channels=self.channels[i]*2, 
                                                  t_dim=self.temb_dim, 
                                                  num_channels=self.channels[i]))
                if has_attention[i] == True:
                    self.upblock.append(AttentionBlock(units=self.channels[i], 
                                                       groups=norm_groups))
                    
            if i != 0:
                self.upblock.append(UpSample(in_channels=self.channels[i], 
                                             out_channels=self.channels[i-1]))
                
        # endblock
        self.gropu_norm = nn.GroupNorm(num_groups=norm_groups, 
                                       num_channels=self.channels[0])
        self.end_conv = nn.Conv2d(in_channels=self.channels[0], 
                                  out_channels=self.img_channels, 
                                  kernel_size=3,
                                  padding='same')
        
        
    
    def forward(self, inputs):
        img, t = inputs
        
        # time embedding
        temb = self.time_emb(t)
        temb = self.time_mlp(temb)
        
        # first image convolution
        x = self.conv0(img)
        skips = [x]
        
        # downblock
        for i in range(len(self.downblock)):
            if type(self.downblock[i]) == nn.Identity:
                skips.append(x)
            elif type(self.downblock[i]) == ResidualBlock:
                x = self.downblock[i]((x, temb))
            else:
                x = self.downblock[i](x)
        
        # middleblock
        x = self.middle_res1((x, temb))
        x = self.middle_att(x)
        x = self.middle_res2((x, temb))
        
        # upblock
        for i in range(len(self.upblock)):
            
            if type(self.upblock[i]) == nn.Identity:
                x = torch.cat((x, skips.pop()), dim=1)
            elif type(self.upblock[i]) == ResidualBlock:
                x = self.upblock[i]((x, temb))
            else:
                x = self.upblock[i](x)
                
        # endblock
        x = self.gropu_norm(x)
        x = self.activation_fn(x)
        x = self.end_conv(x)
        
        return x
    
    
    
    def save_model(self, save_folder):
        """Save the parameters and the model state_dict
        
        Parameters:
        ----------
        save_folder: str
            Folder to save the model
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        param_file = os.path.join(save_folder, 'UnetParameters.pkl')
        parameters = [self.img_channels,
                      self.channels,
                      self.has_attention,
                      self.num_residual_blocks,
                      self.norm_groups,
                      self.activation_fn]
        with open(param_file, 'wb') as f:
            pickle.dump(parameters, f)
    
        model_file = os.path.join(save_folder, 'UnetModel.pt')
        torch.save(self.state_dict(), model_file)
    


    @classmethod
    def load_model(cls, save_folder):
        """Load the parameters and the model state_dict
        
        Parameters:
        ----------
        save_folder: str
            Folder to load the model from
        """
        param_file = os.path.join(save_folder, 'UnetParameters.pkl') 
        with open(param_file, 'rb') as f:
            parameters = pickle.load(f)
        
        model = cls(*parameters)
    
        model_file = os.path.join(save_folder, 'UnetModel.pt')
        model.load_state_dict(torch.load(model_file, map_location='cuda:0'))
    
        return model