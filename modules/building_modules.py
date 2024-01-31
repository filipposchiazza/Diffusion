import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle


class AttentionBlock(nn.Module):
    
    def __init__(self, units, groups):
        """Self-Attention layer.

        Parameters:
        ----------
        units: int
            Number of units in the dense layers
        groups: int
            Number of groups to be used for GroupNormalization layers
        """
        super(AttentionBlock, self).__init__()
        self.units = units
        self.groups = groups
        
        self.gropu_norm = nn.GroupNorm(num_groups=groups, 
                                       num_channels=units)
        self.query = nn.Linear(units, units)
        self.key = nn.Linear(units, units)
        self.value = nn.Linear(units, units)
        self.proj = nn.Linear(units, units)
        
        
    def forward(self, inputs):
        batch_size, channels, height, width = inputs.shape
        scale = self.units ** (-0.5)
        
        inputs = self.gropu_norm(inputs)
        flat_inputs = inputs.permute(0, 2, 3, 1).reshape(-1, channels)
        q = self.query(flat_inputs).reshape(batch_size, height, width, -1)
        k = self.key(flat_inputs).reshape(batch_size, height, width, -1)
        v = self.value(flat_inputs).reshape(batch_size, height, width, -1)
        
        attn_score = torch.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = attn_score.reshape(batch_size, height, width, height * width)

        attn_score = F.softmax(attn_score, -1)
        attn_score = attn_score.reshape(batch_size, height, width, height, width)

        proj = torch.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = proj.reshape(-1, channels)
        proj = self.proj(proj)
        proj = proj.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return inputs + proj
    
    

class TimeEmbedding(nn.Module):
    
    def __init__(self, dim):
        """Time embedding layer.
        
        Parameters:
        ----------
        dim: int
            Dimension of the time embedding
        """

        super(TimeEmbedding, self).__init__()
        self.dim = dim 
        self.half_dim = dim // 2
        emb = math.log(10000) / (self.half_dim - 1)
        emb = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) * - emb)
        self.register_buffer('emb', emb)
    
    def forward(self, inputs):
        inputs = inputs.to(torch.float32)
        inputs_emb = inputs.unsqueeze(1) * self.emb.unsqueeze(0)
        emb = torch.cat([torch.sin(inputs_emb), torch.cos(inputs_emb)], axis=-1)
        return emb
    


class ResidualBlock(nn.Module):
    
    def __init__(self, 
                 input_channels, 
                 t_dim, 
                 num_channels, 
                 groups=8, 
                 activation_fn=F.silu):
        """Residual block with GroupNormalization and time embedding.

        Parameters:
        ----------
        input_channels: int
            Number of input channels
        t_dim: int
            Dimension of the time embedding
        num_channels: int
            Number of output channels
        groups: int
            Number of groups to be used for GroupNormalization layers
        activation_fn: function
            Activation function to be used
        """
        super(ResidualBlock, self).__init__()
        self.activation_fn = activation_fn
        self.groups = groups
        self.t_dim = t_dim
        if input_channels == num_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels=input_channels, 
                                      out_channels=num_channels, 
                                      kernel_size=1)
            
        self.linear = nn.Linear(t_dim, num_channels)
        self.group_norm1 = nn.GroupNorm(groups, input_channels)
        self.group_norm2 = nn.GroupNorm(groups, num_channels)
        self.conv1 = nn.Conv2d(in_channels=input_channels, 
                               out_channels=num_channels, 
                               kernel_size=3,
                               padding='same',
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=num_channels, 
                               out_channels=num_channels, 
                               kernel_size=3,
                               padding='same',
                               bias=False)
    
    def forward(self, inputs):
        x, t = inputs
        residual = self.residual(x)
        temb = self.activation_fn(t)
        temb = self.linear(temb)
        temb.unsqueeze_(2).unsqueeze_(2)
        x = self.group_norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = torch.add(x, temb)
        x = self.group_norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x = torch.add(x, residual)
        return x
        


class DownSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        """Downsampling layer.
        
        Parameters:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        """
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=3,
                              stride=2,
                              padding=1)
        
    def forward(self, inputs):
        return self.conv(inputs)



class UpSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        """Upsampling layer.
        
        Parameters:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels"""
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1)
        
    def forward(self, x):
        return self.deconv(x)



class TimeMLP(nn.Module):
    
    def __init__(self,input_dim, units, activation_fn=F.silu):
        """Time MLP layer.
        
        Parameters:
        ----------
        input_dim: int
            Dimension of the input
        units: int
            Number of units in the dense layers
        activation_fn: function
            Activation function to be used
        """
        super(TimeMLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, units)
        self.activation_fn = activation_fn
        self.linear2 = nn.Linear(units, units)
        
    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x
