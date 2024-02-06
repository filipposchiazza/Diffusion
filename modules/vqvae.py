import torch
import torch.nn as nn
from modules.building_modules import Encoder, Decoder, VectorQuantizer, VectorQuantizerEMA
import os
import pickle


class VQVAE(nn.Module):
    
    def __init__(self, 
                 input_channels,
                 output_channels,
                 channels,
                 num_resblock,
                 num_emb,  
                 emb_dim,
                 beta=0.25,
                 groups=16,
                 ema_update=True):
        """
        VQ-VAE model

        Parameters
        ----------
        input_channels : int
            Number of input channels. Tipically 3 for RGB images, 1 for grayscale.
        output_channels : int
            Number of output channels. Tipically 3 for RGB images, 1 for grayscale.
        channels : list
            List of number of channels for the encoder and decoder.
        num_resblock : int
            Number of residual blocks.
        num_emb : int   
            Number of embedding vectors.
        emb_dim : int
            Dimension of the embedding vectors.
        beta : float, optional
            Weight for the commitment loss. The default is 0.25.
        groups : int, optional
            Number of groups for the group normalization in the Encoder and Decoder. The default is 16.
        ema_update : bool, optional
            Whether to use the exponential moving average update for the embedding vectors. The default is True.

        Returns
        -------
        None.

        """
        super(VQVAE, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.channels = channels
        self.num_resblock = num_resblock
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.beta = beta
        self.groups = groups
        self.ema_update = ema_update

        self.encoder = Encoder(input_channels=input_channels,
                               channels=channels,
                               latent_dim=emb_dim,
                               num_resblock=num_resblock,
                               groups=groups)
        
        self.decoder = Decoder(output_channels=output_channels,
                               channels=channels[::-1],
                               latent_dim=emb_dim,
                               num_resblock=num_resblock,
                               groups=groups)
        
        if ema_update == True:
            self.vq_layer = VectorQuantizerEMA(num_emb=num_emb, 
                                               emb_dim=emb_dim, 
                                               beta=beta, 
                                               decay=0.99)
        else:
            self.vq_layer = VectorQuantizer(num_emb=num_emb,
                                            emb_dim=emb_dim,
                                            beta=beta)

        self.num_trainable_param, self.num_non_trainable_param = self._calculate_num_parameters()
    

    
    def forward(self, inputs):
        z = self.encoder(inputs)
        loss, q, encoding_indices, perplexity, _ = self.vq_layer(z)
        x_rec = self.decoder(q)
        return loss, x_rec, encoding_indices, perplexity
    
    
    
    def _calculate_num_parameters(self):
        "Evaluate the number of trainable and non-trainable model parameters"
        p_total = 0
        p_train = 0
        for p in self.parameters():
            p_total += p.numel()
            if p.requires_grad:
                p_train += p.numel()       
        p_non_train = p_total - p_train
        return p_train, p_non_train
    

    def generate_from_codes(self, codes):
        "Starting from generated codes, generates synthetic images"
        with torch.no_grad():
            latent_shape_after_permute = (codes.shape[0], codes.shape[2], codes.shape[3], self.emb_dim)
            encoding_indices = codes.view(-1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self.vq_layer.num_emb, device=codes.device)
            encodings.scatter_(1, encoding_indices, 1)
            # quantize and unflatten
            quantized = torch.matmul(encodings, self.vq_layer.emb.weight).view(latent_shape_after_permute)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            syn_img = self.decoder(quantized)
        return syn_img
    


    def save_model(self, save_folder):
        """Save the model and the parameters
        
        Parameters
        ----------
        save_folder : str
            Path to the folder where to save the model and the parameters.
        
        Returns
        -------
        None.
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        param_file = os.path.join(save_folder, 'parameters.pkl')
        parameters = [self.input_channels,
                      self.output_channels,
                      self.channels,
                      self.num_resblock,
                      self.num_emb,
                      self.emb_dim,
                      self.beta,
                      self.groups,
                      self.ema_update]
        with open(param_file, 'wb') as f:
            pickle.dump(parameters, f)
        model_file = os.path.join(save_folder, 'model.pt')
        torch.save(self.state_dict(), model_file)
        
    

    @classmethod
    def load_model(cls, save_folder):
        """Load the model and the parameters
        
        Parameters
        ----------
        save_folder : str
            Path to the folder where the model and the parameters are saved.
            
        Returns
        -------
        model : VQVAE
            VQVAE model.
        """
        param_file = os.path.join(save_folder, 'parameters.pkl') 
        with open(param_file, 'rb') as f:
            parameters = pickle.load(f)    
        model = cls(*parameters)
        model_file = os.path.join(save_folder, 'model.pt')
        model.load_state_dict(torch.load(model_file, map_location='cuda:0'))
        return model
    
    

    @staticmethod
    def save_history(history, save_folder):
        """Save the training history
        
        Parameters
        ----------
        history : dict
            Training history.
        save_folder : str
            Path to the folder where to save the training and validation history.
            
        Returns
        -------
        None."""
        filename = os.path.join(save_folder, 'history.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(history, f)
    


    @staticmethod
    def load_history(save_folder):
        """Load the training history
            
        Parameters
        ----------
        save_folder : str
            Path to the folder where the training history is saved.
        
        Returns
        -------
        history : dict
            Training and validation history.
        """
        history_file = os.path.join(save_folder, 'history.pkl')
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        return history
    


    @staticmethod
    def save_codebook_usage(codebooks, save_folder):
        """Save the number of codebook vectors used for each epoch
        
        Parameters
        ----------
        codebooks : list
            List of codebook vectors.
        save_folder : str
            Path to the folder where to save the number of codebook vectors used for each epoch.
        
        Returns
        -------
        None."""
        codebook_file = os.path.join(save_folder, 'codebook_usage.pkl')
        with open(codebook_file, 'wb') as f:
            pickle.dump(codebooks, f)



    @staticmethod
    def load_codebook_usage(save_folder):
        """Load the number of codebook vectors used for each epoch
        
        Parameters
        ----------
        save_folder : str
            Path to the folder where the number of codebook vectors used for each epoch is saved.
        
        Returns
        -------
        n : list
            List of number of codebook vectors used for each epoch."""
        codebook_file = os.path.join(save_folder, 'codebook_usage.pkl')
        with open(codebook_file, 'rb') as f:
            n = pickle.load(f)
        return n
    




        
