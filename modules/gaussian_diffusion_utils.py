import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GaussianDiffusion:
    "Gaussian diffusion utility class"

    def __init__(self,
                 schedule,
                 timesteps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 clip_min=-1.0,
                 clip_max=1.0,
                 s=0.008,
                 img_size=256):
        """Initialize the Gaussian diffusion utility class, according to the specified schedule.

        Parameters:
        ----------
        schedule: str
            Schedule for the variance of the noise. It can be 'linear', 'cosine', or 'cosine_shifted'.
        beta_start: float
            Starting value of the scheduled variance of the noise
        beta_end: float
            Ending value of the scheduled variance of the noise
        timesteps: int
            Number of timesteps for the diffusion process
        clip_min: float
            Minimum value of the data
        clip_max: float
            Maximum value of the data
        """
        self.schedule = schedule
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.s = s
        self.img_size = img_size

        if schedule == 'linear':
            self.set_linear_schedule()
        elif schedule == 'cosine':
            self.set_cosine_schedule()
        elif schedule == 'cosine_shifted':
            self.set_cosine_shifted_schedule()

        # Calculation for diffusion q(x_t | x_{t-1})
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alpha_cumprod)
        self.sqrt_recip_m1_alphas_cumprod = torch.sqrt(1.0 / self.alpha_cumprod - 1)

        # Calculation for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.maximum(self.posterior_variance, torch.Tensor([1e-20])))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        self.posterior_mean_coef2 = torch.sqrt(self.alphas) * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)  
        
        

    def set_linear_schedule(self):
        "Set the linear schedule for the variance of the noise."
        
        self.betas = torch.linspace(start=self.beta_start,
                                    end=self.beta_end,
                                    steps=self.timesteps,
                                    dtype=torch.float64)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat((torch.Tensor([1.0]), self.alpha_cumprod[:-1]), dim=0)
        self.one_minus_alpha_cumprod = 1 - self.alpha_cumprod
          


    def set_cosine_schedule(self):
        "Set the cosine schedule for the variance of the noise."
        t = torch.linspace(start=0.0, end=1.0, steps=self.timesteps, dtype=torch.float64)
        arg_num = torch.Tensor(torch.pi / 2 * (t+self.s)/(1+self.s))
        arg_den = torch.Tensor([torch.pi / 2 * (self.s)/(1+self.s)])
        self.alpha_cumprod = torch.cos(arg_num) ** 2 / torch.cos(arg_den) ** 2
        self.alpha_cumprod_prev = torch.cat((torch.Tensor([1.0]), self.alpha_cumprod[:-1]), dim=0)
        self.one_minus_alpha_cumprod = 1 - self.alpha_cumprod
        self.alphas = self.alpha_cumprod / self.alpha_cumprod_prev
        self.betas = 1 - self.alphas



    def set_cosine_shifted_schedule(self):
        "Set the shifted cosine schedule for the variance of the noise, according to the image dimension."
        t = np.linspace(0, 1, self.timesteps)
        arg = (t+self.s) / (1 + self.s) * np.pi / 2
        logSNR = -2 * np.log(np.tan(arg)) + 2 * np.log(64 / self.img_size)
        self.alpha_cumprod = torch.Tensor(sigmoid(logSNR))
        self.alpha_cumprod_prev = torch.cat((torch.Tensor([1.0]), self.alpha_cumprod[:-1]), dim=0)
        self.alphas = self.alpha_cumprod / self.alpha_cumprod_prev
        self.betas = 1 - self.alphas
    """
    def set_cosine_shifted_schedule(self):
        "Set the shifted cosine schedule for the variance of the noise, according to the image dimension."
        t = torch.linspace(start=0.0, end=1.0, steps=self.timesteps)
        arg_num = torch.pi / 2 * (t+self.s)/(1+self.s)
        arg_den = torch.Tensor([self.s / (1 + self.s) * torch.pi / 2])
        f_t = torch.cos(arg_num)
        f_0 = torch.cos(arg_den)
        dim_shift_factor = torch.Tensor([64 / self.img_size])
        log_SNR_shifted = 2 * torch.log(f_t) - torch.log(f_0 ** 2 - f_t ** 2) + 2 * torch.log(dim_shift_factor)
        self.alpha_cumprod = sigmoid(log_SNR_shifted)
        self.alpha_cumprod_prev = torch.cat((torch.Tensor([1.0]), self.alpha_cumprod[:-1]), dim=0)
        self.alphas = self.alpha_cumprod / self.alpha_cumprod_prev
        self.betas = 1 - self.alphas
    """



    def _extract(self, a, t, x_shape):
        """Extract generic coefficients at specified timestep.

        Parameters:
        ----------
        a: Tensor
            Coefficients to extract from
        t: Tensor
            Timestep for which the coefficients are to be extracted
        x_shape: tuple
            Shape of the current batched samples

        Returns:
        -------
        out: Tensor
            Extracted coefficients
        """
        batch_size = x_shape[0]
        out = a[t.type(torch.int64)]
        return out.view(batch_size, 1, 1, 1)   # reshape to [batch_size, 1, 1, 1] for broadcasting purposes
    
    

    # Extract mean and variance from N(x_t; c1*x_0, c2*eps)
    def q_mean_variance(self, x_start, t):
        """Extract the mean and the variance at the current timestep.

        Parameters:
        ----------
        x_start: Tensor
            Initial sample (before the first diffusion step)
        t: Tensor
            Current timestep
        
        Returns:
        -------
        mean: Tensor
            Mean of the current timestep
        """
        x_start_shape = torch.Tensor(x_start).shape
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(self.sqrt_one_minus_alphas_cumprod ** 2, t, x_start_shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start_shape)
        return mean, variance, log_variance
    
    
    
    # Evaluate x_t = c1 * x_0 + c2 * eps
    def q_sample(self, x_start, t, noise):
        """Diffuse the data.

        Parameters:
        ----------
        x_start: Tensor
            Initial sample (before the first diffusion step)
        t: Tensor
            Current timestep
        noise: Tensor
            Gaussian noise to be added at the current timestep
        
        Returns:
        -------
        x_t: Tensor
            Diffused samples at timestep 't'
        """
        x_start_shape = torch.Tensor(x_start).shape
        c1 = self._extract(self.sqrt_alpha_cumprod, t, x_start_shape).to(x_start.device)
        c2 = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x_start_shape).to(x_start.device)
        x_t = c1 * x_start + c2 * noise
        return x_t 
    

    
    # Evaluate x_0 = c3 * x_t - c4 * eps
    def predict_start_from_noise(self, x_t, t, noise):
        """Evaluate x_0 = c3 * x_t - c4 * eps
        
        Parameters:
        ----------
        x_t: Tensor
            Sample at timestep 't'
        t: Tensor
            Current timestep
        noise: Tensor
            Gaussian noise
        
        Returns:
        -------
        x_0: Tensor
            Sample at timestep '0'.
        """
        x_t_shape = torch.Tensor(x_t).shape
        c3 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape).to(x_t.device)
        c4 = self._extract(self.sqrt_recip_m1_alphas_cumprod, t, x_t_shape).to(x_t.device)
        x_0 = c3 * x_t - c4 * noise
        return x_0
    
    

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and the variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        
        Parameters:
        ----------
        x_start: Tensor
            Initial sample (before the first diffusion step)
        x_t: Tensor
            Sample at timestep 't'
        t: Tensor
            Current timestep
        
        Returns:
        -------
        posterior_mean: Tensor
            Mean of the diffusion posterior
        posterior_variance: Tensor
            Variance of the diffusion posterior
        posterior_log_variance_clipped: Tensor
            Log variance of the diffusion posterior (clipped)
        """
        x_t_shape = torch.Tensor(x_t).shape
        c1 = self._extract(self.posterior_mean_coef1, t, x_t_shape).to(x_t.device)
        c2 = self._extract(self.posterior_mean_coef2, t, x_t_shape).to(x_t.device)
        posterior_mean = c1 * x_start + c2 * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t_shape)
            
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    
    
    def p_mean_variance(self, pred_noise, x_t , t, clip_denoised=True):
        """Compute the mean and the variance of the diffusion model p(x_t | x_0).

        Parameters:
        ----------
        pred_noise: Tensor
            Noise predicted by the diffusion model
        x_t: Tensor
            Sample at timestep 't'
        t: Tensor
            Current timestep
        clip_denoised: bool
            Whether to clip the predicted noise within the specified range or not
        
        Returns:
        -------
        model_mean: Tensor
            Mean of the diffusion model
        model_variance: Tensor
            Variance of the diffusion model
        model_log_variance: Tensor
            Log variance of the diffusion model
        """
        x_recon = self.predict_start_from_noise(x_t=x_t, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, self.clip_min, self.clip_max)
            
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon,
                                                                                     x_t = x_t,
                                                                                     t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    
    
    def p_sample(self, pred_noise, x_t, t, clip_denoised=True):
        """Sample from the diffusion model.

        Parameters:
        ----------
        pred_noise: Tensor
            Noise predicted by the diffusion model
        x_t: Tensor
            Sample at timestep 't'
        t: Tensor
            Current timestep
        clip_denoised: bool
            Whether to clip the predicted noise within the specified range or not

        Returns:
        -------
        model_sample: Tensor
            Sample from the diffusion model
        """
        x_t_shape = x_t.shape  
        model_mean, _, model_log_variance = self.p_mean_variance(pred_noise,
                                                                 x_t=x_t,
                                                                 t=t,
                                                                 clip_denoised=clip_denoised)
        model_mean = model_mean.to(x_t.device)
        model_log_variance = model_log_variance.to(x_t.device)
        
        noise = torch.randn(size=x_t_shape, dtype=torch.float32).to(x_t.device)
        # No noise when t==0
        nonzero_mask = 1 - (t == 0).type(torch.float32).to(x_t.device)
        nonzero_mask = nonzero_mask.view(x_t_shape[0], 1, 1, 1)
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    


    def plot_schedule(self):
        t = torch.linspace(start=0.0, end=1.0, steps=self.timesteps, dtype=torch.float64)
        plt.plot(t, self.sqrt_alpha_cumprod, label=r'$\mu_t=\sqrt{\bar{\alpha_t}}$')
        plt.plot(t, self.sqrt_one_minus_alpha_cumprod, label=r'$\sigma_t=\sqrt{1 - \bar{\alpha_t}}$')
        plt.title(f'Schedule: {self.schedule}')
        plt.legend()
        plt.show()



    def save(self, save_folder):
        """Save the Gaussian diffusion utility class.

        Parameters:
        ----------
        save_folder: str
            Folder to save the Gaussian diffusion utility class
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        parameters = [self.schedule,
                      self.timesteps,
                      self.beta_start,
                      self.beta_end,
                      self.clip_min,
                      self.clip_max,
                      self.s,
                      self.img_size]
        filename = os.path.join(save_folder, 'gdf_util.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(parameters, f)



    @classmethod
    def load(cls, save_folder):
        """Load the Gaussian diffusion utility class.

        Parameters:
        ----------
        save_folder: str
            Folder to load the Gaussian diffusion utility class from
        """
        filename = os.path.join(save_folder, 'gdf_util.pkl')
        with open(filename, 'rb') as f:
            parameters = pickle.load(f)
        gdf_util = cls(*parameters)
        return gdf_util
            
        
