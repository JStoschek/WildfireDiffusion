
# Import the diffusion network
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchvision import transforms as T, utils
import matplotlib.pyplot as plt
import numpy as np
import imageio

# Define a Unet
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True,
    channels = 3
)
# Create the diffusion model
diffusion = GaussianDiffusion(
    model,
    image_size = (32,64),
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)
# Configure the diffusion model trainer
trainer = Trainer(
    diffusion,
    '1x2x3_images',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.load('21')

x1 = plt.imread('WildFireImages1x2x3/image_00000.png')
x2 = plt.imread('WildFireImages1x2x3/image_00001.png')

img = diffusion.interpolate(x1, x2)

print(img.shape)

img = (img * 255).astype(np.uint8)

imageio.imwrite('interpolate.png', img, format='png')

