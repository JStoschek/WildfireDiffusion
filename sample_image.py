
# # Import the diffusion network
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
# from torchvision import transforms as T, utils

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
import numpy as np
from matplotlib.widgets import Button
import os

# Set the folder path where to save sampled images
folder_name = "Samples"

# # Define a Unet
# model = Unet(
#     dim = 64,
#     dim_mults = (1, 2, 4, 8),
#     flash_attn = True,
#     channels = 3
# )
# # Create the diffusion model
# diffusion = GaussianDiffusion(
#     model,
#     image_size = (32,64),
#     timesteps = 1000,           # number of steps
#     sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
# )
# # Configure the diffusion model trainer
# trainer = Trainer(
#     diffusion,
#     '/workspace/1x2x3_images',
#     train_batch_size = 32,
#     train_lr = 8e-5,
#     train_num_steps = 700000,         # total training steps
#     gradient_accumulate_every = 2,    # gradient accumulation steps
#     ema_decay = 0.995,                # exponential moving average decay
#     amp = True,                       # turn on mixed precision
#     calculate_fid = True              # whether to calculate fid during training
# )

# trainer.load('21')


# sampled_images = diffusion.sample(batch_size = 100)

# for i , img in enumerate(sampled_images):

#     utils.save_image(img, str(folder_name + '/' + f'sample-{i}.png'))
    

image_files = [os.path.join(folder_name, file) for file in os.listdir(folder_name) if file.endswith('.png')]


# Define color mapping and normalization
BOUNDS = [0.0, 113, 127.5, 255.0]
CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

# Titles for each sub-plot based on channels
titles = ['Elevation', 'Vegatation', 'Wind Direction', 'Wind Speed', 'Previous Fire Mask', 'Fire Mask']

# Current image index
current_image_index = 0

# Create figure and axes for image display and buttons
fig, axs = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns for channels
fig.subplots_adjust(bottom=0.2)

# Button for previous image
axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
bprev = Button(axprev, 'Previous')

# Button for next image
axnext = plt.axes([0.21, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')

def update_image(index):
    global current_image_index
    current_image_index = index
    img = Image.open(image_files[current_image_index])
    image = np.array(img)

    for i in range(3):
        cmap_choice = 'viridis' if i < 3 else CMAP
        norm_choice = None if i < 3 else NORM

        axs[0, i].imshow(image[:, :32, i], cmap=cmap_choice, norm=norm_choice)
        axs[0, i].set_title(titles[2*i], fontsize=25)
        axs[0, i].axis('off')

        axs[1, i].imshow(image[:, 32:, i], cmap=cmap_choice, norm=norm_choice)
        axs[1, i].set_title(titles[2*i + 1], fontsize=25)
        axs[1, i].axis('off')
    
    plt.draw()

def prev_image(event):
    if current_image_index > 0:
        update_image(current_image_index - 1)

def next_image(event):
    if current_image_index < len(image_files) - 1:
        update_image(current_image_index + 1)

bprev.on_clicked(prev_image)
bnext.on_clicked(next_image)

# Initialize the first image
update_image(0)

plt.show()
