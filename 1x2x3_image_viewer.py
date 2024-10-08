import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
import numpy as np

BOUNDS = [0.0, 114.75, 127.62749999999998, 255.0]
BOUNDS = [0.0, 113, 127.5, 255.0]
CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

# Open the image file
img = Image.open('PATH/TO/IMAGE.png')

# Convert the image to a NumPy array
image = np.array(img)

# Prepare the figure
fig, axs = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns

# Custom titles based on the original channels
titles = ['Elevation', 'Vegatation',
          'Wind Direction', 'Wind Speed',
          'Previous Fire Mask', 'Fire Mask']

# Loop through each channel and its two parts
for i in range(3):  # Three channels
    cmap_choice = 'viridis' if i < 3 else CMAP
    norm_choice = None if i < 3 else NORM

    axs[0, i].imshow(image[:, :32, i], cmap=cmap_choice, norm=norm_choice)
    axs[0, i].set_title(titles[2*i], fontsize=25)
    axs[0, i].axis('off')  # Turn off axis

    axs[1, i].imshow(image[:, 32:, i], cmap=cmap_choice, norm=norm_choice)
    axs[1, i].set_title(titles[2*i + 1], fontsize=25)
    axs[1, i].axis('off')  # Turn off axis

plt.tight_layout()
plt.show()