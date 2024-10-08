import zipfile
import os

# Path to the ZIP file
zip_path = '/workspace/1x2x3_images.zip'

# Directory where files will be extracted
extract_to = '/workspace'

# Create the directory if it doesn't exist
if not os.path.exists(extract_to):
    os.makedirs(extract_to)

# Open the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract all the contents into the directory
    zip_ref.extractall(extract_to)