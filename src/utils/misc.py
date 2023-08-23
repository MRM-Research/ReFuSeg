import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import nibabel as nib
from PIL import Image

# normalizing target image to be compatible with tanh activation function
def normalize_data(data):
    data *= 2
    data -= 1
    return data

def unnormalize_data(data):
    data += 1
    data /= 2
    return data

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def list_img(dir1):
    direc = []
    for root, dirs, files in os.walk(dir1):
        direc.extend(dirs)
    for x in range(len(direc)):
        direc[x] = dir1 + '/' + direc[x]
    direc = sorted(direc)
    lst = []
    for x in range(len(direc)):
        temp = []
        for root, dirs, files in os.walk(direc[x]):
            temp.extend(files)
        for y in range(len(temp)):
            temp[y] = direc[x] + '/' + temp[y]
        lst.extend(temp)
        
    lst = sorted(lst)
    return lst

def convert_nifti_to_png(nifti_path, output_dir):
    # Load the NIfTI file
    img = nib.load(nifti_path)
    data = np.array(img.dataobj)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over the slices and save them as PNG files
    for i in range(data.shape[2]):
        # Extract the slice
        slice_data = data[:, :, i]
        
        # Normalize the slice data (optional)
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
        
        # Convert the slice to a PIL image
        image = Image.fromarray((slice_data * 255).astype(np.uint8))
        
        # Save the image as PNG
        output_path = os.path.join(output_dir, f"slice_{i}.png")
        image.save(output_path)
        
        print(f"Saved slice {i} as PNG: {output_path}")

# Example usage
#nifti_file = "path/to/input.nii.gz"
#output_directory = "path/to/output/"
#convert_nifti_to_png(nifti_file, output_directory)

# 4 channels grayscale images stacked together

