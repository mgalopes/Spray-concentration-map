import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def subtract_images(image1_path, image2_path, output_path):
    """
    Subtracts two images and saves the result.

    Parameters:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        output_path (str): Path to save the subtracted image.
    """
    # Load the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Check if the images were loaded successfully
    if image1 is None:
        print(f"Error: Could not load '{image1_path}'. Skipping.")
        return None
    if image2 is None:
        print(f"Error: Could not load '{image2_path}'. Skipping.")
        return None

    # Ensure the images are of the same size
    if image1.shape != image2.shape:
        print(f"Error: Images '{image1_path}' and '{image2_path}' must be of the same size. Skipping.")
        return None

    # Subtract the images
    subtracted_image = cv2.subtract(image1, image2)

    # Save the result
    cv2.imwrite(output_path, subtracted_image)
    print(f"Subtracted image saved as '{output_path}'")
    return output_path

def add_legend_to_colormap(colormap_image, output_path, vmin, vmax):
    """
    Adds a legend to the colormap image and saves it.

    Parameters:
        colormap_image (numpy.ndarray): The colormap image (BGR format).
        output_path (str): Path to save the image with the legend.
        vmin (float): Minimum value for the colorbar.
        vmax (float): Maximum value for the colorbar.
    """
    # Convert BGR to RGB for correct display in matplotlib
    colormap_image_rgb = cv2.cvtColor(colormap_image, cv2.COLOR_BGR2RGB)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(colormap_image_rgb)
    ax.axis("off")  # Hide axis for a clean display

    # Create a ScalarMappable with the adaptive normalization
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])  # Required for ScalarMappable

    # Add color bar (legend)
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Intensity', rotation=270, labelpad=15)

    # Set ticks at 5 points between vmin and vmax
    tick_positions = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f"{val:.2f}" for val in tick_positions])

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Colormap image with legend saved as '{output_path}'")

def reduce_noise(image, gaussian_sigma=1.0, median_kernel_size=3, morph_kernel_size=5):
    """
    Apply noise reduction techniques:
      - Gaussian blur to smooth the image.
      - Median filtering to reduce salt-and-pepper noise.
      - Morphological opening with a vertical kernel to remove vertical noise.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        gaussian_sigma (float): Sigma for Gaussian blur.
        median_kernel_size (int): Kernel size for median filtering (must be odd).
        morph_kernel_size (int): Kernel size for morphological operations.
    
    Returns:
        numpy.ndarray: The noise-reduced image.
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), gaussian_sigma)
    
    # Ensure the median kernel size is odd
    if median_kernel_size % 2 == 0:
        median_kernel_size += 1
    median_filtered = cv2.medianBlur(blurred, median_kernel_size)
    
    # Create a vertical kernel for morphological opening
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, morph_kernel_size))
    opened = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, vertical_kernel)
    
    return opened

def process_image(input_path, crop_coords, grayscale_output, colormap_output, colormap_legend_output,
                  gaussian_sigma=1.0, median_kernel_size=3, morph_kernel_size=5):
    """
    Crops an image, applies noise reduction, converts it to grayscale, applies an adaptive colormap,
    adds a legend, and saves the results.

    Parameters:
        input_path (str): Path to the input image.
        crop_coords (tuple): Cropping coordinates (x, y, width, height).
        grayscale_output (str): Path to save the cropped grayscale (noise-reduced) image.
        colormap_output (str): Path to save the colormap image.
        colormap_legend_output (str): Path to save the colormap image with a legend.
        gaussian_sigma (float): Sigma for Gaussian blur.
        median_kernel_size (int): Kernel size for median filtering.
        morph_kernel_size (int): Kernel size for morphological vertical noise removal.
    """
    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load '{input_path}'. Skipping.")
        return

    # Crop the image
    crop_x, crop_y, crop_width, crop_height = crop_coords
    cropped_image = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

    # Convert the cropped image to grayscale
    gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply noise reduction techniques to the grayscale image
    noise_reduced_image = reduce_noise(gray_cropped_image, gaussian_sigma, median_kernel_size, morph_kernel_size)
    
    # Save the noise-reduced grayscale image
    cv2.imwrite(grayscale_output, noise_reduced_image)
    print(f"Cropped noise-reduced grayscale image saved as '{grayscale_output}'.")

    # Invert the noise-reduced grayscale image to highlight whites
    inverted_gray_image = cv2.bitwise_not(noise_reduced_image)

    # Calculate adaptive intensity range using percentiles (2% and 98%)
    lower = np.percentile(inverted_gray_image, 2)
    upper = np.percentile(inverted_gray_image, 98)

    # Clip and normalize the image based on the adaptive range
    clipped_image = np.clip(inverted_gray_image, lower, upper)
    normalized_image = cv2.normalize(clipped_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_image = cv2.convertScaleAbs(normalized_image)

    # Apply JET colormap to the normalized image
    colormap_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)

    # Save the colormap image
    cv2.imwrite(colormap_output, colormap_image)
    print(f"Colormap image saved as '{colormap_output}'.")

    # Add a legend with adaptive range to the colormap image and save it
    add_legend_to_colormap(colormap_image, colormap_legend_output, lower, upper)

def process_folder(input_folder, background_image_path,
                   subtracted_root, grayscale_root,
                   colormap_root, colormap_legend_root,
                   crop_coords, gaussian_sigma=1.0,
                   median_kernel_size=3, morph_kernel_size=5):
    """
    Recursively processes all images in the input folder. For each image, the code subtracts a background,
    applies noise reduction and colormap processing, and saves the output files in a folder hierarchy
    that mirrors the input structure.

    Parameters:
        input_folder (str): Root folder containing input images (including subfolders).
        background_image_path (str): Path to the background image.
        subtracted_root (str): Root folder to save subtracted images.
        grayscale_root (str): Root folder to save grayscale images.
        colormap_root (str): Root folder to save colormap images.
        colormap_legend_root (str): Root folder to save colormap images with legends.
        crop_coords (tuple): Cropping coordinates (x, y, width, height).
        gaussian_sigma (float): Sigma for Gaussian blur.
        median_kernel_size (int): Kernel size for median filtering.
        morph_kernel_size (int): Kernel size for morphological operations.
    """
    for root, _, files in os.walk(input_folder):
        # Determine relative path from input_folder
        rel_path = os.path.relpath(root, input_folder)

        # Create corresponding output subdirectories
        subtracted_folder = os.path.join(subtracted_root, rel_path)
        grayscale_folder = os.path.join(grayscale_root, rel_path)
        colormap_folder = os.path.join(colormap_root, rel_path)
        colormap_legend_folder = os.path.join(colormap_legend_root, rel_path)
        os.makedirs(subtracted_folder, exist_ok=True)
        os.makedirs(grayscale_folder, exist_ok=True)
        os.makedirs(colormap_folder, exist_ok=True)
        os.makedirs(colormap_legend_folder, exist_ok=True)

        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue

            input_path = os.path.join(root, filename)
            # Define output paths preserving the directory structure and file name
            subtracted_image_path = os.path.join(subtracted_folder, f"subtracted_{filename}")
            grayscale_output = os.path.join(grayscale_folder, f"grayscale_{filename}")
            colormap_output = os.path.join(colormap_folder, f"colormap_{filename}")
            colormap_legend_output = os.path.join(colormap_legend_folder, f"colormap_legend_{filename}")

            # Subtract the background image
            subtracted_path = subtract_images(input_path, background_image_path, subtracted_image_path)
            if subtracted_path:
                process_image(
                    subtracted_path, crop_coords, grayscale_output,
                    colormap_output, colormap_legend_output,
                    gaussian_sigma, median_kernel_size, morph_kernel_size
                )

# Example usage:
if __name__ == "__main__":
    # Define paths
    input_folder = "C:/Users/garci/Desktop/Test1/selected_img_raw/"                  # Root folder containing input images (and subfolders)
    background_image_path = "subtracao_fundo.png"    # Path to the background image

    # Define output root folders (these will mirror the input folder hierarchy)
    subtracted_root = "output_images/subtracted"
    grayscale_root = "output_images/grayscale"
    colormap_root = "output_images/colormap"
    colormap_legend_root = "output_images/colormap_with_legend"

    # Define cropping coordinates (x, y, width, height)
    crop_coords = (240, 30, 300, 670)

    # Noise reduction parameters (adjust as needed)
    gaussian_sigma = 1.0
    median_kernel_size = 3
    morph_kernel_size = 5

    process_folder(
        input_folder, background_image_path,
        subtracted_root, grayscale_root,
        colormap_root, colormap_legend_root,
        crop_coords, gaussian_sigma, median_kernel_size, morph_kernel_size
    )

