import cv2
import numpy as np
import os
import sys


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


def process_image(input_path, crop_coords, grayscale_output, colormap_output):
    """
    Crops an image, converts it to grayscale, applies a colormap, and saves the results.

    Parameters:
        input_path (str): Path to the input image.
        crop_coords (tuple): Cropping coordinates (x, y, width, height).
        grayscale_output (str): Path to save the cropped grayscale image.
        colormap_output (str): Path to save the colormap image.
    """
    # Load the image
    image = cv2.imread(input_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load '{input_path}'. Skipping.")
        return

    # Crop the image
    crop_x, crop_y, crop_width, crop_height = crop_coords
    cropped_image = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

    # Convert the cropped image to grayscale
    gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Normalize the intensity range and apply a colormap
    normalized_image = cv2.normalize(gray_cropped_image, None, 0, 255, cv2.NORM_MINMAX)
    colormap_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)

    # Save the results
    cv2.imwrite(grayscale_output, gray_cropped_image)
    cv2.imwrite(colormap_output, colormap_image)

    print(f"Cropped grayscale image saved as '{grayscale_output}'.")
    print(f"Colormap image saved as '{colormap_output}'")


def process_folder(input_folder, background_image_path, output_folder, crop_coords):
    """
    Processes all images in a folder by subtracting a background and generating colormap outputs.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        background_image_path (str): Path to the background image.
        output_folder (str): Path to save the processed images.
        crop_coords (tuple): Cropping coordinates (x, y, width, height).
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all images in the folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        # Subtract the background
        subtracted_image_path = os.path.join(output_folder, f"subtracted_{filename}")
        subtracted_path = subtract_images(input_path, background_image_path, subtracted_image_path)

        if subtracted_path:
            # Process the subtracted image
            grayscale_output = os.path.join(output_folder, f"grayscale_{filename}")
            colormap_output = os.path.join(output_folder, f"colormap_{filename}")
            process_image(subtracted_path, crop_coords, grayscale_output, colormap_output)


# Example Usage:
if __name__ == "__main__":
    input_folder = "input_images"  # Folder containing input images
    background_image_path = "subtracao_fundo.png"  # Path to the background image
    output_folder = "output_images"  # Folder to save processed images
    crop_coords = (240, 30, 300, 670)  # Example cropping coordinates

    process_folder(input_folder, background_image_path, output_folder, crop_coords)