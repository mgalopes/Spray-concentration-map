import cv2
import numpy as np
import sys


def subtract_images(image1_path, image2_path, output_path='subtracted_image.jpg'):
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
        print(f"Error: Could not load '{image1_path}'. Check the file path.")
        sys.exit(1)
    if image2 is None:
        print(f"Error: Could not load '{image2_path}'. Check the file path.")
        sys.exit(1)

    # Ensure the images are of the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must be of the same size")

    # Subtract the images
    subtracted_image = cv2.subtract(image1, image2)

    # Save the result
    cv2.imwrite(output_path, subtracted_image)
    print(f"Subtracted image saved as '{output_path}'")

    # Display the result
    cv2.imshow('Subtracted Image', subtracted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_image(input_path, crop_coords, grayscale_output='cropped_image_grayscale.jpg',
                  colormap_output='colormap_image.jpg'):
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
        print(f"Error: Could not load '{input_path}'. Check the file path.")
        exit(1)

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

    # Display the results
    cv2.imshow("Cropped Image (Grayscale)", gray_cropped_image)
    cv2.imshow("Colormap Image", colormap_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example Usage:
if __name__ == "__main__":
    # Subtract two images
    subtract_images('etanol_div_25C_70bar.jpg', 'subtracao_fundo.png')

    # Process the subtracted image (crop and create colormap)
    process_image(
        input_path='subtracted_image.jpg',
        crop_coords=(240, 30, 300, 670)  # Example cropping coordinates
    )
