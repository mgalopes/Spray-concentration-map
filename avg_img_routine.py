import cv2
import os
import numpy as np


def load_images_from_folder(folder):
    """
    Recursively loads all valid images from a folder and its subfolders.

    Parameters:
        folder (str): Path to the root folder.

    Returns:
        list: List of loaded images.
    """
    images = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
    return images


def average_images(images):
    """
    Computes the average of a list of images.

    Parameters:
        images (list): List of images.

    Returns:
        numpy.ndarray: The average image, or None if the list is empty.
    """
    if len(images) == 0:
        return None
    average_img = np.zeros_like(images[0], dtype=np.float64)
    for img in images:
        average_img += img / len(images)
    average_img = np.round(average_img).astype(np.uint8)
    return average_img


def process_folder(input_folder, output_folder):
    """
    Processes all subfolders in the input folder, computes average images, and saves results.

    Parameters:
        input_folder (str): Path to the root input folder containing images and subfolders.
        output_folder (str): Path to the root output folder for average images.
    """
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        # Collect images only for this specific folder
        images = [
            cv2.imread(os.path.join(root, file))
            for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]

        if not images:
            print(f"No valid images found in {root}. Skipping.")
            continue

        # Compute the average image
        average_img = average_images(images)

        if average_img is not None:
            # Generate a unique filename based on the folder hierarchy
            relative_path = os.path.relpath(root, input_folder).replace(os.sep, '_')
            avg_image_filename = f"average_image_{relative_path}.png"

            # Save the average image to the output folder
            avg_image_path = os.path.join(output_folder, avg_image_filename)
            cv2.imwrite(avg_image_path, average_img)
            print(f"Average image saved to {avg_image_path}")
        else:
            print(f"Failed to compute average image for {root}")


if __name__ == "__main__":
    input_folder = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/output_images/grayscale"
    output_folder = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/average_results/grayscale"
    process_folder(input_folder, output_folder)
