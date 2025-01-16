import cv2
import os
import numpy as np


def load_images_from_folder(folder):
    """
    Loads all valid images from a folder.

    Parameters:
        folder (str): Path to the folder.

    Returns:
        list: List of loaded images.
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
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
        numpy.ndarray: The average image.
    """
    if len(images) == 0:
        return None
    average_img = np.zeros_like(images[0], dtype=np.float64)
    for img in images:
        average_img += img / len(images)
    average_img = np.round(average_img).astype(np.uint8)
    return average_img


def process_folders(input_folder, output_folder):
    """
    Processes all subfolders in the input folder, computes average images, and saves results.

    Parameters:
        input_folder (str): Path to the root input folder containing subfolders with images.
        output_folder (str): Path to the root output folder for average images.
    """
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        # Skip folders without image files
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        if not images:
            continue

        # Load images and compute the average
        folder_path = root
        images = load_images_from_folder(folder_path)
        average_img = average_images(images)

        if average_img is not None:
            # Generate a unique filename based on the folder hierarchy
            relative_path = os.path.relpath(folder_path, input_folder).replace(os.sep, '_')
            avg_image_filename = f"average_image_{relative_path}.png"

            # Save the average image directly in the output folder
            avg_image_path = os.path.join(output_folder, avg_image_filename)
            cv2.imwrite(avg_image_path, average_img)
            print(f"Average image saved to {avg_image_path}")
        else:
            print(f"No valid images found in {folder_path}")


if __name__ == "__main__":
    input_folder = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/output_images/colormap"
    output_folder = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/average_results"
    process_folders(input_folder, output_folder)
