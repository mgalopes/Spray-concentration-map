import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def generate_histograms_and_line_overlay(colormap_folder, line_height, output_histogram_folder, output_image_folder):
    """
    Generates histograms for all colormap images in a folder at a specified height
    and saves the output images with the selected line overlay. Histograms are saved
    with transparent backgrounds for future overlays.

    Parameters:
        colormap_folder (str): Path to the folder containing colormap images.
        line_height (int): The height (row) from which to extract intensity data.
        output_histogram_folder (str): Path to save the histogram plots.
        output_image_folder (str): Path to save the images with line overlay.
    """
    # Ensure the output folders exist
    os.makedirs(output_histogram_folder, exist_ok=True)
    os.makedirs(output_image_folder, exist_ok=True)

    # Iterate through all images in the folder
    for filename in os.listdir(colormap_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue  # Skip non-image files

        colormap_image_path = os.path.join(colormap_folder, filename)
        output_histogram_path = os.path.join(output_histogram_folder, f"histogram_{os.path.splitext(filename)[0]}_height_{line_height/3.36}.png")
        output_image_path = os.path.join(output_image_folder, f"line_overlay_{os.path.splitext(filename)[0]}_height_{line_height/3.36}.png")


        # Load the colormap image
        colormap_image = cv2.imread(colormap_image_path)
        if colormap_image is None:
            print(f"Error: Could not load '{colormap_image_path}'. Skipping.")
            continue

        # Ensure the line height is within image bounds
        if line_height < 0 or line_height >= colormap_image.shape[0]:
            print(f"Error: Line height {line_height} is out of bounds for '{filename}'. Skipping.")
            continue

        # Extract the row of pixel intensities
        line_data = colormap_image[line_height, :, :]  # Shape: (width, channels)

        # Convert to grayscale to get a single intensity value per pixel
        gray_line = cv2.cvtColor(colormap_image, cv2.COLOR_BGR2GRAY)[line_height, :]
        normalized_intensity = gray_line / 255.0  # Normalize to range [0, 1]

        # Generate pixel positions
        pixel_positions = np.arange(line_data.shape[0])

        # Plot the histogram with a transparent background
        plt.figure(figsize=(8, 4))
        plt.plot(pixel_positions, normalized_intensity, color='red', label='Normalized Intensity')
        plt.title(f"Color Intensity Profile at Height {line_height/3.36}mm")
        plt.xlabel("Pixel Position")
        plt.ylabel("Normalized Intensity (0-1)")
        plt.ylim(0, 1)  # Fix y-axis range
        plt.legend()
        plt.grid(True)

        # Save the histogram plot with a transparent background
        plt.savefig(output_histogram_path, dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        print(f"Histogram saved as '{output_histogram_path}' with a transparent background.")

        # Create a copy of the image with a line overlay
        line_overlay_image = colormap_image.copy()
        line_color = (0, 0, 255)  # Red color in BGR format
        line_thickness = 2
        cv2.line(line_overlay_image, (0, line_height), (line_overlay_image.shape[1] - 1, line_height), line_color, line_thickness)

        # Save the image with the line overlay
        cv2.imwrite(output_image_path, line_overlay_image)
        print(f"Line overlay image saved as '{output_image_path}'.")

if __name__ == "__main__":
    colormap_folder = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/input_for_histogram"  # Folder containing colormap images
    line_height = 588 # Row in the image to analyze
    output_histogram_folder = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/output_histograms"  # Folder to save histograms
    output_image_folder = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/output_line_overlay"  # Folder to save images with line overlay

    # Generate histograms and line overlay images for all images in the folder
    generate_histograms_and_line_overlay(colormap_folder, line_height, output_histogram_folder, output_image_folder)
