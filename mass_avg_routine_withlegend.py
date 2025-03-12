import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_images_from_folder(folder):
    """
    Loads all valid images from a folder.
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
    """
    if len(images) == 0:
        return None
    average_img = np.zeros_like(images[0], dtype=np.float64)
    for img in images:
        average_img += img / len(images)
    return np.round(average_img).astype(np.uint8)

def filter_by_contour_distance(binary_image, max_distance):
    """
    Filters objects based on their distance from the main object's contour, 
    preserving internal details while excluding distant artifacts.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    if num_labels < 2:
        return binary_image

    main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_mask = (labels == main_label).astype(np.uint8) * 255

    contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary_image

    contour_mask = np.zeros_like(main_mask)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)

    dist_transform = cv2.distanceTransform(cv2.bitwise_not(contour_mask), cv2.DIST_L2, 5)

    filtered_mask = main_mask.copy()
    for label in range(1, num_labels):
        if label == main_label:
            continue
        component_mask = (labels == label).astype(np.uint8) * 255
        component_pixels = dist_transform[component_mask == 255]
        if component_pixels.size == 0:
            continue
        min_distance = np.min(component_pixels)
        if min_distance <= max_distance:
            filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)

    return filtered_mask

def process_image(image, max_distance):
    """
    Processes an image by:
      - Converting to grayscale,
      - Applying Gaussian blur,
      - Binarizing with Otsu thresholding,
      - Filtering by contour distance,
      - Applying morphological closing,
      - Using a soft mask to blend and preserve original colors.
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    filtered_mask = filter_by_contour_distance(binary, max_distance)
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

    mask_float = closed_mask.astype(np.float32) / 255.0
    if len(original.shape) == 3:
        mask_float = cv2.merge([mask_float, mask_float, mask_float])
    blended = (original.astype(np.float32) * mask_float).astype(np.uint8)

    return blended

def plot_and_save_average_image(avg_img, output_path, folder_name):
    """
    Displays the average image in a matplotlib figure that:
      - Preserves the image's original pixel scale (no distortion),
      - Displays with (0,0) at the top-left,
      - Places a thicker colorbar on a white bar immediately to the right,
      - Shows dashed grid lines every 50 pixels in X and 100 pixels in Y,
      - Titles as "Averaged Image: <folder_name>" with extra space below the title,
      - Labels the colorbar as "Normalized Colour Intensity (0-1)".
    """
    # Convert to grayscale if needed for colormap
    if len(avg_img.shape) == 3 and avg_img.shape[2] == 3:
        gray_avg = cv2.cvtColor(avg_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_avg = avg_img

    # Normalize the image to [0,1] for display and colorbar range
    norm_data = gray_avg.astype(np.float32) / 255.0
    height, width = norm_data.shape

    # Determine figure size to approximate 1:1 pixel scale
    dpi = 100
    fig_width_inch = (width / dpi) + 0.8  # extra space for colorbar
    fig_height_inch = height / dpi
    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)

    ax_img = fig.add_subplot(111)
    # Set extent so that x=0...width and y=0 is at the top, y=height at the bottom.
    im = ax_img.imshow(
        norm_data,
        cmap='viridis',
        origin='upper',  
        vmin=0, vmax=1,
        extent=[0, width, height, 0],  # Top edge is y=0, bottom edge is y=height.
        aspect='equal'
    )

    # Add title with extra padding (space line) below the title.
    ax_img.set_title(f"Averaged Image: {folder_name}", pad=20)
    ax_img.set_xlabel("X Pixel")
    ax_img.set_ylabel("Y Pixel")
    
    # Set grid ticks: every 50 pixels in x and every 100 pixels in y.
    ax_img.set_xticks(np.arange(0, width+1, 50))
    ax_img.set_yticks(np.arange(0, height+1, 100))
    ax_img.grid(True, which='both', linestyle='--', color='white', linewidth=0.5)

    # Create a divider for the colorbar and append an axis on the right.
    divider = make_axes_locatable(ax_img)
    # Double the colorbar width from 6% to 12%
    cax = divider.append_axes("right", size="12%", pad=0.1)
    cax.set_facecolor('white')
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Normalized Colour Intensity (0-1)")

    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

def process_folders(input_folder, output_folder, max_distance=50):
    """
    Processes all subfolders in the input folder:
      - Applies noise reduction to each image,
      - Computes average images from processed images,
      - Saves the raw average image and a corresponding matplotlib graph with a colorbar.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Gather all directories containing valid images
    valid_folders = []
    for root, dirs, files in os.walk(input_folder):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        if image_files:
            valid_folders.append(root)

    total_folders = len(valid_folders)
    print(f"Found {total_folders} folders with valid images.\n")

    # Process each folder and compute the average image
    for idx, folder_path in enumerate(valid_folders, start=1):
        print(f"Processing folder {idx}/{total_folders}: {folder_path}")
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        processed_images = []
        for filename in image_files:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                proc = process_image(img, max_distance)
                processed_images.append(proc)

        if processed_images:
            avg_img = average_images(processed_images)
            # Create unique filenames based on folder structure
            relative_path = os.path.relpath(folder_path, input_folder).replace(os.sep, '_')
            raw_filename = f"average_image_{relative_path}.png"
            graph_filename = f"average_image_graph_{relative_path}.png"
            raw_path = os.path.join(output_folder, raw_filename)
            graph_path = os.path.join(output_folder, graph_filename)
            
            # Save the raw average image using OpenCV
            cv2.imwrite(raw_path, avg_img)
            # Save the matplotlib graph preserving the original scale and adding the thicker colorbar
            plot_and_save_average_image(avg_img, graph_path, relative_path)
            print(f"  --> Raw average image saved to {raw_path}")
            print(f"  --> Graph with color scheme saved to {graph_path}\n")
        else:
            print(f"  --> No valid images found in {folder_path}\n")

if __name__ == "__main__":
    input_folder = "C:/Users/garci/Desktop/Test1/output_images/adaptative_threshold"
    output_folder = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/average_results"
    process_folders(input_folder, output_folder, max_distance=50)
