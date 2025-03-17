import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

###########################
# GLOBAL THRESHOLD CALCULATION
###########################
def compute_global_thresholds(root_folder):
    """
    Scans all images (in root_folder and its subfolders) and returns:
      - global_min: the minimum nonzero intensity found
      - global_max: the maximum intensity found

    Prints progress while scanning.
    """
    global_min = 255  # highest possible intensity in an 8-bit image
    global_max = 0
    image_paths = []
    
    # Collect all image file paths
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(root, file))
    
    total_images = len(image_paths)
    print(f"Found {total_images} images. Scanning for global thresholds...")
    
    # Process each image and update global thresholds
    for idx, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Only consider nonzero pixels for the minimum
            nonzero = img[img > 0]
            if nonzero.size > 0:
                cur_min = nonzero.min()
                cur_max = img.max()
                if cur_min < global_min:
                    global_min = cur_min
                if cur_max > global_max:
                    global_max = cur_max
        if idx % 10 == 0 or idx == total_images:
            print(f"Scanned {idx}/{total_images} images. Current thresholds: min = {global_min}, max = {global_max}")
    
    if global_min == 255:
        global_min = 1
    if global_max == 0:
        global_max = 255
    print(f"Global threshold values computed: min = {global_min}, max = {global_max}")
    return global_min, global_max

###########################
# IMAGE LOADING & AVERAGING
###########################
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
    Computes the average of a list of images (pixelwise average).
    Assumes the images are already in a jet-colormap scheme.
    """
    if len(images) == 0:
        return None
    average_img = np.zeros_like(images[0], dtype=np.float64)
    for img in images:
        average_img += img / len(images)
    return np.round(average_img).astype(np.uint8)

###########################
# PROCESSING FUNCTIONS (NEW THRESHOLDING)
###########################
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

def process_image(image, max_distance, thresh_min, thresh_max):
    """
    Processes an image by:
      - Converting to grayscale,
      - Applying Gaussian blur,
      - Thresholding using the global thresh_min and thresh_max (via inRange),
      - Filtering by contour distance,
      - Applying morphological closing,
      - Using a soft mask to blend and preserve original colors.
    
    Note: The input images are already in a jet-colormap scheme.
    """
    original = image.copy()
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.inRange(blurred, int(thresh_min), int(thresh_max))
    
    filtered_mask = filter_by_contour_distance(binary, max_distance)
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
    
    mask_float = closed_mask.astype(np.float32) / 255.0
    if len(original.shape) == 3:
        mask_float = cv2.merge([mask_float, mask_float, mask_float])
    blended = (original.astype(np.float32) * mask_float).astype(np.uint8)
    
    return blended

###########################
# PLOTTING & SAVING FUNCTIONS
###########################
def plot_and_save_average_image(avg_img, output_path, folder_name, axis_mode, mm_per_pixel_x, mm_per_pixel_y):
    """
    Displays the raw averaged image (which is already in jet-colormap) in a figure.
    The averaged image is placed in the Matplotlib grid without reprocessing.
    For axis settings, if axis_mode=='mm', pixel coordinates are converted to mm.
    A colorbar is created using a ScalarMappable with the 'jet' colormap, but its range is fixed to 0-1.
    This ensures the legend shows intensity values from 0 to 1.
    
    The image is normalized (divided by 255) for plotting so that its intensities are scaled to [0, 1].
    """
    # Convert BGR to RGB if the image is color
    if avg_img.ndim == 3 and avg_img.shape[2] == 3:
        image_to_show = cv2.cvtColor(avg_img.copy(), cv2.COLOR_BGR2RGB)
    else:
        image_to_show = avg_img.copy()
    
    # Normalize the image for plotting: convert intensities to [0,1]
    norm_data = image_to_show.astype(np.float32) / 255.0
    
    # Fix the colorbar range to 0-1
    vmin = 0.0
    vmax = 1.0
    
    height, width = image_to_show.shape[:2]
    dpi = 100
    
    if axis_mode == "mm":
        x_max = width * mm_per_pixel_x
        y_max = height * mm_per_pixel_y
        extent = [0, x_max, y_max, 0]
        tick_spacing = 25
        xticks = np.arange(0, x_max, tick_spacing)
        yticks = np.arange(0, y_max, tick_spacing)
        xlabel = "X (mm)"
        ylabel = "Y (mm)"
    else:
        x_max = width
        y_max = height
        extent = [0, x_max, y_max, 0]
        xticks = np.arange(0, x_max, 50)
        yticks = np.arange(0, y_max, 100)
        xlabel = "X Pixel"
        ylabel = "Y Pixel"
    
    image_width_inch = width / dpi
    image_height_inch = height / dpi
    colorbar_width_inch = 0.5
    total_fig_width = image_width_inch + colorbar_width_inch
    
    fig = plt.figure(figsize=(total_fig_width, image_height_inch), dpi=dpi)
    ax_img = fig.add_axes([0, 0, image_width_inch / total_fig_width, 1])
    
    # Display the RGB image without applying a colormap
    im = ax_img.imshow(norm_data, origin='upper', vmin=vmin, vmax=vmax,
                       extent=extent, aspect='equal')
    ax_img.set_title(f"Averaged Image: {folder_name}", pad=20)
    ax_img.set_xlabel(xlabel)
    ax_img.set_ylabel(ylabel)
    ax_img.set_xticks(xticks)
    ax_img.set_yticks(yticks)
    ax_img.grid(True, which='both', linestyle='--', color='white', linewidth=0.5)
    
    # Create a ScalarMappable for the colorbar with fixed range 0-1
    sm = ScalarMappable(cmap='jet', norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # dummy, required for colorbar
    
    pos = ax_img.get_position()
    cb_pad = 0.01
    cb_width = 0.05
    ax_cb = fig.add_axes([pos.x1 + cb_pad, pos.y0, cb_width, pos.height])
    ax_cb.set_facecolor('white')
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Intensity")
    
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


###########################
# FOLDER PROCESSING
###########################
def process_folders(input_folder, output_folder, max_distance=50, axis_mode="pixels",
                    mm_per_pixel_x=1.0, mm_per_pixel_y=1.0):
    """
    Processes each subfolder in the input folder:
      - Computes global thresholds for the entire folder,
      - Applies processing using global thresholding,
      - Computes the average image from processed images,
      - Saves the raw average image and a corresponding graph.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    global_min, global_max = compute_global_thresholds(input_folder)
    
    valid_folders = []
    for root, dirs, files in os.walk(input_folder):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        if image_files:
            valid_folders.append(root)
    total_folders = len(valid_folders)
    print(f"Found {total_folders} folders with valid images.\n")
    
    for idx, folder_path in enumerate(valid_folders, start=1):
        print(f"Processing folder {idx}/{total_folders}: {folder_path}")
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        processed_images = []
        for filename in image_files:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                proc = process_image(img, max_distance, global_min, global_max)
                processed_images.append(proc)
        if processed_images:
            avg_img = average_images(processed_images)
            relative_path = os.path.relpath(folder_path, input_folder).replace(os.sep, '_')
            raw_filename = f"average_image_{relative_path}.png"
            graph_filename = f"average_image_graph_{relative_path}.png"
            raw_path = os.path.join(output_folder, raw_filename)
            graph_path = os.path.join(output_folder, graph_filename)
            cv2.imwrite(raw_path, avg_img)
            plot_and_save_average_image(avg_img, graph_path, relative_path, axis_mode, mm_per_pixel_x, mm_per_pixel_y)
            print(f"  --> Raw average image saved to {raw_path}")
            print(f"  --> Graph with color scheme saved to {graph_path}\n")
        else:
            print(f"  --> No valid images found in {folder_path}\n")

###########################
# MAIN
###########################
if __name__ == "__main__":
    axis_choice = input("Do you want the axis in mm or in pixels? Enter 'mm' or 'pixels': ").strip().lower()
    if axis_choice == "mm":
        try:
            mm_per_pixel_x = float(input("Enter mm per pixel for x: ").strip())
            mm_per_pixel_y = float(input("Enter mm per pixel for y: ").strip())
        except ValueError:
            print("Invalid input for mm conversion. Using default 1.0 mm per pixel.")
            mm_per_pixel_x = 1.0
            mm_per_pixel_y = 1.0
    else:
        mm_per_pixel_x = 1.0
        mm_per_pixel_y = 1.0

    input_folder = "C:/Users/garci/Desktop/Test1/output_images/colormap"
    output_folder = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/average_results"
    process_folders(input_folder, output_folder, max_distance=50,
                    axis_mode=axis_choice, mm_per_pixel_x=mm_per_pixel_x, mm_per_pixel_y=mm_per_pixel_y)
