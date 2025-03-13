import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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

def plot_and_save_average_image(avg_img, output_path, folder_name, axis_mode, mm_per_pixel_x, mm_per_pixel_y):
    """
    Displays the average image in a figure that:
      - Preserves the image's original pixel scale (no distortion),
      - Displays with (0,0) at the top-left,
      - Uses mm units if axis_mode=='mm' (with grid lines every 25 mm) or pixels if axis_mode=='pixels'
      - Places a colorbar to the right whose height exactly matches the image height.
      
    The function creates the image axis manually so its size (in inches) equals the image's size (width/dpi, height/dpi).
    Then it retrieves its position and creates a colorbar axis immediately to its right with the same height.
    """
    # Convert to grayscale if needed.
    if len(avg_img.shape) == 3 and avg_img.shape[2] == 3:
        gray_avg = cv2.cvtColor(avg_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_avg = avg_img

    # Normalize image to [0,1]
    norm_data = gray_avg.astype(np.float32) / 255.0
    height, width = norm_data.shape
    dpi = 100

    if axis_mode == "mm":
        x_max = width * mm_per_pixel_x
        y_max = height * mm_per_pixel_y
        extent = [0, x_max, y_max, 0]
        tick_spacing = 25  # every 25 mm
        xticks = np.arange(0, x_max, tick_spacing)
        yticks = np.arange(0, y_max, tick_spacing)
        xlabel = "X (mm)"
        ylabel = "Y (mm)"
    else:
        x_max = width
        y_max = height
        extent = [0, x_max, y_max, 0]
        xticks = np.arange(0, x_max, 50)   # every 50 pixels in X
        yticks = np.arange(0, y_max, 100)  # every 100 pixels in Y
        xlabel = "X Pixel"
        ylabel = "Y Pixel"

    # Compute image size in inches (1:1 pixel scale)
    image_width_inch = width / dpi
    image_height_inch = height / dpi

    # Reserve a fixed width for the colorbar (in inches)
    colorbar_width_inch = 0.5
    total_fig_width = image_width_inch + colorbar_width_inch

    # Create figure with the total width
    fig = plt.figure(figsize=(total_fig_width, image_height_inch), dpi=dpi)
    # Create the image axis with exactly the image size:
    ax_img = fig.add_axes([0, 0, image_width_inch / total_fig_width, 1])
    
    im = ax_img.imshow(norm_data, cmap='viridis', origin='upper', vmin=0, vmax=1,
                       extent=extent, aspect='equal')
    ax_img.set_title(f"Averaged Image: {folder_name}", pad=20)
    ax_img.set_xlabel(xlabel)
    ax_img.set_ylabel(ylabel)
    ax_img.set_xticks(xticks)
    ax_img.set_yticks(yticks)
    ax_img.grid(True, which='both', linestyle='--', color='white', linewidth=0.5)

    # Get the position of the image axis in figure coordinates
    pos = ax_img.get_position()
    # Create a colorbar axis immediately to the right of the image axis, with the same height
    cb_pad = 0.01  # small padding in figure fraction
    cb_width = 0.05  # width in figure fraction (you can adjust if needed)
    ax_cb = fig.add_axes([pos.x1 + cb_pad, pos.y0, cb_width, pos.height])
    ax_cb.set_facecolor('white')
    cbar = fig.colorbar(im, cax=ax_cb)
    cbar.set_label("Normalized Colour Intensity (0-1)")

    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

def process_folders(input_folder, output_folder, max_distance=50, axis_mode="pixels", mm_per_pixel_x=1.0, mm_per_pixel_y=1.0):
    """
    Processes all subfolders in the input folder:
      - Applies noise reduction to each image,
      - Computes average images from processed images,
      - Saves the raw average image and a corresponding graph with a colorbar.
      
    axis_mode determines if the graph uses "mm" or "pixels". For "mm" mode, mm_per_pixel_x and mm_per_pixel_y
    convert pixel dimensions to millimeters.
    """
    os.makedirs(output_folder, exist_ok=True)
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
                proc = process_image(img, max_distance)
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

    input_folder = "C:/Users/garci/Desktop/Test1/output_images/adaptative_threshold"
    output_folder = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/average_results"
    process_folders(input_folder, output_folder, max_distance=50,
                    axis_mode=axis_choice, mm_per_pixel_x=mm_per_pixel_x, mm_per_pixel_y=mm_per_pixel_y)
