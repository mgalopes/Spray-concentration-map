import cv2
import os
import numpy as np

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
    # Detect connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    if num_labels < 2:
        return binary_image

    # Identify the largest component (main object)
    main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_mask = (labels == main_label).astype(np.uint8) * 255

    # Find contours of the main object
    contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary_image

    # Create a mask from the contour
    contour_mask = np.zeros_like(main_mask)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)

    # Compute distance transform from the main object's border
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(contour_mask), cv2.DIST_L2, 5)

    # Create filtered mask, keeping only the main object and nearby details
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
    
    # Compute the refined mask
    filtered_mask = filter_by_contour_distance(binary, max_distance)

    # Apply morphological closing to remove small holes
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

    # Create a soft mask and blend with the original image
    mask_float = closed_mask.astype(np.float32) / 255.0
    if len(original.shape) == 3:
        mask_float = cv2.merge([mask_float, mask_float, mask_float])
    blended = (original.astype(np.float32) * mask_float).astype(np.uint8)

    return blended

def process_folders(input_folder, output_folder, max_distance=50):
    """
    Processes all subfolders in the input folder:
      - Applies noise reduction to each image,
      - Computes average images from processed images,
      - Saves the results in the output folder.
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

    # Process each folder and average the resulting images
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
            # Create a unique filename based on the folder structure
            relative_path = os.path.relpath(folder_path, input_folder).replace(os.sep, '_')
            avg_image_filename = f"average_image_{relative_path}.png"
            avg_image_path = os.path.join(output_folder, avg_image_filename)
            cv2.imwrite(avg_image_path, avg_img)
            print(f"  --> Average image saved to {avg_image_path}\n")
        else:
            print(f"  --> No valid images found in {folder_path}\n")

if __name__ == "__main__":
    input_folder = "C:/Users/garci/Desktop/Test1/output_images/adaptative_threshold"
    output_folder = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/average_results"
    process_folders(input_folder, output_folder, max_distance=50)

