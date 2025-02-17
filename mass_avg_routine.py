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
    average_img = np.round(average_img).astype(np.uint8)
    return average_img

def filter_by_contour_distance(binary_image, max_distance):
    """
    Given a binary image, identifies the main object (largest connected component),
    extracts its contour, and then removes (clears) other connected components
    whose pixels are all farther than max_distance from the nearest contour pixel
    of the main object.
    
    Parameters:
        binary_image (np.ndarray): Binary image (0/255) with potential objects.
        max_distance (float): Maximum allowed distance from the main object's frontier.
    
    Returns:
        np.ndarray: Filtered binary image.
    """
    # Detect connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    if num_labels < 2:
        return binary_image

    # Identify the main object as the largest component (ignore background, label 0)
    main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_mask = (labels == main_label).astype(np.uint8) * 255

    # Find the contour (fronteira) of the main object
    contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return binary_image
    main_contour = contours[0]

    # Create a mask for the contour by drawing it with thickness 1
    contour_mask = np.zeros_like(main_mask)
    cv2.drawContours(contour_mask, [main_contour], -1, 255, thickness=1)

    # Calculate the distance transform: for each pixel, the distance to the contour
    inverted_contour = cv2.bitwise_not(contour_mask)
    dist_transform = cv2.distanceTransform(inverted_contour, cv2.DIST_L2, 5)

    # Start the filtered mask with the main object already included
    filtered_mask = main_mask.copy()

    # For each component (except background and the main object)
    for label in range(1, num_labels):
        if label == main_label:
            continue
        component_mask = (labels == label).astype(np.uint8) * 255
        # Get the distances for the pixels in this component
        component_pixels = dist_transform[component_mask == 255]
        if component_pixels.size == 0:
            continue
        min_distance = np.min(component_pixels)
        # If the minimum distance is less than or equal to the threshold, keep the component
        if min_distance <= max_distance:
            filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)
    return filtered_mask

def process_image(image, max_distance):
    """
    Processes an image:
      - Converts to grayscale for processing;
      - Applies Gaussian blur;
      - Applies Otsu thresholding for binarization;
      - Removes objects that are farther than max_distance from the main object's contour.
      - Uses the binary mask to extract the colored region of interest.
    """
    # Keep a copy of the original image
    original = image.copy()
    # Convert to grayscale for processing (if the image is colored)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filtered_mask = filter_by_contour_distance(binary, max_distance)
    
    # If the original image is colored, apply the mask to preserve colors
    if len(original.shape) == 3:
        color_filtered = cv2.bitwise_and(original, original, mask=filtered_mask)
        return color_filtered
    else:
        return filtered_mask

def process_folders(input_folder, output_folder, max_distance=50):
    """
    Processes all subfolders in the input folder, applies noise reduction to each image,
    computes average images, and saves results.
    
    Parameters:
        input_folder (str): Root folder with subfolders containing images.
        output_folder (str): Folder where average images will be saved.
        max_distance (float): Distance threshold to filter out objects far from the main object's frontier.
    """
    os.makedirs(output_folder, exist_ok=True)

    # First, collect all directories that contain valid images
    valid_folders = []
    for root, dirs, files in os.walk(input_folder):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        if image_files:
            valid_folders.append(root)

    total_folders = len(valid_folders)
    print(f"Found {total_folders} folders with valid images.\n")

    # Process each valid folder with a progress message
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
            # Generate a unique filename based on the folder hierarchy
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
    # Adjust max_distance as needed to remove objects far from the main object's frontier
    process_folders(input_folder, output_folder, max_distance=50)
