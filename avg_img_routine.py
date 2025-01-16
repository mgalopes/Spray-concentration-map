import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def average_images(images):
    if len(images) == 0:
        return None
    average_img = np.zeros_like(images[0], dtype=np.float64)
    for img in images:
        average_img += img / len(images)
    average_img = np.round(average_img).astype(np.uint8)
    return average_img

def main(folder_path, output_path):
    images = load_images_from_folder(folder_path)
    average_img = average_images(images)

    if average_img is not None:
        cv2.imwrite(output_path, average_img)
        print(f"Average image saved to {output_path}")
    else:
        print("No valid images found in the folder.")

if __name__ == "__main__":
    folder_path = "C:/Users/garci/Documents/GitHub/Spray-concentration-map/output_images/colormap"
    output_path = "/Users/garci/Desktop/Test1/agua_dest_div_25_70bar2.png"
    main(folder_path, output_path)