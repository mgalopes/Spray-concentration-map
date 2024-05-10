import cv2
import numpy as np

def gamma_correction(image, gamma):
    # Normalize the pixel values to the range [0, 1]
    normalized_image = image / 255.0

    # Apply gamma correction
    gamma_corrected = np.power(normalized_image, gamma)

    # Scale the pixel values back to [0, 255]
    gamma_corrected = np.uint8(gamma_corrected * 255)

    return gamma_corrected

if __name__ == "__main__":
    # Load the image using OpenCV
    image_path = "Img000009.tif"  # Replace with the actual image file path
    original_image = cv2.imread(image_path)

    # Set the gamma value (experiment with different values to get the desired effect)
    gamma_value = 0.45

    # Apply gamma correction
    corrected_image = gamma_correction(original_image, gamma_value)

    # Display the original and gamma-corrected images
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Gamma Corrected Image", corrected_image)
    cv2.imwrite('gamacorrected.png', corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()