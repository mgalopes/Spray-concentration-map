import cv2
import sys

# Load the images
image1 = cv2.imread('etanol_div_25C_70bar.jpg')
image2 = cv2.imread('subtracao_fundo.png')

# Check if the images were loaded successfully
if image1 is None:
    print("Error: Could not load 'gasolina_div_40C_70bar.png'. Check the file path.")
    sys.exit(1)
if image2 is None:
    print("Error: Could not load 'subtracao_fundo2.png'. Check the file path.")
    sys.exit(1)

# Ensure the images are of the same size
if image1.shape != image2.shape:
    raise ValueError("Images must be of the same size")

# Subtract the images
subtracted_image = cv2.subtract(image1, image2)

# Display the result
cv2.imshow('Subtracted Image', subtracted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
output_path = 'subtracted_image.jpg'
cv2.imwrite(output_path, subtracted_image)
print(f"Subtracted image saved as '{output_path}'")
