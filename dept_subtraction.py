import cv2

# Load the images
image1 = cv2.imread('gasolina_conv_40C_50bar.png')
image2 = cv2.imread('subtracao_fundo2.png')

# Ensure the images are of the same size
if image1.shape != image2.shape:
    raise ValueError("Images must be of the same size")

# Subtract the images
subtracted_image = cv2.subtract(image1, image2)

# Display or save the result
cv2.imshow('Subtracted Image', subtracted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# If you want to save the result
cv2.imwrite('subtracted_image.jpg', subtracted_image)
