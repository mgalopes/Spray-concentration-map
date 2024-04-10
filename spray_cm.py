import cv2

#subtração de fundo

def subtract_images(image1_path, image2_path):
    
    # Read the two input images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    if image1 is None or image2 is None:
        print("Error: One or both images could not be read.")
        return
    
    # Make sure both images have the same dimensions
    if image1.shape != image2.shape:
        print("Error: The images have different dimensions.")
        return
    
    # Subtract the images
    subtracted_image = cv2.subtract(image1, image2)

    cv2.imwrite("subtracted_image.png", subtracted_image )

    # Display the subtracted image
    cv2.imshow("Subtracted Image", subtracted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace these paths with the paths to your images
image1_path = "Img000226.tif"
image2_path = "fundo_Subtracao_coflow.png"

original=subtract_images(image1_path, image2_path)


#Campo de concentração
#original=cv2.imread('testes08_08\Frente\Img000003.tif', 1)
im_gray = cv2.imread('subtracted_image.png', cv2.IMREAD_GRAYSCALE)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
cv2.imshow("Original image", im_gray)
cv2.waitKey(0)
cv2.imshow("edgeDetection2", im_color)
cv2.waitKey(0)

#Desenvolvimento da remoção do fundo

# Applying thresholding technique
_, alpha = cv2.threshold(im_gray, 30, 255, cv2.THRESH_BINARY)
  
# Using cv2.split() to split channels 
# of coloured image
b, g, r = cv2.split(im_color)
  
# Making list of Red, Green, Blue
# Channels and alpha
rgba = [b, g, r, alpha]
  
# Using cv2.merge() to merge rgba
# into a coloured/multi-channeled image
dst = cv2.merge(rgba, 4)
  
# Writing and saving to a new image
cv2.imwrite("gfg_white.png", dst)
