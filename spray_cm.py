import cv2

original=cv2.imread('/home/mglopes/Documents/GitHub/Spray-concentration-map/gamacorrected.png', 1)
im_gray = cv2.imread('/home/mglopes/Documents/GitHub/Spray-concentration-map/gamacorrected.png', cv2.IMREAD_GRAYSCALE)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
cv2.imshow('Original image', im_gray)
cv2.waitKey(0)
cv2.imshow('edgeDetection2', im_color)
cv2.waitKey(0)

#Desenvolvimento da remoção do fundo

# Applying thresholding technique
_, alpha = cv2.threshold(im_gray, 3, 255, cv2.THRESH_BINARY)
  
# Using cv2.split() to split channels 
# of coloured image
b, g, r = cv2.split(original)
  
# Making list of Red, Green, Blue
# Channels and alpha
rgba = [b, g, r, alpha]
  
# Using cv2.merge() to merge rgba
# into a coloured/multi-channeled image
dst = cv2.merge(rgba, 4)
  
# Writing and saving to a new image
cv2.imwrite('gfg_white.png', dst)

