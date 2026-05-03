import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('images/img5.jpg', cv2.IMREAD_GRAYSCALE)

# Image dimensions
rows, cols = image.shape

# Compute Fourier transform
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)  # Shift zero frequency to center

# Create low-pass mask
crow, ccol = rows // 2, cols // 2  # Image center
radius = 30  # Filter radius
mask = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask, (ccol, crow), radius, (1, 1), thickness=-1)

# Apply filter
filtered_dft = dft_shift * mask

# Return to spatial domain
filtered_dft_shift = np.fft.ifftshift(filtered_dft)
img_back = cv2.idft(filtered_dft_shift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Display images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('DFT Magnitude Spectrum')
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Filtered Image')
plt.imshow(img_back, cmap='gray')
plt.axis('off')

plt.show()
