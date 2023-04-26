import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.data import shepp_logan_phantom
from scipy.ndimage import rotate as scipy_rotate

def radon_transform(image, theta_start=0, theta_end=180, number_of_angles=180):
    # Image dimensions
    height, width = image.shape

    # Diagonal of image -> denotes maximum value for s
    maximum_s_value = int(np.ceil(np.sqrt(height**2 + width**2)))

    # Pad the image to make it a square with side length equal to the maximum_s_value
    y_padding = (maximum_s_value - height) // 2
    x_padding = (maximum_s_value - width) // 2
    padded_image = np.pad(image, ((y_padding, y_padding), (x_padding, x_padding)), mode='constant')

    # Evenly spaced (number_of_angles) angles between theta_start and theta_end
    theta_range = np.linspace(theta_start, theta_end, number_of_angles, endpoint=False)

    # Initialize sinogram with zeros
    sinogram = np.zeros((maximum_s_value, number_of_angles))

    # Loop over theta_range and calculate Radon transform for each theta
    for i, theta in enumerate(theta_range):
        # Rotate padded image by current theta
        rotated_image = scipy_rotate(padded_image, theta, reshape=False, order=1, mode='constant', cval=0)

        # Build line integrals of vertical lines along rotated image by summing up intensity values
        sinogram[:, i] = np.sum(rotated_image, axis=0)

    # Remove padding to obtain original height
    cropped_sinogram = sinogram[y_padding:y_padding+height, :]

    return cropped_sinogram

# Load Shepp-Logan phantom
image = shepp_logan_phantom()

# For animation, spacing between Theta values will always be 1
max_theta = 360

# Initialize the sinogram (for animation) with zeros
sinogram = np.zeros((image.shape[0], max_theta))

# Create figure and two axes for displaying rotated image and sinogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

def update(frame):
    ax1.clear()
    ax2.clear()

    # Calculate current theta based on frame number
    theta = frame % max_theta

    # If frame number is multiple of number_of_angles, reset the sinogram to zeros
    if frame % max_theta == 0:
        sinogram.fill(0)

    # Calculate Radon transform for current theta
    current_sinogram_column = radon_transform(image, theta_start=theta, theta_end=theta+1, number_of_angles=1)

    # Update sinogram (for animation) with calculated column
    sinogram[:, theta] = current_sinogram_column[:, 0]

    # Rotate input image by current theta
    rotated_image = scipy_rotate(image, theta, reshape=False, order=1, mode='constant', cval=0)

    # Display input image
    ax1.imshow(rotated_image, cmap='gray')
    ax1.set_title('Input image')
    ax1.axis('off')

    # Display sinogram
    ax2.imshow(sinogram, cmap='gray', aspect='auto', extent=(0, max_theta, 0, sinogram.shape[0]))
    ax2.set_title('Sinogram')
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel(r'$s$')
    ax2.axis('on')

animation = FuncAnimation(fig, update, frames=range(max_theta), interval=50, repeat=True)
plt.show()
