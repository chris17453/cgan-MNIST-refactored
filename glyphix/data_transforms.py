import torch
from torchvision import transforms as T
from torchvision.datasets import EMNIST,MNIST
from PIL import Image
import numpy as np
from PIL import ImageFilter, Image
import cv2



def remove_noise_and_blur(image):
    """
    Apply morphological opening to remove noise and then apply Gaussian blur.
    Converts a PyTorch tensor to a PIL Image, applies processing, and then converts back.
    """
    # Ensure image is a NumPy array in the correct format
    if isinstance(image, torch.Tensor):
        image = image.squeeze().numpy()  # Remove any extra dimensions
    image = image.astype(np.uint8)  # Ensure the type is uint8 for PIL compatibility
    
    # Apply morphological opening
    kernel = np.ones((2 ,2), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Convert to PIL Image for Gaussian blur
    image_pil = Image.fromarray(opening)
    image_blurred = image_pil.filter(ImageFilter.GaussianBlur(radius=1))  # Adjust radius as needed
    
    return image_blurred

class GaussianBlurTransform(object):
    def __init__(self, kernel_size=(2, 2), sigmaX=0):
        """
        Initializes the GaussianBlurTransform class.

        Parameters:
        - kernel_size: Tuple of 2 integers, specifying the width and height of the Gaussian kernel. Defaults to (5, 5).
        - sigmaX: Gaussian kernel standard deviation in the X direction. If 0, it is calculated from the kernel size.
        """
        self.kernel_size = kernel_size
        self.sigmaX = sigmaX

    def __call__(self, img):
        """
        Applies Gaussian Blur to the input image.

        Parameters:
        - img: A PIL.Image object.

        Returns:
        - A PIL.Image object with Gaussian Blur applied.
        """
        img_array = np.array(img)
        blurred_img_array = cv2.GaussianBlur(img_array, self.kernel_size, self.sigmaX)
        return Image.fromarray(blurred_img_array)
    
class HistogramEqualizationTransform(object):
    def __call__(self, img):
        img_array = np.array(img)
        if len(img_array.shape) == 2:  # Grayscale image
            equalized_img = cv2.equalizeHist(img_array)
        else:  # Color image
            img_y_cr_cb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
            img_y_cr_cb[:, :, 0] = cv2.equalizeHist(img_y_cr_cb[:, :, 0])
            equalized_img = cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2RGB)
        return Image.fromarray(equalized_img)
    
class CLAHETransform(object):
    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        img_array = np.array(img)
        if len(img_array.shape) == 2:  # Grayscale image
            clahe_img = clahe.apply(img_array)
        else:  # Color image
            img_y_cr_cb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
            img_y_cr_cb[:, :, 0] = clahe.apply(img_y_cr_cb[:, :, 0])
            clahe_img = cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2RGB)
        return Image.fromarray(clahe_img)


class AdjustBrightnessTransform(object):
    def __init__(self, target_brightness):
        self.target_brightness = target_brightness

    def __call__(self, img):
        current_brightness = img.mean().item()
        adjustment_factor = self.target_brightness / current_brightness
        adjusted_img = torch.clamp(img * adjustment_factor, 0, 1)  # Ensure values are still in [0, 1]
        return adjusted_img
    
class ConvertToBlackAndWhite(object):
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, x):
        # Convert to black and white
        x = torch.where(x > self.threshold/255, torch.ones_like(x), torch.zeros_like(x))
        return x

class ApplyGaussianBlur(object):
    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, x):
        # Apply Gaussian Blur as a form of anti-aliasing
        x = x.filter(ImageFilter.GaussianBlur(self.radius))
        return x
    

class CombineWithOriginal(object):
    def __init__(self, original):
        """
        Initializes the class with the original binary image.
        """
        self.original = original

    def __call__(self, blurred):
        """
        Combines the blurred image with the original binary image,
        effectively overlaying the original white components on top.
        """
        # Assuming original and blurred are PIL Images, and original is binary (black and white)
        # Convert both images to arrays for processing
        original_array = np.array(self.original)
        blurred_array = np.array(blurred)
        
        # Overlay the original white components (255 in binary image) on top of the blurred image
        combined_array = np.where(original_array == 255, original_array, blurred_array)
        
        # Convert back to PIL Image
        combined_image = Image.fromarray(combined_array.astype('uint8'))
        return combined_image
class ConvertAndCombineWithBlur(object):
    def __init__(self, threshold=0.5, radius=1):
        self.threshold = threshold
        self.radius = radius

    def __call__(self, img):
        # Convert to binary (black and white)
        bw_img = img.convert('L').point(lambda x: 255 if x > self.threshold*255 else 0, mode='1')
        
        # Apply Gaussian blur to the original grayscale image
        blurred_img = img.filter(ImageFilter.GaussianBlur(self.radius))
        
        # Combine the original binary image with the blurred grayscale image
        combined_img = Image.composite(bw_img.convert('L'), blurred_img, bw_img)
        
        return combined_img


def remove_noise(img):
    """
    Apply a morphological opening-like operation (erosion followed by dilation)
    to remove small noise dots. This is simulated using MinFilter and MaxFilter.
    """
    # Simulate erosion to remove small dots/noise
    eroded_img = img.filter(ImageFilter.MinFilter(3))  # Using a small filter size
    
    # Simulate dilation to restore the shape after erosion
    cleaned_img = eroded_img.filter(ImageFilter.MaxFilter(1))
    
    return cleaned_img

def conditional_average(img):
    """
    Adjusts a pixel to the average of its 8 surrounding pixels only if
    its value is less than half the average of those surrounding pixels.
    Assumes img is a PIL Image in grayscale.
    """
    # Convert PIL Image to a NumPy array
    img_array = np.array(img, dtype=np.float32)
    
    # Calculate the average of surrounding pixels
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]]) / 8.0  # Excluding the center pixel for averaging
    
    # Apply convolution to get the surrounding average
    surrounding_avg = cv2.filter2D(img_array, -1, kernel)
    
    # Create a mask where pixel values are less than half the surrounding average
    mask = img_array < (surrounding_avg / 8.0)
    
    # Apply conditional average
    img_array[mask] = surrounding_avg[mask]
    
    # Convert back to PIL Image and ensure values are within valid range
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def anti_aliasing(img):
    """
    Applies a simple anti-aliasing effect by averaging pixel values with their neighbors.
    
    Parameters:
    - img: A PIL.Image object in grayscale mode.
    
    Returns:
    - A new PIL.Image object with the anti-aliasing effect applied.
    """
    # Convert the image to a NumPy array for processing
    img_array = np.array(img, dtype=np.float32)
    
    # Create an empty array to hold the anti-aliased image
    aa_img_array = np.zeros_like(img_array)
    
    # Define the neighborhood range for each pixel (8-connected neighborhood)
    for y in range(1, img_array.shape[0] - 1):
        for x in range(1, img_array.shape[1] - 1):
            # Calculate the average value of the pixel's neighborhood, including the pixel itself
            neighborhood = img_array[y-1:y+2, x-1:x+2]
            aa_img_array[y, x] = np.mean(neighborhood)
    
    # Handle border pixels by copying them from the original image (optional enhancement)
    aa_img_array[0, :] = img_array[0, :]
    aa_img_array[-1, :] = img_array[-1, :]
    aa_img_array[:, 0] = img_array[:, 0]
    aa_img_array[:, -1] = img_array[:, -1]
    
    # Convert the processed array back to a PIL image
    aa_img = Image.fromarray(np.uint8(aa_img_array))
    return aa_img
