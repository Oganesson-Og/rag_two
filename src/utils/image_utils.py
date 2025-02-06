import cv2
import numpy as np
from PIL import Image

def convert_to_grayscale(image):
    """Convert image to grayscale"""
    if isinstance(image, Image.Image):
        return image.convert('L')
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resize_image(image, scale_factor):
    """Resize image by scale factor"""
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height))

def enhance_contrast(image):
    """Enhance image contrast"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def remove_noise(image, strength=7):
    """Remove image noise"""
    return cv2.fastNlMeansDenoising(image, None, strength)