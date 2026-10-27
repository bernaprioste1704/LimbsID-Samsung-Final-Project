import cv2
import numpy as np

from typing import Tuple

def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (0, 0, 0)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0] if new_shape[0] > new_size[0] else 0
    delta_h = new_shape[1] - new_size[1] if new_shape[1] > new_size[1] else 0
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def resize(image: np.array, new_shape: Tuple[int, int]) -> np.array:
    """Resizes an image to a given shape.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
    Returns:
        image: Resized image.
    """
    image = cv2.resize(image, new_shape)
    return image

  
def rotate(image: np.array, angle: int) -> np.array:
    """Rotates an image by a given angle.
    Params:
        image: Image to be rotated.
        angle: Angle of rotation in degrees.
    Returns:
        image: Rotated image.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def apply_brightness(image: np.array, value: int) -> np.array:
    """Brightens an image by a given value.
    Params:
        image: Image to be brightened.
        value: Value to brighten the image by.
    Returns:
        image: Brightened image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_random_noise(image, intensity=25):
    noisy_image = image.copy()
    noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)
    noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


image = cv2.imread("./EDA/000003072.jpg")
resized_image = resize(image, (224, 224))

padded_image = resize_with_pad(image, (224, 224))
rotated_image = rotate(padded_image, 10)
deg90_image = rotate(padded_image, 90)
brighten_image = resize_with_pad(apply_brightness(image, 50), (224, 224))
noisy_image = resize_with_pad(add_gaussian_noise(image, mean=0, std=25), (224, 224))
random_noisy_image = resize_with_pad(add_random_noise(image, intensity=25), (224, 224))
blurred_image = resize_with_pad(add_blur(image, kernel_size=11), (224, 224))
    
cv2.imshow("Original image", image)
cv2.imshow("Resized image", resized_image)
cv2.imshow("Padded image", padded_image)
cv2.imshow("Rotated image", rotated_image)
cv2.imshow("90 degree rotated image", deg90_image)
cv2.imshow("Brightened image", brighten_image)
cv2.imshow("Noisy image", noisy_image)
cv2.imshow("Random noisy image", random_noisy_image)
cv2.imshow("Blurred image", blurred_image)
cv2.waitKey()







#https://docs.roboflow.com/datasets/image-augmentation