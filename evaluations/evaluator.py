import numpy as np
import lpips
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(reference_img, predicted_img):
    mse = np.mean((reference_img - predicted_img) ** 2)
    max_pixel_value = np.max(reference_img)
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr
def calculate_lpips(reference_img, predicted_img):
    # If the images have only one channel, duplicate it to get 3 channels
    predicted_img = predicted_img.unsqueeze(1)
    print(reference_img.shape)
    print(predicted_img.shape)
    if reference_img.shape[1] == 1:
        reference_img = reference_img.repeat(1, 3, 1, 1)
    if predicted_img.shape[1] == 1:
        predicted_img = predicted_img.repeat(1, 3, 1, 1)

    loss_fn = lpips.LPIPS(net='alex')
    return loss_fn(reference_img, predicted_img).mean()



def calculate_ssim(reference_img, predicted_img):
    predicted_img = np.expand_dims(predicted_img, axis=1)
    print(reference_img.shape)
    print(predicted_img.shape)

    # Calculate the data range
    data_range = reference_img.max() - reference_img.min()

    return np.mean(ssim(reference_img[:,0], predicted_img[:,0], data_range=data_range, multichannel=False))