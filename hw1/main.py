import os
import numpy as np
from scipy.io import loadmat
from typing import List, Dict
from scipy.fft import fft2, ifft2, fftfreq
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

MOTION_PATH = "100_motion_paths.mat"
IMAGE_PATH = "DIPSourceHW1.jpg"
NUM_IMAGES = None


def calculate_point_spread_function(motion_paths: Dict[str, np.ndarray], size_response_box: float = 1,
                                    plot: bool = False, save: bool = True):
    psf_directory = "psf_matrices"
    if save and not os.path.exists(psf_directory):
        os.makedirs(psf_directory)

    motion_x, motion_y = motion_paths["X"][:NUM_IMAGES], motion_paths["Y"][:NUM_IMAGES]
    psf_matrices = []

    min_value = int(min(np.min(motion_x), np.min(motion_y)))
    max_value = int(max(np.max(motion_x), np.max(motion_y)))

    x_bins = np.arange(min_value, max_value + 1, dtype=float)
    y_bins = np.arange(min_value, max_value + 1, dtype=float)

    # Center the pixels and resize to the response box size
    x_bins = (x_bins - 1 / 2) * size_response_box
    y_bins = (y_bins - 1 / 2) * size_response_box

    x_range = [x_bins[0], x_bins[-1]]
    y_range = [y_bins[0], y_bins[-1]]

    for index, (x, y) in enumerate(zip(motion_x, motion_y)):
        if plot:
            # Plot the trajectory
            plt.scatter([x[0], x[-1]], [y[0], y[-1]], c=["green", "red"])
            plt.plot(x, y)
            plt.xlim(x_range)
            plt.ylim(y_range)
            plt.show()

        # Calculate & plot the PSF
        histogram, *_ = np.histogram2d(y, x, bins=(y_bins, x_bins), density=True)
        psf_matrices.append(histogram)

        if plot:
            plt.imshow(histogram, cmap="gray", origin="lower", extent=[*x_range, *y_range])
            plt.show()
        if save:
            np.save(os.path.join(psf_directory, f"matrix_{index}.npy"), histogram)

    return psf_matrices


def blur_image(image: np.ndarray, psf_matrices: List[np.ndarray], plot: bool = False, save: bool = True):
    images_directory = "blurred_images"
    if save and not os.path.exists(images_directory):
        os.makedirs(images_directory)

    blurred_images = []
    for index, psf_matrix in enumerate(psf_matrices):
        blurred = convolve2d(image, psf_matrix, mode="same")
        blurred_images.append(blurred)

        if plot:
            plt.imshow(blurred, cmap="gray")
            plt.show()
        if save:
            plt.imsave(os.path.join(images_directory, f"image_{index}.png"), blurred, cmap="gray")

    return blurred_images


def restore_image(blurred_images: List[np.ndarray], p: float, plot: bool = False, save: bool = True):
    image_directory = "restored_images"
    if save and not os.path.exists(image_directory):
        os.makedirs(image_directory)

    # Restore the image
    fourier_images = np.stack([fft2(image) for image in blurred_images], axis=0)

    x_freq, y_freq = (fftfreq(dim_size) for dim_size in blurred_images[0].shape)
    frequencies = np.stack(np.meshgrid(y_freq, x_freq), axis=0)
    frequency_weights = np.prod(np.sinc(frequencies), axis=0)

    restored_images = []
    cumulative_weights = np.zeros_like(blurred_images[0])
    cumulative_fourier_image = np.zeros_like(blurred_images[0], dtype=complex)

    for index, fourier_image in enumerate(fourier_images):
        if p == float("inf"):
            # as p -> \infty, the average becomes a maximum
            weights = np.abs(fourier_image)

            mask = weights > cumulative_weights
            cumulative_weights[mask] = weights[mask]
            cumulative_fourier_image[mask] = fourier_image[mask]

            fourier_image = cumulative_fourier_image / frequency_weights
        else:
            weights = np.abs(fourier_image) ** p
            weights /= np.max(weights)

            cumulative_weights += weights
            cumulative_fourier_image += fourier_image * weights

            fourier_image = cumulative_fourier_image / cumulative_weights
            fourier_image /= frequency_weights

        restored_image = ifft2(fourier_image).real
        restored_images.append(restored_image)

        if plot:
            plt.imshow(restored_image, cmap="gray")
            plt.show()
        if save:
            plt.imsave(os.path.join(image_directory, f"image_{index}.png"), restored_image, cmap="gray")

    return restored_images


def calculate_psnr(restored_images: List[np.ndarray], original_image: np.ndarray, plot: bool = True, save: bool = True):
    psnr_values = []
    for image in restored_images:
        mse = np.mean((original_image - image) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse)
        psnr_values.append(psnr)

    if plot:
        plt.plot(psnr_values)
        plt.title("PSNR of the restored images")
        plt.xlabel("PSNR value")
        plt.ylabel("Number of blurred image used")
        plt.show()

    if save:
        np.savetxt("psnr_values.csv", psnr_values)

    return psnr_values


def main():
    motion_paths = loadmat(MOTION_PATH, variable_names=["X", "Y"])

    psf_matrices = calculate_point_spread_function(motion_paths, size_response_box=1, plot=False)

    original_image = plt.imread(IMAGE_PATH)[..., 0]
    blurred_images = blur_image(original_image, psf_matrices, plot=False)

    restore_images = restore_image(blurred_images, p=15, plot=False)
    calculate_psnr(restore_images, original_image, plot=True)


if __name__ == '__main__':
    main()
