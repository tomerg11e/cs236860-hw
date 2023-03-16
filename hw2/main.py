import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import BallTree
from typing import Tuple, Optional, Dict
from scipy import signal, special, linalg, fft

ALPHA = 3
SAVE_IMAGES = False


def create_filtered_images(image: np.ndarray, kernel_size) -> Dict[str, Dict[str, np.ndarray]]:
    pixels = np.linspace(-1, 1, kernel_size) * (kernel_size / 2)

    low_xx, low_yy = np.meshgrid(pixels, pixels)
    start = kernel_size // 2 % ALPHA
    high_xx, high_yy = low_xx[start::ALPHA, start::ALPHA], low_yy[start::ALPHA, start::ALPHA]

    sigma = 10

    # Gaussian Kernel
    normalization_factor = 1 / (2 * np.pi * sigma ** 2)

    low_res_gaussian_kernel = np.exp(- (low_xx ** 2 + low_yy ** 2) / (2.0 * sigma ** 2)) * normalization_factor

    high_res_gaussian_kernel = ALPHA ** 2 * np.exp(- ((ALPHA * high_xx) ** 2 + (ALPHA * high_yy) ** 2) / (2.0 * sigma ** 2)) * normalization_factor

    # Sinc Kernel
    low_res_sinc_kernel = np.sinc(low_xx / sigma ** 2) * np.sinc(low_yy / sigma ** 2)

    high_res_sinc_kernel = ALPHA ** 2 * np.sinc(ALPHA * high_xx / sigma ** 2) * np.sinc(ALPHA * high_yy / sigma ** 2)

    kernels = {
        "gaussian": {
            "low": low_res_gaussian_kernel,
            "high": high_res_gaussian_kernel,
        },
        "sinc": {
            "low": low_res_sinc_kernel,
            "high": high_res_sinc_kernel,
        }
    }

    # Normalize kernels
    kernels = {
        key: {
            resolution_name: (kernel - kernel.min()) / (kernel.max() - kernel.min())
            for resolution_name, kernel in type_kernels.items()
        }
        for key, type_kernels in kernels.items()
    }

    images = {
        key: {
            resolution_name: signal.convolve2d(image, kernel, mode="same", boundary="wrap")
            for resolution_name, kernel in type_kernels.items()
        }
        for key, type_kernels in kernels.items()
    }

    # Normalize images
    images = {
        key: {
            resolution_name: (image - image.min()) / (image.max() - image.min())
            for resolution_name, image in type_images.items()
        }
        for key, type_images in images.items()
    }

    for kernel_type, kernels_ in kernels.items():
        for resolution_name, kernel in kernels_.items():
            plt.imshow(kernel, cmap="gray")
            plt.title(f"{kernel_type} {resolution_name}")
            plt.show()

    for kernel_type, images_ in images.items():
        for resolution_name, image in images_.items():
            plt.imshow(image, cmap="gray")
            plt.title(f"{kernel_type} {resolution_name}")
            if SAVE_IMAGES:
                plt.savefig(f"{kernel_type}_{resolution_name}.png")
            else:
                plt.show()

    return images


def get_convolution_matrix(image_shape: Tuple[int, int], convolution_kernel: np.ndarray, remove_extra: bool = True) -> np.ndarray:
    kernel = convolution_kernel[::-1, ::-1]

    height, width = image_shape
    kernel_height, kernel_width = kernel.shape

    left_pad = (width - kernel_width) // 2
    right_pad = width - kernel_width - left_pad
    top_pad = (height - kernel_height) // 2
    bottom_pad = height - kernel_height - top_pad

    kernel = np.pad(kernel, ((top_pad, bottom_pad), (left_pad, right_pad)), mode="constant", constant_values=0)

    def rolling_block(values: np.ndarray, block_size: int, fill_value: int):
        return linalg.toeplitz(np.concatenate([values[block_size // 2::-1], np.full(block_size // 2, fill_value)]),
                               np.concatenate([values[block_size // 2:], np.full(block_size // 2, fill_value)]))

    blocks = [rolling_block(kernel_row, width, 0) for kernel_row in kernel]
    block_indices = rolling_block(np.arange(height), height, -1)

    convolution_matrix = sum(np.kron(block_indices == index, block) for index, block in enumerate(blocks))

    if remove_extra:
        # Remove extra rows
        convolution_matrix = convolution_matrix[width * top_pad: width * (height - bottom_pad)]

        # Remove extra columns
        indices = np.arange(convolution_matrix.shape[0]) % (left_pad + kernel_width + right_pad)
        convolution_matrix = convolution_matrix[(left_pad <= indices) & (indices < width - right_pad)]

    return convolution_matrix


def laplacian_squared(length):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    kernel_size = kernel.shape[0]
    kernel_radius = kernel_size // 2

    block_mask = linalg.toeplitz([1] * (length - kernel_radius) + [0] * kernel_radius)
    blocks = [linalg.circulant(np.roll(np.concatenate([row, [0] * (length - kernel_size)]), -kernel_radius)).T * block_mask for row in kernel]
    block_indices = [np.diag([1] * (length - abs(index)), -index) for index in reversed(range(-kernel_radius, kernel_radius + 1))]
    convolution_matrix = sum(np.kron(index, block) for block, index in zip(blocks, block_indices))

    return convolution_matrix.T @ convolution_matrix


def patchify(image: np.ndarray, patch_size: int):
    image = np.ascontiguousarray(image)
    height, width = image.shape
    shape = (height - patch_size + 1, width - patch_size + 1, patch_size, patch_size)
    strides = image.itemsize * np.array([width, 1, width, 1])
    return np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides).reshape((-1, patch_size, patch_size))


def get_patches_weights(low_res_patches, neighborhood_size, noise_std, convolution_matrices, kernel):
    reconstructed_parent_patches = convolution_matrices @ kernel.flatten()
    tree = BallTree(reconstructed_parent_patches, metric="euclidean", leaf_size=neighborhood_size)

    weights = np.zeros((len(low_res_patches), len(convolution_matrices)))
    distances, indices = tree.query(low_res_patches.reshape(low_res_patches.shape[0], -1), k=neighborhood_size, return_distance=True)
    weights[np.repeat(np.arange(len(weights)), neighborhood_size), indices.flatten()] = special.softmax(-distances ** 2 / (2 * noise_std ** 2), axis=-1).flatten()
    return weights


def estimate_kernel(image: np.ndarray, kernel_size: int,
                    patch_size: int, neighborhood_size: int,
                    noise_std: float, operator_std: float, max_iterations: int, num_patches: Optional[int] = None) -> np.ndarray:
    kernel = signal.unit_impulse((kernel_size, kernel_size), "mid")
    new_kernel = kernel

    low_res_patches = patchify(image, patch_size // ALPHA)
    high_res_patches = patchify(image, patch_size)

    if num_patches is not None:
        indices = np.random.choice(range(len(low_res_patches)), num_patches, replace=False)
        low_res_patches = low_res_patches[indices]
        indices = np.random.choice(range(len(high_res_patches)), 2 * num_patches, replace=False)
        high_res_patches = high_res_patches[indices]

    laplacian_sq = laplacian_squared(kernel_size)
    convolution_matrices = np.stack([get_convolution_matrix(kernel.shape, patch)[::ALPHA ** 2] for patch in high_res_patches])

    dummy_weights = np.empty((len(low_res_patches), len(high_res_patches)))
    weights_path, _ = np.einsum_path("ab, bdc, bde -> ce", dummy_weights, convolution_matrices, convolution_matrices, optimize="optimal")
    bias_path, _ = np.einsum_path("ab, bcd, ac -> d", dummy_weights, convolution_matrices, low_res_patches.reshape(low_res_patches.shape[0], -1), optimize="optimal")

    for iteration_num in range(max_iterations):
        print(iteration_num)
        weights = get_patches_weights(low_res_patches, neighborhood_size, noise_std, convolution_matrices, kernel)

        weighted_convolution_matrix = np.einsum("ab, bdc, bde -> ce", weights, convolution_matrices, convolution_matrices, optimize=weights_path)

        kernel_coefficient = laplacian_sq / operator_std ** 2 + weighted_convolution_matrix / noise_std ** 2

        kernel_bias = np.einsum("ab, bcd, ac -> d", weights, convolution_matrices, low_res_patches.reshape(low_res_patches.shape[0], -1), optimize=bias_path)

        new_kernel = np.linalg.inv(kernel_coefficient) @ kernel_bias.flatten()
        new_kernel = new_kernel.reshape(kernel_size, kernel_size)
        new_kernel = new_kernel / np.sum(np.abs(new_kernel))

        kernel = new_kernel

    return new_kernel


def wiener_filter(kernel: np.ndarray, image: np.ndarray, noise_std: float):
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape

    left_pad = (width - kernel_width + 1) // 2
    right_pad = width - kernel_width - left_pad
    top_pad = (height - kernel_height + 1) // 2
    bottom_pad = height - kernel_height - top_pad

    kernel = np.pad(kernel, ((top_pad, bottom_pad), (left_pad, right_pad)), mode="constant", constant_values=0)
    kernel = fft.ifftshift(kernel)
    fourier_kernel = fft.fft2(kernel)

    fourier_image = fft.fft2(image)
    reconstructed_image = np.abs(fft.ifft2(np.conj(fourier_kernel) / (np.abs(fourier_kernel) ** 2 + noise_std ** 2) * fourier_image))

    reconstructed_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())

    return reconstructed_image


def main():
    image = plt.imread("DIPSourceHW2.png")[..., 0]  # Grayscale needs only one channel (all are the same)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize to [0, 1]

    kernel_size = 15

    images = create_filtered_images(image, kernel_size=kernel_size)

    gaussian_kernel = estimate_kernel(images["gaussian"]["low"], kernel_size=kernel_size, patch_size=kernel_size,
                                      neighborhood_size=20, noise_std=10, operator_std=0.01, max_iterations=5, num_patches=500)
    sinc_kernel = estimate_kernel(images["sinc"]["low"], kernel_size=kernel_size, patch_size=kernel_size,
                                  neighborhood_size=20, noise_std=10, operator_std=0.1, max_iterations=5, num_patches=500)

    estimated_kernels = {
        "gaussian": gaussian_kernel,
        "sinc": sinc_kernel
    }

    for kernel_type, kernel in estimated_kernels.items():
        plt.imshow(kernel, cmap="gray")
        plt.title(f"Estimated kernel: {kernel_type}")
        plt.show()

    reconstructed_images = {
        kernel_type: {
            image_type: wiener_filter(kernel, image["low"], noise_std=1)
            for image_type, image in images.items()
        }
        for kernel_type, kernel in estimated_kernels.items()
    }

    for kernel_type, reconstructed in reconstructed_images.items():
        for image_type, image in reconstructed.items():
            mse = np.mean((images[image_type]["high"] - image) ** 2)
            psnr = -10 * np.log10(mse)

            plt.imshow(image, cmap="gray")
            plt.title(f"Predicted kernel: {kernel_type} on PSF: {image_type} - PSNR: {psnr:.2f}")
            if SAVE_IMAGES:
                plt.savefig(f"predicted_{kernel_type}_on_{image_type}.png")
            else:
                plt.show()


if __name__ == "__main__":
    main()
