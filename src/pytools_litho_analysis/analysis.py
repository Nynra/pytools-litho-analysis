import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import os
from typing import Tuple
from .step_functions import fit_block_step

# Not used here but still imported for easy access by the end user
# (these modules are now importable from this file)
from pytools_image_processing.segmentation import (
    get_bounding_rect,
)


def calculate_profile_psd(
    profile: np.ndarray, dx: float, use_window: bool = True, show_steps: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the Power Line Edge Roughness (LER) PSD.

    Calculate the Power Spectral Density (PSD) of the Line Edge Roughness (LER)
    of a line profile. The profile should be a condensed line profile of the
    line edge. The dx is the distance between the pixels in the profile.

    Parameters
    ----------
    profile : np.ndarray
        The condensed line profile.
    dx : float
        The distance between the pixels in the profile in nm.
    use_window : bool, optional
        If True, use a window function to reduce the edge effects. The default is True.
    show_steps : bool, optional
        If True, show the steps of the calculation. The default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The PSD of the profile and the frequency.
    """
    if not isinstance(profile, np.ndarray):
        raise ValueError(
            "profile should be a numpy array not type {}".format(type(profile))
        )
    if not isinstance(dx, (int, float)):
        raise ValueError("dx should be a number not type {}".format(type(dx)))
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    # Calculate the FFT of the profile
    fft = np.fft.fft(profile)
    freq = np.fft.fftfreq(len(profile), dx)

    # Calculate the PSD
    psd = np.abs(fft) ** 2

    # Use a window function to reduce edge effects
    if use_window:
        window = np.hanning(len(profile))
        psd = psd * window

    # Make the spectrum one sided
    psd = psd[: len(psd) // 2]
    freq = freq[: len(freq) // 2]

    # The ler is the square root of the integral of the PSD
    # over the frequency range
    roughness = np.sqrt(np.trapz(psd, freq))

    if show_steps:
        plt.plot(freq, psd)
        plt.xlabel("Wave number (1/nm)")
        plt.ylabel("PSD")

        # Use loglog scale
        plt.title("Roughness PSD: {:.2f}".format(roughness))
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    return psd, freq, roughness


def condense_line(img: np.ndarray, show_steps: bool = False) -> np.ndarray:
    """Condense the image to one line

    Use Riemann integration (summing pixels over the vertical axis) to
    condense the image to one line. The vertical axis is the longest
    axis in the image

    .. attention::

        The function expects a rotation corrected image. If the image is
        not corrected weird results will be returned.

    Parameters
    ----------
    img : np.ndarray
        The image to condense.

    Returns
    -------
    np.ndarray
        The condensed image.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("img should be a numpy array not type {}".format(type(img)))

    # Make sure the largest axis is on the horizontal otherwise transpose
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)

    # Calculate the riemann sum of the image over the horizontal
    # axis
    sums = -1 * np.sum(img, axis=0)

    # Normalize the sums
    sums = (sums - np.min(sums)) / (np.max(sums) - np.min(sums))

    if show_steps:
        # Plot the image and the condensed line
        plt.subplot(121)
        plt.imshow(img, cmap="gray")
        plt.title("Image")
        plt.subplot(122)
        plt.plot(sums)
        plt.title("Condensed line")
        plt.show()

    return sums


def extract_profiles(
    image: np.ndarray,
    crop_margin: float = 0.05,
    fitter: callable = fit_block_step,
    accepted_failure: float = 0.20,
    correct_offsets: bool = True,
    show_steps: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts the intensity profiles for each column in the image.

    The profiles are extracted by fitting a step function to each column in the
    image. In the case of a vertical line the image will be transposed to make
    sure the longest axis is on the horizontal. The left and right edge of the
    step are determined by the half height of the step.

    .. note::

        A custom function can be used long as the signature is the same as
        fit_block_step. Meaning, the function should take the data and return
        the left and right edge of the step. The function should also be able
        to handle the invert_step parameter. To allow for errors to occur when
        fitting the function should raise an OptimizeWarning if the fit is not
        correct.

    Parameters
    ----------
    image : np.ndarray
        The image to extract the profiles from.
    crop_margin : float, optional
        The area on the left and right side of the image that will be cropped
        off. This is to make sure that the ends of the lines in the image do not
        influence the fitting. The default is 0.05, meaning 5% on each side.
    fitter : callable, optional
        The function to fit the step function. The default is fit_block_step.
    accepted_failure : float, optional
        The maximum failure rate for the fitting function, by default 0.20.
    correct_offsets : bool, optional
        If the offsets should be corrected, by default True. This will make the
        average of the profiles zero.
    show_steps : bool, optional
        If the fitting steps should be shown, by default False.
    verbose : bool, optional
        If the warnings should be printed, by default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The top edge, bottom edge and width of profile for the object.

    Raises
    ------
    OptimizeWarning
        If the fitting fails too often.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "The image must be a numpy array not type {}".format(type(image))
        )
    if not len(image.shape) == 2:
        raise ValueError(
            "The image must be a grayscale image not shape {}".format(image.shape)
        )
    if not isinstance(crop_margin, float):
        raise TypeError(
            "The crop margin must be a float not type {}".format(type(crop_margin))
        )
    if 2 * crop_margin >= 1:
        raise ValueError(
            "The crop margin must be smaller than 0.5 not {}".format(crop_margin)
        )
    if not callable(fitter):
        raise TypeError(
            "The fitter must be a callable function not type {}".format(type(fitter))
        )
    if not isinstance(accepted_failure, float):
        raise TypeError(
            "The accepted failure must be a float not type {}".format(
                type(accepted_failure)
            )
        )
    if not isinstance(show_steps, bool):
        raise TypeError(
            "The show steps must be a boolean not type {}".format(type(show_steps))
        )
    if not isinstance(verbose, bool):
        raise TypeError(
            "The verbose must be a boolean not type {}".format(type(verbose))
        )

    # Crop the image to remove the ends
    crop = int(crop_margin * image.shape[1])
    image = image[:, crop:-crop]

    # Fit the step function
    left, right = [], []  # The left and right edge of the object
    error_count = 0  # The number of errors that occured during fitting

    # Make sure the longest axis is on the horizontal
    if image.shape[0] > image.shape[1]:
        image = cv2.transpose(image)

    # Make a fit for each column
    for i in range(0, image.shape[1], 1):
        profile = image[:, i].transpose()
        try:
            l, r = fitter(profile, invert_step=True, verbose=verbose)
        except (OptimizeWarning, RuntimeError) as e:
            # This is very bad practice but that is an issue for another time
            error_count += 1
            continue
        left.append(l)
        right.append(r)

    # Check if the failure rate was acceptable
    if error_count > accepted_failure * image.shape[1]:
        raise Exception("Too many errors occured during fitting. Aborting.")

    left_array = np.array(left)
    right_array = np.array(right)
    width_array = right_array - left_array

    # Correct the offsets so the average of the profiles is zero
    if correct_offsets:
        left_array = left_array - np.mean(left_array)
        right_array = right_array - np.mean(right_array)
        width_array = width_array - np.mean(width_array)

    if show_steps:
        # Show the image and the profiles in 4 seperate plots
        plt.subplot(221)
        plt.imshow(image, cmap="gray")
        plt.title("Image")
        plt.subplot(222)
        plt.plot(left_array)
        plt.title("Left edge")
        plt.subplot(223)
        plt.plot(right_array)
        plt.title("Right edge")
        plt.subplot(224)
        plt.plot(width_array)
        plt.title("Width")
        plt.show()

    return left_array, right_array, width_array


def find_object_edges(
    img: np.ndarray, show_steps: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Find the edges of the lithography line in a SEM image.

    The function expects only one line to be given in the image. The function
    will return the left and right edge of the line. To find the edges a block
    step is fitted to each column of the image. The left and right edge are
    determined by the half height of the step.

    Parameters
    ----------
    img : np.ndarray
        The image with the lithography line.
    show_steps : bool, optional
        If True, show the steps of the calculation. The default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The left and right edge of the line.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("img should be a numpy array not type {}".format(type(img)))
    if not len(img.shape) == 2:
        raise ValueError(
            "img should be a grayscale image not shape {}".format(img.shape)
        )
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if img.shape[0] < 10:
        raise ValueError("img should have at least 10 rows not {}".format(img.shape[0]))
    if img.shape[1] < 10:
        raise ValueError(
            "img should have at least 10 columns not {}".format(img.shape[1])
        )
    # Make sure the longest axis is on the horizontal
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)

    # Create empty edge arrays
    left_edge = np.zeros(img.shape[0])
    right_edge = np.zeros(img.shape[0])

    print(img.shape)

    for i in range(img.shape[0]):
        col = img[i, :]
        _, left, right = fit_block_step(col, show_steps=True)
        left_edge[i] = left
        right_edge[i] = right

    if show_steps:
        # Draw the edges in the image
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(img.shape[0]):
            img[i, int(left_edge[i])] = 255
            img[i, int(right_edge[i])] = 255
        plt.imshow(img)

        # # Also show the graphs
        # plt.subplot(121)
        # plt.plot(left_edge)
        # plt.title("Left edge")
        # plt.subplot(122)
        # plt.plot(right_edge)
        # plt.title("Right edge")
        plt.show()

    return left_edge, right_edge
