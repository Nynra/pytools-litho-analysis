import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning


# def fit_gaus_step(
#     data: np.ndarray,
#     s: float = None,
#     dx: float = None,
#     bl: float = None,
#     br: float = None,
#     mu: float = None,
#     sigma_l: float = None,
#     sigma_r: float = None,
#     show_steps: bool = False,
# ) -> tuple[np.ndarray, float, float, float, float, float]:
#     """Fit the double gaussian function to the steps.

#     The function is defined as 2 equations:

#     P(x) = bl + (1 - bl) * np.exp(-0.5 * (x - mu)**2 / sigma_l**2) v x < mu
#     P(x) = br + (1 - br) * np.exp(-0.5 * (x - mu)**2 / sigma_r**2) v x >= mu

#     Where the function becomes a normal gaussian curve when bl and br approach zero.

#     To make the fitting more versatile the function is defined with a shift s and a dx
#     These are used as s * P(x - dx).

#     Parameters
#     ----------
#     data: np.ndarray
#         The data to fit.
#     s: float, optional
#         The initial guess height shift of the step. The default is None.
#     bl: float, optional
#         The initial guess baseline on the left side of the step. The default is None.
#     br: float, optional
#         The initial guess baseline on the right side of the step. The default is None.
#     mu: float, optional
#         The initial guess center of the step. The default is None.
#     sigma_l: float, optional
#         The initial guess standard deviation of the left side of the step. The default is None.
#     sigma_r: float, optional
#         The initial guess standard deviation of the right side of the step. The default is None.
#     show_steps: bool, optional
#         If True, show the steps of the fitting. The default is False.

#     Returns
#     -------
#     tuple[np.ndarray, float,float,float,float,float]
#         The fitted data and the parameters of the fit.
#     """
#     if not isinstance(data, np.ndarray):
#         raise ValueError("data should be a numpy array not type {}".format(type(data)))
#     if not isinstance(s, (int, float, type(None))):
#         raise ValueError("s should be a number not type {}".format(type(s)))
#     if not isinstance(dx, (int, float, type(None))):
#         raise ValueError("dx should be a number not type {}".format(type(dx)))
#     if not isinstance(bl, (int, float, type(None))):
#         raise ValueError("bl should be a number not type {}".format(type(bl)))
#     if not isinstance(br, (int, float, type(None))):
#         raise ValueError("br should be a number not type {}".format(type(br)))
#     if not isinstance(mu, (int, float, type(None))):
#         raise ValueError("mu should be a number not type {}".format(type(mu)))
#     if not isinstance(sigma_l, (int, float, type(None))):
#         raise ValueError("sigma_l should be a number not type {}".format(type(sigma_l)))
#     if not isinstance(sigma_r, (int, float, type(None))):
#         raise ValueError("sigma_r should be a number not type {}".format(type(sigma_r)))

#     # If no initial guess is given use some default values
#     if s is None:
#         s = np.max(data)
#     if dx is None:
#         dx = 0
#     if bl is None:
#         bl = 0.1
#     if br is None:
#         br = 0.1
#     if mu is None:
#         mu = len(data) // 2
#     if sigma_l is None:
#         sigma_l = 1
#     if sigma_r is None:
#         sigma_r = 1

#     def left_gaussian(x, bl, mu, sigma_l):
#         return bl + (1 - bl) * np.exp(-0.5 * (x - mu) ** 2 / sigma_l**2)

#     def right_gaussian(x, br, mu, sigma_r):
#         return br + (1 - br) * np.exp(-0.5 * (x - mu) ** 2 / sigma_r**2)

#     # Fit the curves seperately
#     popt_l, _ = curve_fit(left_gaussian, np.arange(mu), data[:mu], p0=[bl, mu, sigma_l])
#     popt_r, _ = curve_fit(
#         right_gaussian, np.arange(mu, len(data)), data[mu:], p0=[br, mu, sigma_r]
#     )

#     # Combine the curves
#     fit_l = left_gaussian(np.arange(mu), *popt_l)
#     fit_r = right_gaussian(np.arange(mu, len(data)), *popt_r)
#     fit = np.concatenate((fit_l, fit_r))

#     if show_steps:
#         plt.plot(data, label="Data")
#         plt.plot(fit, label="Fit")
#         plt.legend()
#         plt.show()

#     return fit, *popt_l, *popt_r


def fit_block_step(
    data: np.ndarray, invert_step: bool = False, show_steps: bool = False, verbose=False
) -> tuple[float, float]:
    """Fit a block step function to the data.

    While the function works most of the time it is not perfect. The function
    will raise a warning if the fit is not correct so make sure to catch the
    exceptions if you are analyzing larger sample sets

    Parameters
    ----------
    data : np.ndarray
        The data to fit.
    invert_step : bool, optional
        If True, use an inverted step response, this means the step will
        go from 1 to 0. The default is False.
    show_steps : bool, optional
        If True, show the steps of the fitting. The default is False.
    verbose : bool, optional
        If True, print the warnings. The default is False.

    Returns
    -------
    tuple[float, float]
        The half height on the left and right side of the step.

    Raises
    ------
    OptimizeWarning
        If the fit is not correct.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data should be a numpy array not type {}".format(type(data)))
    if not isinstance(invert_step, bool):
        raise ValueError(
            "invert_step should be a boolean not type {}".format(type(invert_step))
        )
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    def step_up(x, a, b, c):
        """Equation representing a step up using htan."""
        return 0.5 * a * (np.tanh((x - b) / c) + 1)

    def step_down(x, a, b, c):
        """Equation representing a step down using htan."""
        return 0.5 * a * (np.tanh((x - b) / c) - 1)

    # Fit the curves seperately
    if invert_step:
        # Normalize the data between 0 and 1 to make fitting easier
        data = 1 - (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    popt_left, cov_left = curve_fit(step_up, np.arange(len(data)), data, p0=[1, 10, 1])
    popt_right, cov_right = curve_fit(
        step_down, np.arange(len(data)), data, p0=[1, 10, 1]
    )

    # Fit the curves
    fit_left = step_up(np.arange(len(data)), *popt_left)
    fit_right = step_down(np.arange(len(data)), *popt_right)
    fit = fit_left + fit_right

    # Determine half the height on the left and right side
    left_min = fit_left.min()
    left_max = fit_left.max()
    half_height = (left_max - left_min) / 2

    # Find the index with the value closest to half height
    h_left = np.argmin(np.abs(fit_left - half_height))

    right_min = fit_right.min()
    right_max = fit_right.max()
    half_height = (right_max - right_min) / 2

    # Find the index with the value closest to half height
    h_right = np.argmin(np.abs(fit_right - half_height))

    if h_right == h_left:
        # borders are the same implicating there is no line
        e = OptimizeWarning(
            "The fit is not correct, left half height position is the same as the right side"
        )
        if verbose:
            print(e)
        raise e
    if h_right < 0 or h_left < 0:
        # Borders are outside the data
        e = OptimizeWarning(
            "The fit is not correct, left or right half height position is outside the data"
        )
        if verbose:
            print(e)
        raise e
    if h_left < int(0.025 * len(data)):
        # left half height is at the start of the data
        e = OptimizeWarning(
            "The fit is not correct, left half height position is at the start of the data"
        )
        if verbose:
            print(e)
        raise e
    if h_right > int(0.975 * len(data)):
        # right half height is at the end of the data
        e = OptimizeWarning(
            "The fit is not correct, right half height position is at the end of the data"
        )
        if verbose:
            print(e)
        raise e
    if h_right - h_left < 10:
        # The step is too small
        e = OptimizeWarning("The fit is not correct, the step is too small")
        if verbose:
            print(e)
        raise e
    if h_right < 0.5 * len(data):
        # The right border is on the left side of the image?
        e = OptimizeWarning(
            "The fit is not correct, the right border is on the left side of the image"
        )
        if verbose:
            print(e)
        raise e
    if h_left > 0.5 * len(data):
        # The left border is on the right side of the image?
        e = OptimizeWarning(
            "The fit is not correct, the left border is on the right side of the image"
        )
        if verbose:
            print(e)
        raise e

    # Normalize the fit
    fit = (fit - np.min(fit)) / (np.max(fit) - np.min(fit))

    if show_steps or h_left == 0:
        print(left_min, left_max, h_left)
        plt.plot(data, label="Data")
        plt.plot(fit, label="Fit")
        plt.legend()
        plt.show()

    return h_left, h_right
