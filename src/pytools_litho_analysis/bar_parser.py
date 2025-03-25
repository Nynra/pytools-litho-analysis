from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
import cv2
import datetime as dt


class AbstractBarParser(ABC):

    @abstractmethod
    def __init__(self, image: np.ndarray):
        pass

    @abstractmethod
    def get_bar(self) -> np.ndarray:
        """Get the bar from the image."""
        pass

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """Get the image without the bar."""
        pass

    @abstractmethod
    def get_image_size(self) -> np.ndarray:
        """Get the size of the image without the bar."""
        pass

    @abstractmethod
    def get_bar_coord(self) -> tuple[int, int, int, int]:
        """Get the coordinates of the bar in the original image."""
        pass

    @abstractmethod
    def get_image_coords(self) -> tuple[int, int, int, int]:
        """Get the coordinates of the image without the bar in the original image.

        This is the inverse of get_bar_coord.
        """
        pass

    @abstractmethod
    def get_image_date(self) -> dt.date:
        """Get the scanning date from the image."""
        pass

    @abstractmethod
    def get_image_time(self) -> dt.time:
        """Get the scanning time from the image."""
        pass

    @abstractmethod
    def get_image_datetime(self) -> dt.datetime:
        """Get the scanning date and time from the image."""
        pass


class GenericBarParser(AbstractBarParser):
    # The bar is given in xywh format, as we are working with opencv images
    # the origin is the top left corner of the image and positive y is down
    # To make locating the bar easier though for the BOTTOM left of the bar is
    # given and the height of the bar is substracted from the y coordinate
    # w accepts the none value to indicate the bar spans the whole image

    # This bar starts in the bottom left corner, spans the whole width of the image
    # and is 50 pixels high
    bar_coord = [0, 0, None, 50]

    def __init__(self, image: np.ndarray):
        self.image = image

    def get_bar(self) -> np.ndarray:
        return self.image[
            self.bar_coord[1] - self.bar_coord[3] : self.bar_coord[1],
            self.bar_coord[0] : self.bar_coord[0] + self.bar_coord[2],
        ]

    def get_image(self) -> np.ndarray:
        return self.image[
            : self.bar_coord[1] - self.bar_coord[3],
            :,
        ]

    def get_image_size(self) -> np.ndarray:
        return self.get_image().shape

    def get_bar_coord(self) -> tuple[int, int, int, int]:
        return tuple(self.bar_coord)

    def get_image_coords(self) -> tuple[int, int, int, int]:
        return (
            0,
            self.bar_coord[1] - self.bar_coord[3],
            self.image.shape[1],
            self.image.shape[0],
        )

    def get_image_date(self) -> dt.date:
        raise NotImplementedError("This method is not implemented yet.")

    def get_image_time(self) -> dt.time:
        raise NotImplementedError("This method is not implemented yet.")

    def get_image_datetime(self) -> dt.datetime:
        raise NotImplementedError("This method is not implemented yet.")
