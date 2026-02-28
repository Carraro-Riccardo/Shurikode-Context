from PIL import Image, ImageDraw
from typing import Literal
import numpy as np
import torch


class CanvaBit:

    def __init__(self, diagonal: int, size: int = 1, padding: bool = True):
        """
        Creates a CanvaBit object, on which to draw bits in order to form two-dimensional codes.

        :param diagonal: Height and width of the canva, in term of bits.
        :param size: Size of each bit (and as a consequence of the resulting image).
        :param padding: If to include the white border around the canva or not.
        """
        assert padding == 1 or padding == 0, "The padding must be 0 or 1."
        self.__matrix_diagonal = diagonal
        self.__cell_size = int(10 * size)
        self.__image_padding = padding * self.__cell_size
        canva_diagonal = diagonal * self.__cell_size + self.__image_padding * 2
        self.__image = Image.new("RGB", (canva_diagonal, canva_diagonal), "white")
        self.__drawable_image = ImageDraw.Draw(self.__image)

    def __draw_bit(self, value: bool, x: int, y: int) -> "CanvaBit":
        """
        Draws a black pixel if value is `True` otherwise it draws a white pixel, in the specified matrix coordinates.

        :param value: The boolean value of the bit to be represented in the specified coordinates.
        :param x: The row index of the matrix.
        :param y: The column index of the matrix.
        """
        assert (
            x >= 0
            and x < self.__matrix_diagonal
            and y >= 0
            and y < self.__matrix_diagonal
        ), f"The x and y coordinates must be in the range 0 to {self.__matrix_diagonal} for this canva."
        color = "black" if value else "white"
        canva_x = y  # the input x is the row index for the matrix, but the x on the canva means the horizontal axis
        canva_y = x  # the input y is the column index for the matrix, but the y on the canva means the vertical axis
        x0 = self.__image_padding + canva_x * self.__cell_size
        y0 = self.__image_padding + canva_y * self.__cell_size
        x1 = x0 + self.__cell_size
        y1 = y0 + self.__cell_size
        self.__drawable_image.rectangle([x0, y0, x1, y1], fill=color)
        return self

    def from_2D_array(self, a: np.ndarray) -> "CanvaBit":
        """
        Given a 2D NumPy array, it draws the specified two-dimensional matrix code.

        :param a: The two-dimensional matrix code to draw.
        """
        assert len(a.shape) == 2, "The provided array must be with two dimensions."
        rows, cols = a.shape
        for i in range(rows):
            for j in range(cols):
                self.__draw_bit(bool(a[i][j]), i, j)
        return self

    def save(self, path: str):
        """
        Saves image of the two-dimensional code, created through the interaction with the `CanvaBit` object, in a
        specified path.

        :param path: The path on which to save the two-dimensional matrix code as an image.
        """
        self.__image.save(path)

    def get_PIL_image(self):
        """
        Returns the image of the two-dimensional code, created through the interaction with the `CanvaBit` object, as a
        PIL Image.
        """
        return self.__image


DeviceType = Literal["cpu", "cuda", "mps"]

ModelType = Literal["r18", "r34", "r50"]

ModelSizes = Literal["S", "M", "L"]


def find_device() -> DeviceType:
    """
    Returns the available device of the machine.

    :return: A value between "cuda", "mps", "cpu", under the custom `DeviceType` type annotation.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
