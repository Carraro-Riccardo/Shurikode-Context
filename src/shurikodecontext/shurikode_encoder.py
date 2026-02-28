from shurikodecontext.utils import CanvaBit
import numpy as np


class ShurikodeEncoder:
    __diagonal = 10

    __positions = [
        # Corresponding positions of each bit array index in the 2D matrix grid
        # + if to invert the pixel in that specific position or not
        #  External        Medium        Internal
        [(0, 1, False), (1, 2, True), (2, 4, False)],
        [(0, 2, False), (1, 3, True), (2, 5, False)],
        [(0, 3, False), (1, 4, True), (2, 6, False)],
        [(0, 4, False), (1, 5, True), (2, 7, False)],
        [(0, 5, False), (1, 6, True), (3, 4, False)],
        [(0, 6, False), (1, 7, True), (3, 5, False)],
        [(0, 7, False), (1, 8, True), (3, 6, False)],
        [(0, 8, False), (2, 3, True), (4, 4, False)],
    ]

    def __init__(self, size: int = 1) -> None:
        """
        Creates a Shurikode encoder.

        :param size: The size of each pixel in the code (as a consequence the size of the code as a whole).
        """
        self.__canvabit = CanvaBit(self.__diagonal, size, padding=True)
        self.__matrix = np.zeros([self.__diagonal, self.__diagonal])

    def encode(self, value: int) -> "ShurikodeEncoder":
        """
        Given the value to be encoded, it returns a `shurikode_encoder` object that contains the newly created shurikode
        code.

        :param value: The value to be converted into a Shurikode code.
        """
        assert value < 256, "The value must be less then 256."
        assert value >= 0, "The value must be greater or equal to 0."

        bit_array = self.__int_to_reversed_bit_array(value)
        ed_bit = bool(1 - np.sum(bit_array) % 2)  # Calculating the parity bit

        for idx, bit in enumerate(bit_array):  # Drawing the matrix
            for x, y, to_invert in self.__positions[idx]:
                self.__draw_specular(bit != to_invert, x, y)
        self.__draw_specular(ed_bit, 0, 0)

        return self

    def __draw_specular(self, value: bool, x: int, y: int):
        """
        Utility function that given coordinates, draws specularly on the four sides of the canva the boolean value.

        :param value: The boolean value to draw.
        :param x: The x coordinate of the code matrix.
        :param y: The y coordinate of the code matrix.
        """
        diagonal = self.__diagonal - 1
        self.__matrix[x, y] = value
        self.__matrix[y, diagonal - x] = value
        self.__matrix[diagonal - x, diagonal - y] = value
        self.__matrix[diagonal - y, x] = value

    def save(self, path: str):
        """
        Given a path, saves the Shurikode code, that's stored inside the object, to that specific path.

        :parmam path: The path that specifies where to save the Shurikode code created.
        """
        self.__canvabit.from_2D_array(self.__matrix).save(path)

    def get_PIL_image(self):
        """
        Returns the Shurikode code, that's stored inside the object, as a PIL Image.
        """
        return self.__canvabit.from_2D_array(self.__matrix).get_PIL_image()

    @staticmethod
    def __int_to_reversed_bit_array(value: int) -> np.ndarray:
        """
        Utility function that encodes the input value into a binary NumPy array, and returns it reversed.
        """
        bit_array = [0] * 8
        i = -1
        while value:  # Calculating the bit representation of the input value
            bit_array[i] = bool(1 & value)
            value = value >> 1
            i -= 1

        bit_array.reverse()  # Reversing to follow the shurikode clockwise arrangement
        return np.array(bit_array)
