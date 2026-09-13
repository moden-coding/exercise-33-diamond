#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import numpy as np

from src.diamond import diamond


class TestDiamond(unittest.TestCase):

    def test_type(self):
        d = diamond(3)
        self.assertEqual(d.dtype, int, msg="Incorrect element type!")

    def test_shape(self):
        for n in range(1, 10):
            d = diamond(n)
            correct_shape = (2 * n - 1,) * 2
            self.assertEqual(
                d.shape, correct_shape,
                msg="Incorrect shape for call 'diamond(%i)'!" % n)

    def test_content(self):
        for n in range(1, 10):
            d = diamond(n)
            if n == 1:
                size = 1
            else:
                size = 4 * n - 4
            self.assertEqual(
                np.sum(d), size,
                msg="Incorrect number of 1s for call 'diamond(%i)'!" % n)

    def test_calls(self):
        with patch("src.diamond.np.eye", wraps=np.eye) as peye:
            with patch("src.diamond.np.concatenate", wraps=np.concatenate) as pconcatenate:
                diamond(3)
                peye.assert_called()
                pconcatenate.assert_called()

    def test_diamond_of_one_is_a_single_cell(self):
        d = diamond(1)
        np.testing.assert_array_equal(
            d, np.array([[1]]),
            err_msg="diamond(1) should be the 1x1 array [[1]].")


if __name__ == '__main__':
    unittest.main()
