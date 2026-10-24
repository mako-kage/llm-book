import unittest
from product import has_close_elements

class TestHasCloseElements(unittest.TestCase):
    def test_no_close_elements(self):
        self.assertFalse(has_close_elements([1.0, 2.0, 3.0], 0.5))

    def test_with_close_elements(self):
        self.assertTrue(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))

    def test_empty_and_single_element(self):
        self.assertFalse(has_close_elements([], 0.5))
        self.assertFalse(has_close_elements([1.0], 0.5))

    def test_close_elements_small_list(self):
        self.assertTrue(has_close_elements([1.0, 1.2], 0.3))

    def test_exact_threshold(self):
        self.assertFalse(has_close_elements([1.0, 1.5], 0.5))

if __name__ == '__main__':
    unittest.main()
