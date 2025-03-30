import unittest
from pico import Tensor

#class UnitTestSample(unittest.TestCase):

#    def test_one(self):
#        self.assertEqual('foo'.upper(), 'FOO')


class PicoTestCase(unittest.TestCase):

    def test_simple(self):
        simple_tensor = Tensor([1.2, 3.4], label='a')
        self.assertTrue(isinstance(simple_tensor, Tensor))


if __name__ == "__main__":
    unittest.main()
