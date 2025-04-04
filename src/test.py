import unittest
from pico import Tensor

#class UnitTestSample(unittest.TestCase):

#    def test_one(self):
#        self.assertEqual('foo'.upper(), 'FOO')


class PicoTestCase(unittest.TestCase):

    def test_simple(self):
        simple_tensor = Tensor([1.2, 3.4], label='a')
        self.assertTrue(isinstance(simple_tensor, Tensor))



class PicoOpsTest(unittest.TestCase):

    def test_add_op(self):
        t1 = Tensor([1,1,2,4], label='a')
        t2 = Tensor([1,-4,-5,2], label='b')

        t3 = t1 + t2
        t3.backward()
        
        self.assertTrue(t3._prev != None)
        self.assertTrue(t1._backward != None)
        self.assertTrue(t2._backward != None)


if __name__ == "__main__":
    unittest.main()
