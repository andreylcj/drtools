

import unittest
from drtools.utils import (
    flatten, 
    re_flatten, 
    progress
)


class TestProgress(unittest.TestCase):

    def test_current_gt_total(self):
        observed = (51, 50)
        expected = 100
        response = progress(*observed)
        self.assertEqual(response, expected)
        
    def test_current_eq_total(self):
        observed = (50, 50)
        expected = 100 
        response = progress(*observed)
        self.assertEqual(response, expected)

    def test_current_lt_total(self):
        observed = (40, 50)
        expected = 80
        response = progress(*observed)
        self.assertEqual(response, expected)


class TestFlatten(unittest.TestCase):

    def test_ordinary(self):
        observed = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': {'f': 4, 'g': {'h': 5}}}}
        expected = {'a': 1, 'b': 2, 'c.d': 3, 'c.e.f': 4, 'c.e.g.h': 5}
        response = flatten(observed)
        self.assertEqual(response, expected)
        
    def test_with_list(self):
        observed = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': {'f': [1,2,3], 'g': {'h': 5}}}}
        expected = {'a': 1, 'b': 2, 'c.d': 3, 'c.e.f': [1,2,3], 'c.e.g.h': 5}
        response = flatten(observed)
        self.assertEqual(response, expected)

    def test_with_dict_inside_list(self):
        observed = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': {'f': [1,2,{'k1': 1, 'k2': {'k3': 123, 'k4': {'k5': 'asdasd'}}}], 'g': {'h': 5}}}}
        expected = {'a': 1, 'b': 2, 'c.d': 3, 'c.e.f': [
            1, 2,
            {'k1': 1, 'k2.k3': 123, 'k2.k4.k5': 'asdasd'}
        ], 'c.e.g.h': 5}
        response = flatten(observed)
        self.assertEqual(response, expected)
        
        
class TestReFlatten(unittest.TestCase):

    def test_ordinary1(self):
        observed = {'a': 1, 'b': 2, 'c.d': 3, 'c.f': 4}
        expected = {'a': 1, 'b': 2, 'c': {'d': 3, 'f': 4}}
        response = re_flatten(observed)
        self.assertEqual(response, expected)
        
    def test_ordinary2(self):
        observed = {'a': 1, 'b': 2, 'c.d': 3, 'c.e.f': 4, 'c.e.g.h': 5}
        expected = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': {'f': 4, 'g': {'h': 5}}}}
        response = re_flatten(observed)
        self.assertEqual(response, expected)
        
    def test_with_list(self):
        observed = {'a': 1, 'b': 2, 'c.d': 3, 'c.e.f': [1,2,3], 'c.e.g.h': 5}
        expected = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': {'f': [1,2,3], 'g': {'h': 5}}}}
        response = re_flatten(observed)
        self.assertEqual(response, expected)

    """ def test_with_dict_inside_list(self):
        observed = {'a': 1, 'b': 2, 'c.d': 3, 'c.e.f': [
            1, 2,
            {'k1': 1, 'k2.k3': 123, 'k2.k4.k5': 'asdasd'}
        ], 'c.e.g.h': 5}
        expected = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': {'f': [1,2,{'k1': 1, 'k2': {'k3': 123, 'k4': {'k5': 'asdasd'}}}], 'g': {'h': 5}}}}
        response = re_flatten(observed)
        self.assertEqual(response, expected) """

if __name__ == '__main__':
    unittest.main()