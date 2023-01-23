

import unittest
from numpy import inf
import numpy as np
import pandas as pd
from drtools.data_science.test_data_handle import (
    get_labels_from_bin_interval,
    binning_numerical_variable,
    prepare_bins
)


class Test_binning_numerical_variable(unittest.TestCase):

    def test_ordinary(self):
        observed = (
            pd.DataFrame(np.random.randint(5000, size=10000), columns=['test']),
            'test',
            [0, 10, 20, 50, 100, 200, 1000, inf]
        )
        response = binning_numerical_variable(*observed)
        response['test_analyses'] = response.apply(
            lambda x: \
                float(x['binning_test'].split('_')[0]) \
                <= x['test'] < \
                float(x['binning_test'].split('_')[1]),
            axis=1
        )
        correctness = response['test_analyses'].eq(True).sum()
        self.assertTrue('binning_test' in response.columns)
        self.assertEqual(correctness, len(response))


class Test_prepare_bins(unittest.TestCase):

    def test_ordinary(self):
        observed = (0, 100, 10)
        expected_values = np.linspace(0, 100, 11)
        expected_middle_points = np.arange(5, 96, 10)
        expected_labels = get_labels_from_bin_interval(expected_values)
        values, middle_points, labels = prepare_bins(*observed)
        self.assertTrue(np.array_equal(np.array(values), expected_values))
        self.assertTrue(np.array_equal(np.array(middle_points), expected_middle_points))
        self.assertEqual(labels, expected_labels)
        
    def test_ordinary_include_upper(self):
        observed = (0, 100, 10, True)
        expected_values = np.linspace(0, 100, 11)
        expected_values[-1] = expected_values[-1] * (1 + 0.001)
        expected_middle_points = np.arange(5, 96, 10).astype(float)
        expected_middle_points[-1] = (expected_values[-2] + expected_values[-1]) / 2
        expected_labels = get_labels_from_bin_interval(expected_values)
        values, middle_points, labels = prepare_bins(*observed)
        self.assertTrue(np.array_equal(np.array(values), expected_values))
        self.assertTrue(np.array_equal(np.array(middle_points), expected_middle_points))
        self.assertEqual(labels, expected_labels)
        
    def test_complex(self):
        observed = (0, 100, [0, 1, 20, 50, 100])
        expected_values = [0, 1, 20, 50, 100]
        expected_middle_points = [.5, 10.5, 35., 75.]
        expected_labels = get_labels_from_bin_interval(expected_values)
        values, middle_points, labels = prepare_bins(*observed)
        self.assertTrue(np.array_equal(np.array(values), expected_values))
        self.assertTrue(np.array_equal(np.array(middle_points), expected_middle_points))
        self.assertEqual(labels, expected_labels)
        
    def test_complex_include_upper(self):
        observed = (0, 100, [0, 1, 20, 50, 100], True)
        expected_values = [0, 1, 20, 50, 100  * (1 + 0.001)]
        expected_middle_points = [.5, 10.5, 35., 75. + 100 * 0.001 / 2]
        expected_labels = get_labels_from_bin_interval(expected_values)
        values, middle_points, labels = prepare_bins(*observed)
        self.assertTrue(np.array_equal(np.array(values), expected_values))
        self.assertTrue(np.array_equal(np.array(middle_points), expected_middle_points))
        self.assertEqual(labels, expected_labels)


class Test_get_labels_from_bin_interval(unittest.TestCase):

    def test_ordinary(self):
        observed = [0, 10, 20]
        expected = ['0_10', '10_20']
        response = get_labels_from_bin_interval(observed)
        self.assertEqual(response, expected)
        
    def test_ordinary1(self):
        observed = [0, 10, 20.001]
        expected = ['0_10', '10_20.001']
        response = get_labels_from_bin_interval(observed)
        self.assertEqual(response, expected)
        
    def test_complex(self):
        observed = np.concatenate((np.arange(0, 100, 10), [inf]))
        expected = [
            '0.0_10.0', '10.0_20.0', '20.0_30.0', '30.0_40.0', '40.0_50.0',
            '50.0_60.0', '60.0_70.0', '70.0_80.0', '80.0_90.0', '90.0_inf',
        ]
        response = get_labels_from_bin_interval(observed)
        self.assertEqual(response, expected)
        

if __name__ == '__main__':
    unittest.main()