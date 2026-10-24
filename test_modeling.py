import unittest
import numpy as np
import pandas as pd
from modeling import train_best_model, prepare_prediction

class ModelingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(23)
        n = 120
        cls.classification = pd.DataFrame({
            'Age': rng.integers(18, 80, n),
            'Department': rng.choice(['A', 'B', 'C'], n),
            'Billing Amount': rng.normal(2500, 500, n),
        })
        cls.classification['Outcome'] = np.where((cls.classification.Age > 50) & (cls.classification.Department == 'A'), 'Review', 'Routine')
        cls.regression = pd.DataFrame({'Area': rng.normal(100, 20, n), 'Bedrooms': rng.integers(1, 5, n), 'City': rng.choice(['X', 'Y'], n)})
        cls.regression['Price'] = cls.regression.Area * 1200 + cls.regression.Bedrooms * 15000 + rng.normal(0, 200, n)

    def test_classification_train_predict(self):
        info = train_best_model(self.classification, 'Outcome')
        x = prepare_prediction({'Age': 44, 'Department': 'A', 'Billing Amount': 2300}, info)
        self.assertEqual(info['task'], 'classification')
        self.assertIn(info['pipeline'].predict(x)[0], ['Review', 'Routine'])

    def test_regression_train_predict(self):
        info = train_best_model(self.regression, 'Price')
        x = prepare_prediction({'Area': 95, 'Bedrooms': 3, 'City': 'Y'}, info)
        self.assertEqual(info['task'], 'regression')
        self.assertTrue(np.isfinite(float(info['pipeline'].predict(x)[0])))

    def test_identifier_target_rejected(self):
        df = pd.DataFrame({'Record ID': range(50), 'Value': np.arange(50)})
        with self.assertRaisesRegex(ValueError, 'identifier'):
            train_best_model(df, 'Record ID')

    def test_too_small_data_rejected(self):
        with self.assertRaisesRegex(ValueError, '20 rows'):
            train_best_model(pd.DataFrame({'x': range(10), 'y': range(10)}), 'y')

if __name__ == '__main__': unittest.main()
