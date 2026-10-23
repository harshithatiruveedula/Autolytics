"""Regression tests for healthcare semantic detection and automatic chart planning."""
from pathlib import Path
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from domain import detect_domain, plan_hospital_charts

class HospitalEdaTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        n = 120
        self.df = pd.DataFrame({
            "Diagnosis": rng.choice(["Cancer", "Diabetes", "Asthma"], n),
            "Drug": rng.choice(["Aspirin", "Metformin", "Ibuprofen"], n),
            "Bill Amount": rng.uniform(100, 5000, n),
            "Admit Date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%d-%m-%Y"),
            "Blood Group": rng.choice(["A+", "O+", "B-"], n),
            "Lab Results": rng.choice(["Normal", "Abnormal", "Inconclusive"], n),
            "Sex": rng.choice(["Male", "Female", "Other"], n),
            "Patient Age": rng.integers(18, 90, n),
        })

    def test_aliases_map_to_expected_concepts(self):
        result = detect_domain(self.df)
        self.assertTrue(result["is_healthcare"])
        mapping = {k: v["col"] for k, v in result["mappings"].items()}
        self.assertEqual(mapping["condition"], "Diagnosis")
        self.assertEqual(mapping["medication"], "Drug")
        self.assertEqual(mapping["billing"], "Bill Amount")
        self.assertEqual(mapping["admission_date"], "Admit Date")
        self.assertEqual(mapping["blood_type"], "Blood Group")
        self.assertEqual(mapping["test_result"], "Lab Results")
        self.assertEqual(mapping["gender"], "Sex")
        self.assertEqual(mapping["age"], "Patient Age")

    def test_core_charts_include_gender(self):
        result = detect_domain(self.df)
        plan = plan_hospital_charts(self.df, result["mappings"], max_charts=10)
        titles = {item["title"] for item in plan}
        self.assertIn("Medication Usage by Medical Condition", titles)
        self.assertIn("Patient Admissions Over Time", titles)
        self.assertIn("Blood Type Distribution by Medical Condition", titles)
        self.assertIn("Test Results by Medical Condition", titles)
        self.assertIn("Patient Distribution by Gender", titles)
        self.assertIn("Age Distribution", titles)

    def test_gender_chart_skipped_without_gender_field(self):
        df = self.df.drop(columns=["Sex"])
        result = detect_domain(df)
        plan = plan_hospital_charts(df, result["mappings"], max_charts=10)
        self.assertNotIn("Patient Distribution by Gender", {x["title"] for x in plan})

    def test_real_hospital_dataset(self):
        dataset = Path(__file__).resolve().parent.parent / "Hospital_Operations_Analytics.csv"
        if not dataset.exists():
            self.skipTest("Uploaded hospital dataset is not in the project parent folder")
        df = pd.read_csv(dataset)
        result = detect_domain(df)
        self.assertTrue(result["is_healthcare"])
        plan = plan_hospital_charts(df, result["mappings"], max_charts=10)
        self.assertIn("Patient Distribution by Gender", {x["title"] for x in plan})

if __name__ == "__main__":
    unittest.main()
