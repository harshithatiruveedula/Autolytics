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

    def test_admission_type_by_condition_is_planned_when_available(self):
        df = self.df.copy()
        df["Admission Type"] = np.random.default_rng(2).choice(
            ["Emergency", "Urgent", "Elective"], len(df)
        )
        result = detect_domain(df)
        plan = plan_hospital_charts(df, result["mappings"], max_charts=10)
        chart = next(
            (item for item in plan
             if item["title"] == "Admission Type by Medical Condition"),
            None,
        )
        self.assertIsNotNone(chart)
        self.assertEqual(chart["cols"], ["Diagnosis", "Admission Type"])


    def test_admission_chart_is_protected_by_small_chart_limit(self):
        df = self.df.copy()
        df["Visit Category"] = np.random.default_rng(8).choice(
            ["Emergency", "Routine", "Urgent"], len(df)
        )
        result = detect_domain(df)
        plan = plan_hospital_charts(df, result["mappings"], max_charts=3)
        self.assertEqual(plan[0]["title"], "Admission Type by Medical Condition")
        self.assertLessEqual(len(plan), 3)

    def test_compact_patient_dataset_is_detected(self):
        df = pd.DataFrame({
            "Patient Age": [34, 51, 63, 45, 29, 70],
            "Sex": ["F", "M", "F", "M", "F", "M"],
            "Diagnosis": ["Asthma", "Diabetes", "Cancer", "Asthma", "Cancer", "Diabetes"],
        })
        result = detect_domain(df)
        self.assertTrue(result["is_healthcare"])
        titles = {x["title"] for x in plan_hospital_charts(df, result["mappings"], max_charts=10)}
        self.assertIn("Patient Distribution by Gender", titles)
        self.assertIn("Age Distribution", titles)

    def test_admission_type_aliases_are_detected(self):
        df = self.df.copy().rename(columns={"Diagnosis": "Disease", "Sex": "Gender"})
        df["Admission Category"] = np.random.default_rng(9).choice(
            ["Emergency", "Elective", "Urgent"], len(df)
        )
        result = detect_domain(df)
        self.assertEqual(result["mappings"]["admission_type"]["col"], "Admission Category")
        plan = plan_hospital_charts(df, result["mappings"], max_charts=10)
        self.assertIn("Admission Type by Medical Condition", {x["title"] for x in plan})

    def test_unfamiliar_column_names_inferred_from_values(self):
        # Deliberately non-standard headers: semantic value signatures should
        # still identify diagnosis, admission type, sex and lab-result fields.
        df = pd.DataFrame({
            "field_alpha": ["Asthma", "Diabetes", "Cancer", "Asthma", "Cancer", "Diabetes"],
            "field_beta": ["Emergency", "Elective", "Urgent", "Routine", "Emergency", "Urgent"],
            "field_gamma": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "field_delta": ["Normal", "Abnormal", "Positive", "Negative", "Normal", "Abnormal"],
            "amount_billed": [100.0, 250.0, 500.0, 125.0, 800.0, 450.0],
        })
        result = detect_domain(df)
        mapping = {k: v["col"] for k, v in result["mappings"].items()}
        self.assertEqual(mapping.get("condition"), "field_alpha")
        self.assertEqual(mapping.get("admission_type"), "field_beta")
        self.assertEqual(mapping.get("gender"), "field_gamma")
        self.assertEqual(mapping.get("test_result"), "field_delta")
        self.assertEqual(mapping.get("billing"), "amount_billed")
        self.assertTrue(result["is_healthcare"])

    def test_more_uncommon_admission_and_date_aliases(self):
        df = self.df.rename(columns={"Diagnosis": "Primary Dx", "Admit Date": "Encounter Date"})
        df["Encounter Class"] = np.random.default_rng(31).choice(
            ["Emergency", "Elective", "Outpatient"], len(df)
        )
        result = detect_domain(df)
        mapping = {k: v["col"] for k, v in result["mappings"].items()}
        self.assertEqual(mapping.get("condition"), "Primary Dx")
        self.assertEqual(mapping.get("admission_date"), "Encounter Date")
        self.assertEqual(mapping.get("admission_type"), "Encounter Class")

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
        titles = {x["title"] for x in plan}
        self.assertIn("Patient Distribution by Gender", titles)
        self.assertIn("Admission Type by Medical Condition", titles)

if __name__ == "__main__":
    unittest.main()
