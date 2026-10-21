AUTOLYTICS individual-record query fix
========================================

Replace your existing query.py with the included query.py.

The update adds a schema-driven individual-detail lookup before the existing
natural-language query rules. It identifies the requested column and then
filters by a matching ID value or person/name value. It does not hard-code
education or hospital column names.

Education dataset examples:
- gender of student_id 1
- gender of student 1
- gender of 1
- math_score of student_id 1
- graduation date of student 2
- show all details of student_id 1 (use the existing row display behavior)

Hospital Operations Analytics examples:
- Age of Bobby JacksOn
- Medical Condition of Bobby JacksOn
- Billing Amount of Bobby JacksOn
- Test Results of Bobby JacksOn
- Patient Satisfaction of patient_id PAT00001
- Risk Level of patient_id PAT00002

Workbook note:
- healthcare_dataset sheet has 15 columns and supports person/name lookups.
- Working_Data has 25 columns including Patient ID, and supports patient-ID lookups.
Use the data sheet that you intend to query in AUTOLYTICS.

Validation performed on the provided education CSV and hospital workbook sheets.
