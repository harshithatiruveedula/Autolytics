# AUTOLYTICS Natural-Language Query Guide

AUTOLYTICS uses the active dataset's column names and values. These are supported query patterns and examples, not a fixed list of every sentence a person could type. Use the real column names shown after uploading a dataset; spaces, underscores, and many hyphenated column names are normalized for matching.

## 1. General query patterns (work across datasets)

### Individual values / record lookup
- `gender of student_id 1`
- `math_score of student 1`
- `age of patient_id PAT00001`
- `amount of order id 405-8078784-5731545`
- `card_type of client 825`
- `credit_limit of id 4524`
- `order id of index 1`
- `client id of index 1`
- `client id of 1` (when no matching record identifier is explicitly named, a bare numeric index is treated as a row position)

For an exact entity lookup, name the identifier where possible: `amount of order id <actual order ID>` or `card_type of client 825`. `index N` uses an actual `index`/`row index` column when one exists; otherwise it uses dataframe row position N (zero-based, so position 1 is the second row).

### Filter a selected column
- `order id who have status as shipped`
- `order id where status is cancelled`
- `card brand where card_type is credit`
- `student_id who have gender as female`
- `client id where card_on_dark_web is yes`
- `order id with status shipped and category kurta`

### Filter and return complete records
- `show all rows where status is shipped`
- `show records with card_type as credit`
- `list rows where gender is female and grade_level is 6`

### Grouping / aggregation
- `group extracurriculars by student_id`
- `group by category and status`
- `group category and status`
- `count orders by status`
- `count card_type by card_brand`
- `average amount by ship-state`
- `total amount by category`
- `sum qty by size`
- `average math_score by gender and grade_level`
- `count rows by card_brand and card_type`

### Dataset exploration
- `show dataset` / `show data`
- `first 20 rows`
- `how many rows`
- `unique values of status`
- `count status`
- `missing values in Amount`
- `average Amount`
- `total Amount`
- `highest Amount` / `lowest Amount`
- `top 10 Amount`

You can also enter read-only SQL `SELECT` queries.

## 2. Amazon Sale Report dataset

Columns: `index`, `Order ID`, `Date`, `Status`, `Fulfilment`, `Sales Channel `, `ship-service-level`, `Style`, `SKU`, `Category`, `Size`, `ASIN`, `Courier Status`, `Qty`, `currency`, `Amount`, `ship-city`, `ship-state`, `ship-postal-code`, `ship-country`, `promotion-ids`, `B2B`, `fulfilled-by`, `Unnamed: 22`.

Examples:
- `order id of index 1`
- `status of order id 405-8078784-5731545`
- `amount of order id 405-8078784-5731545`
- `order id who have status as shipped`
- `order id where status is cancelled`
- `order id with status shipped and category kurta`
- `show all rows where ship-state is Maharashtra`
- `count orders by status`
- `total amount by category`
- `average amount by ship-state`
- `sum qty by size`
- `group category and status`
- `unique values of ship-city`
- `top 10 Amount`
- `missing values in Amount`

The result display is capped at 1,000 rows for responsiveness. Add a filter or use an aggregate/count query for larger results.

## 3. Finance `cards_data` dataset

Columns: `id`, `client_id`, `card_brand`, `card_type`, `card_number`, `expires`, `cvv`, `has_chip`, `num_cards_issued`, `credit_limit`, `acct_open_date`, `year_pin_last_changed`, `card_on_dark_web`.

Examples:
- `client id of index 1`
- `card_type of client 825`
- `card_brand of id 4524`
- `credit_limit of client 825`
- `card brand where card_type is credit`
- `client id where card_on_dark_web is yes`
- `count cards by card_brand`
- `count card_type by card_brand`
- `average num_cards_issued by card_brand`
- `group card_brand and card_type`
- `unique values of has_chip`
- `count card_on_dark_web`

**Privacy:** this file contains `card_number` and `cvv` columns. Do not publish the raw file or expose those fields in a public demo. Prefer removing or masking them before deployment.

## 4. Education dataset

Columns: `Student_ID`, `Age`, `Gender`, `Grade_Level`, `Math_Score`, `Reading_Score`, `Writing_Score`, `Attendance_Rate`, `Extracurriculars`, `Graduation_Date`.

Examples:
- `gender of student_id 1`
- `math_score of student 1`
- `graduation date of student 2`
- `student_id who have gender as female`
- `student_id where grade_level is 6`
- `average math_score by gender`
- `average attendance_rate by grade_level`
- `group extracurriculars by student_id`
- `count extracurriculars by gender`
- `unique values of grade_level`
- `missing values in extracurriculars`

## 5. Hospital Operations dataset

Use the actual sheet uploaded. Common raw fields include `Name`, `Age`, `Gender`, `Blood Type`, `Medical Condition`, `Date of Admission`, `Doctor`, `Hospital`, `Insurance Provider`, `Billing Amount`, `Room Number`, `Admission Type`, `Discharge Date`, `Medication`, and `Test Results`. The `Working_Data` sheet also contains fields such as `Patient ID`, `Department`, `Length of Stay`, `Age Group`, `Waiting Time`, `Patient Satisfaction`, `Billing Category`, `Risk Level`, `Discharge Status`, and `Readmission`.

Examples:
- `Age of Bobby JacksOn`
- `Medical Condition of Bobby JacksOn`
- `Patient Satisfaction of patient_id PAT00001`
- `Risk Level of patient_id PAT00002`
- `patient id where risk level is medium`
- `count medical condition by gender`
- `average billing amount by department`
- `average length of stay by admission type`
- `group risk level and discharge status`
- `unique values of insurance provider`
- `missing values in patient satisfaction`

## 6. Important behaviour

- Individual lookups return the requested field rather than all columns.
- Explicit IDs are preferred for a record lookup; `index N` selects a row by index/position.
- Filtered projection queries return only the requested output columns and matching rows.
- Multi-column grouping should keep all grouping dimensions in the generated SQL.
- Common category filtering uses the values present in the uploaded dataset; a short category such as `shipped` can match longer labels such as `Shipped - Delivered to Buyer`.
- If a question has several possible meanings, name the identifier/column explicitly or use SQL to remove ambiguity.
