AUTOLYTICS — Natural-Language Grouping and Query Fix
======================================================

Files
-----
- query.py: replacement for the existing AUTOLYTICS query.py
- README.txt: installation notes and verified test cases

What changed
------------
1. Multi-column grouping now uses every explicitly mentioned grouping column.
2. "group X by Y" groups X and Y and returns a record_count.
3. "group by X and Y" supports two or more grouping dimensions.
4. Aggregate-by queries support multiple dimensions, for example:
   - average Math_Score by Gender and Grade_Level
   - total Amount by Category and ship-state
5. "count X by Y" counts X for each Y value instead of counting all values of X
   and ignoring the requested grouping column.
6. The pandas fast path no longer overwrites grouped SQL results with a
   single-column count or sum.
7. Column matching avoids treating "id" as a separate column when the question
   actually mentions "client_id"; person/record lookup prioritizes the ID
   column explicitly named by the user.
8. Earlier individual-detail lookups and category-count handling are retained.

Installation
------------
1. Back up the current query.py.
2. Copy this query.py into the same project folder, replacing the existing file.
3. Keep app.py and the other project files unchanged.
4. Stop Streamlit (Ctrl+C) and restart it, for example:
      streamlit run app.py

Verified query examples
-----------------------
Education:
- group extracurriculars by student_id
- group extracurriculars and gender by student_id
- group by extracurriculars and gender
- count extracurriculars by student_id
- average math_score by gender and grade_level
- gender of student_id 1

Hospital Operations (Working_Data and healthcare_dataset sheets):
- group medical condition by age
- average age by gender
- count of medical condition
- Medical Condition of Bobby JacksOn
- group gender and medical condition by hospital

E-commerce (Amazon Sale Report.csv):
- group Category by ship-state
- group Category and Status by ship-state
- group by Category and Status
- total Amount by Category and ship-state
- count Category by ship-state
- Status of Order ID 405-8078784-5731545

Finance/cards_data.csv:
- group card_brand by card_type
- group card_brand and has_chip by client_id
- group by card_brand and card_type
- count card_brand by client_id
- card_type of client_id 825

Important
---------
The query logic is schema-driven; it does not hard-code these dataset names or
columns. The dataset must still be uploaded/loaded by the existing AUTOLYTICS
application as usual.

The uploaded cards_data.csv includes columns named card_number and cvv. Do not
publish that raw file or real payment-card details to a public repository or
deployment.
