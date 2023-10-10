# This is all the code for HW2.

- ```HW2.m``` contains all the code for Question 1, Part A.
- ```HW2_1B.m``` contains all code for Question 1, Part B.
- ```HW2_Q2.m``` contains all code for Question 2, Part A.
- ```HW2_Q2B.m``` contains all code for Question 2, Part B. This code requires modifying line 8 to use the correct cost matrix. For example:
```matlab
Mdl = fitcnb(samples',true_label','Cost',costA100');
```

Note that ```generateDataA1Q1.m``` was initially provided by the instructor. It was modified and saved as ```generateDataQ2.m``` to support Question 2.
