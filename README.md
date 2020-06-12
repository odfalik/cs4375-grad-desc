
### How to Run:
python assignment1.py

For no plots use:
python assignment1.py noplot

### Libraries Used
- numpy
- pandas
- scikit-learn (sklearn)
- matplotlib
- requests, io, sys

Developed on Python 3.7.5
Should work on Python 3+

### Description
For this assignment, we processed a dataset from the UCI Machine Learning Repository [1]. This dataset contained the MPG for several hundred models of cars along with their specifications such as displacement, mileage, horsepower, year, origin, and name. After an initial preprocessing which involved stripping non-numerical attributes and deleting samples with missing attributes, resulting in a total dataset of 392 samples with seven attributes. Next, we wrote a Python program to learn a gradient descent-based linear regression model to predict the MPG of a car and evaluated it against Scikit-learn’s linear regression model