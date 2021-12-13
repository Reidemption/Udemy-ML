# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data via .csv
dataset = pd.read_csv("Data.csv")
#purchased section is the dependent variable, usually in the last column
#features are everything else. e.g. Country, Age, Salary
x = dataset.iloc[:, :-1].values #independent
y = dataset.iloc[:, -1].values #dependent
#print(f"x: {x}")
#print(f"y: {y}")

#taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #replaces all empty values with the mean of the column
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3]) #returns updated version with changes
# print(x)

# encoding categorical data "independent data"
# using numbers to represent string values can cause errors in ML outcomes because the algorithm might interpret these numerical values as important
# instead make a vector for a different values in a category "one hot encoding"
# dependent variable can change "No" and "Yes" to 0 and 1; won't impact the outcome
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
# print(x)

# encoding the DEPENDENT variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#Split the dataset into the Training and Test sets
