# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Data Preprocessing: Cleanse data, handle missing values, encode categorical variables.
2.Model Training: Fit logistic regression model on preprocessed data.
3.Model Evaluation: Assess model performance using metrics like accuracy, precision, recall.
4.Prediction: Predict placement status for new student data using trained model.
```

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PRAVEENA N
RegisterNumber: 212222040122 

import pandas as pd
data=pd.read_csv("C:/Users/admin/OneDrive/Documents/INTRO TO ML/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data.head()

data1.isnull()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![outp1](https://github.com/Praveenanagaraji22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393514/79a74076-69b9-4811-bb47-985ba13c3ed6)
![outp2](https://github.com/Praveenanagaraji22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393514/a1f1ddc0-2e90-4dbd-a0ce-c14a35a890e8)
![outp3](https://github.com/Praveenanagaraji22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393514/a7992a71-4528-42f7-88fa-575203abc960)
![outp4](https://github.com/Praveenanagaraji22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393514/bc56dedf-595d-4224-9d3e-3466cfc88cb4)
![outp5](https://github.com/Praveenanagaraji22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393514/68450eb5-3ce8-4db4-bdb4-97fa8a8fa908)
![outp6](https://github.com/Praveenanagaraji22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393514/09b9dcb1-c614-4c38-bb93-c9cba6fe76b1)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
