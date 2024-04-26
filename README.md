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
*/
```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/OneDrive/Documents/INTRO TO ML/Placement_Data.csv")
data.head()
```
```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data.head()
```
```
data1.isnull()
```
```
data1.duplicated().sum()
```
```
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
```
```
x=data1.iloc[:,:-1]
x
```
```
y=data1["status"]
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:

![image](https://github.com/Jaiganesh235/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118657189/7b380642-9d1e-41cc-9170-80ae57cf6b18)

![image](https://github.com/Jaiganesh235/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118657189/18535a8c-5a2a-4436-9751-5557e1cf3bb2)
![image](https://github.com/Jaiganesh235/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118657189/93f286c6-b0fc-4f5b-ba6e-b048f2562660)
![image](https://github.com/Jaiganesh235/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118657189/704a14ac-b9b3-4866-8630-c291e17e9500)
![image](https://github.com/Jaiganesh235/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118657189/ecf7a832-93a4-4d6e-b2f6-119bb0118768)
![image](https://github.com/Jaiganesh235/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118657189/349daa92-6bec-4078-812e-4ac98b100848)
![image](https://github.com/Jaiganesh235/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118657189/1c5ceec3-c7a4-4802-b499-d6568f5999a1)
![image](https://github.com/Jaiganesh235/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118657189/ac7b7dc9-d82d-462d-9e24-72f44a3a45e1)
![image](https://github.com/Jaiganesh235/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118657189/67c6f6b0-3be8-4367-85fe-6bf0e2fe0876)
![image](https://github.com/Jaiganesh235/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118657189/c5a7f625-e631-4963-9405-c30c08afcb11)
![image](https://github.com/Jaiganesh235/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118657189/1f1f78c5-7264-4bde-ab61-b3371f9064ba)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
