# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: J.JANANI
RegisterNumber:  212223230085
*/

import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


```

## Output:

![Screenshot 2025-05-17 140847](https://github.com/user-attachments/assets/bdabdd29-fa05-4fed-b046-78b414661660)

![Screenshot 2025-05-17 140855](https://github.com/user-attachments/assets/4d00cb32-dcd8-43a5-9bfe-1eca164c3ad6)


![Screenshot 2025-05-17 140901](https://github.com/user-attachments/assets/2c06fa26-9a32-41f2-8c50-a489e51b29c8)


![Screenshot 2025-05-17 141837](https://github.com/user-attachments/assets/54602052-c3bd-45fc-b440-19af96a1d9c8)


![Screenshot 2025-05-17 141845](https://github.com/user-attachments/assets/a75b7e38-d16f-4858-89a2-4ea57d574f39)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
