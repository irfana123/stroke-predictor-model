import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df=pd.read_csv("stroke.csv")

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["gender"]= le.fit_transform(df["gender"])

le=LabelEncoder()
df["work_type"]= le.fit_transform(df["work_type"])

le=LabelEncoder()
df["Residence_type"]= le.fit_transform(df["Residence_type"])


le=LabelEncoder()
df["ever_married"]= le.fit_transform(df["ever_married"])

le=LabelEncoder()
df["smoking_status"]= le.fit_transform(df["smoking_status"])

median=df["bmi"].median()
df_m=df.copy()
df_m["bmi"]=df_m["bmi"].fillna(median)
df=df_m
df=df.drop("id",axis=1)

x=df.iloc[:,0:10].values
y=df.iloc[:,-1].values

#Standard scaler

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x= sc.fit_transform(x)  


#splitting into training and testing

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=0)



#  fitting training and testing dataset into randomforest model

from sklearn.ensemble import RandomForestClassifier

random_model = RandomForestClassifier(criterion="gini",n_estimators=10,random_state=1)
random_model.fit(x_train,y_train)

y_pred= random_model.predict(x_test)

#confusion metrics

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm =confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
cr = classification_report(y_test,y_pred)


