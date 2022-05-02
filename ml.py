import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
a=pd.read_csv('heart.csv')
a

lbcode = LabelEncoder()
arr=[]
for i in a:
  k=0                                        
  for j in a[i]:
    if(pd.isna(j)):
      if(k not in arr):
        arr.append(k)
    k+=1
a=a.drop(arr)
# df.info()
s=0
c=[]
k=[]
arr2=[]
for j in a:
  x=a[j].dtype
  if(x=="object" ):
    a[j]=LabelEncoder().fit_transform(a[j])
    arr2.append(j)
    s+=1
  elif(x!="object"):
    k.append(j)
    c.append(s)
    s+=1

x = a.iloc[:,0:11]
y = a.iloc[:,11]

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators = 100) 
 
# # Training the model on the training dataset
# # fit function is used to train the model using the training sets as parameters
# d=clf.fit(x_train, y_train)

# pickle.dump(d,open('iri.pkl','wb'))


# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# a["Sex"]=le.fit_transform(a["Sex"])
# a["ChestPainType"]=le.fit_transform(a["ChestPainType"])
# a["RestingECG"]=le.fit_transform(a["RestingECG"])
# a["ExerciseAngina"]=le.fit_transform(a["ExerciseAngina"])
# a["ST_Slope"]=le.fit_transform(a["ST_Slope"])
# a=a.drop(columns=['Age','Oldpeak']) 

# x=a.drop(columns=["HeartDisease"])
# y=a["HeartDisease"]
x_new=x.copy()
for column in x_new:
    x_new[column] = (x_new[column] - x_new[column].min()) / (x_new[column].max() - x_new[column].min())    
x_new
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = LGBMClassifier()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')
d=eclf.fit(x_new,y)
# y_pred=eclf.predict(X_test)
# d=clf.fit(x_new, y)
# b=clf.predict(x_test)
# from sklearn.metrics import accuracy_score
# c=accuracy_score(y_test,b)
# c
pickle.dump(d,open('classifier3.pkl','wb'))
