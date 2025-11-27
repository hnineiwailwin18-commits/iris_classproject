from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import pickle

X,y=load_iris(return_X_y=True,as_frame=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)

lr=LogisticRegression()

lr.fit(X_train,y_train)

pred_value=lr.predict(X_test)

print("Accuracy score ",(pred_value,y_test)*100)