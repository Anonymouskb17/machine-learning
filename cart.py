import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('Milk-Grading-1.csv')
le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
dt_train, dt_test = train_test_split(data, test_size=0.3, shuffle=True)

xtrain = dt_train.drop(['Grade'],axis=1)
ytrain = dt_train['Grade']
xtest = dt_test.drop(['Grade'],axis=1)
ytest = np.array(dt_test['Grade'])

model = DecisionTreeClassifier(criterion='gini', max_depth=1000,splitter='random')
# model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
a = accuracy_score(ytest, y_pred)
precision = precision_score(ytest, y_pred, average='macro')
recall = recall_score(ytest,y_pred, average='macro')
f1 = f1_score(ytest, y_pred, average='macro')
print("Ty le accuracy cua CART: ", a)
print("Ty le precision cua CART: ", precision)
print("Ty le recall cua CART: ", recall)
print("Ty le f1_score cua CART: ", f1)

