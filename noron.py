import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv('Milk-Grading-1.csv')

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
dt_train, dt_test = train_test_split(data, test_size=0.3, shuffle=True)

xtrain = dt_train.drop(['Grade'],axis=1)
ytrain = dt_train['Grade']
xtest = dt_test.drop(['Grade'],axis=1)
ytest = np.array(dt_test['Grade'])

model = MLPClassifier(hidden_layer_sizes=(600,500),max_iter=100)
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)

# count = 0
# for i in range (0, len(y_pred)):
#     if(ytest[i]==y_pred[i]): count += 1
# print(count/len(ytest))
a = accuracy_score(ytest, y_pred)
print("Ty le du doan dung Noron: ", a)

