import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


data = pd.read_csv('weatherAUS.csv')
le=preprocessing.LabelEncoder()
data=data.apply(le.fit_transform)

dt_train, dt_test = train_test_split(data, test_size=0.3, shuffle=True)

xtrain = dt_train.drop(['Date','Location','Evaporation','Sunshine',
                        'WindGustDir','WindDir9am','WindDir3pm',],axis=1)
ytrain = dt_train['']
xtest = dt_test.drop(['Grade'],axis=1)
ytest = np.array(dt_test['Grade'])

model = make_pipeline(StandardScaler(),SVC(gamma='scale',kernel='rbf', C=1000.0, tol=2e-3))
# model = make_pipeline(StandardScaler(),SVC())
model.fit(xtrain,ytrain)
y_pred=model.predict(xtest)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

a = accuracy_score(ytest, y_pred)
precision = precision_score(ytest, y_pred, average='macro')
recall = recall_score(ytest,y_pred, average='macro')
f1 = f1_score(ytest, y_pred, average='macro')
print("Ty le accuracy cua SVM: ", a)
print("Ty le precision cua SVM: ", precision)
print("Ty le recall cua SVM ", recall)
print("Ty le f1_score cua SVM: ", f1)