# importing the dataset
import pandas as pd
import numpy as np
data=pd.read_csv("combined.csv")
data.info()



#overlooking the dataset
data=pd.DataFrame(data)
data.columns
data.nunique()


#data preparation
data['Category'].describe()
data = data.drop('Measure:volume', axis=1)   
X = data.drop('Category', axis=1)  
y=data['Category']

data.corr()


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

#SVM without optimisation

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)  
y_pred = svclassifier.predict(X_test)  


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 

# # from sklearn import svm, grid_search
# # def svc_param_selection(X, y, nfolds):
# #     Cs = [0.001, 0.01, 0.1, 1, 10]
# #     gammas = [0.001, 0.01, 0.1, 1]
# #     param_grid = {'C': Cs, 'gamma' : gammas}kernel
# #     grid_search = GridSearchCV(svm.SVC(='linear'), param_grid, cv=nfolds)
# #     grid_search.fit(X, y)
# #     grid_search.best_params_
# #     return grid_search.best_params_



# from sklearn import svm, datasets
# from sklearn.model_selection import GridSearchCV
# gammas = [0.001, 0.01, 0.1, 1]
# Cs = [0.001, 0.01, 0.1, 1, 10]

# parameters = {'C':[1, 10],'gamma':gammas}
# svc = svm.SVC(kernel='linear')
# clf = GridSearchCV(svc, parameters, cv=5)
# clf.fit(X_train, y_train)


X = data.drop('Category', axis=1)  
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
y=data['Category']

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)  

from sklearn.svm import SVC
from sklearn import metrics
svc=SVC() #Default hyperparameters
svc.fit(X_train,y_train)
y_pred=svc.predict(X_train)
print('Accuracy Score: when cross-validated on training set')
print(metrics.accuracy_score(y_train,y_pred))

y_pred_test=svc.predict(X_test)
print('Accuracy Score: when checked on test set')
print(metrics.accuracy_score(y_test,y_pred_test))


svc=SVC(kernel='linear')
svc.fit(X_train,y_train)

y_pred=svc.predict(X_train)

print('Accuracy Score: when cross-validated on training set')
print(metrics.accuracy_score(y_train,y_pred))

y_pred_test=svc.predict(X_test)
print('Accuracy Score: when checked on test set')

print(metrics.accuracy_score(y_test,y_pred_test))



svc=SVC(kernel='rbf')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_train)

print('Accuracy Score: when cross-validated on training set')
print(metrics.accuracy_score(y_train,y_pred))

y_pred_test=svc.predict(X_test)
print('Accuracy Score: when checked on test set')

print(metrics.accuracy_score(y_test,y_pred_test))


from sklearn.cross_validation import cross_val_score
svc=SVC(kernel='linear')
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
print(scores)
print(scores.mean())

from sklearn.cross_validation import cross_val_score
svc=SVC(kernel='rbf')
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
print(scores)
print(scores.mean())

#checking performance of model over a range of C values 
C_range=list(range(1,26))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)    
#c=2, gives the best score


import matplotlib.pyplot as plt
%matplotlib inline


C_values=list(range(1,26))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score)
plt.xticks(np.arange(0,27,2))
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')

C_range=list(np.arange(0.4,3,0.1))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)

import matplotlib.pyplot as plt
%matplotlib inline

C_values=list(np.arange(0.4,3,0.1))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score)
plt.xticks(np.arange(0.0,6,0.3))
plt.xlabel('Value of C for SVC ')
plt.ylabel('Cross-Validated Accuracy')

#c=0.5


#gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]
gamma_range=[0.0001,0.001,0.01,0.1,1]
acc_score=[]
for g in gamma_range:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)    
    
    
import matplotlib.pyplot as plt
%matplotlib inline

# gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]
gamma_range=[0.0001,0.001,0.01,0.1,1]
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(gamma_range,acc_score)
plt.xlabel('Value of gamma for SVC ')
plt.xticks(np.arange(0.0001,1,5))
plt.ylabel('Cross-Validated Accuracy')

gamma_range=[0.0001,0.001,0.01,0.1]
acc_score=[]
for g in gamma_range:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)

import matplotlib.pyplot as plt
%matplotlib inline

gamma_range=[0.0001,0.001,0.01,0.1]

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(gamma_range,acc_score)
plt.xlabel('Value of gamma for SVC ')
plt.ylabel('Cross-Validated Accuracy')


gamma_range=[0.001,0.002,0.003,0.004,0.005]
acc_score=[]
for g in gamma_range:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)  

import matplotlib.pyplot as plt
%matplotlib inline

gamma_range=[0.001,0.002,0.003,0.004,0.005]

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(gamma_range,acc_score)
plt.xlabel('Value of gamma for SVC ')
plt.ylabel('Cross-Validated Accuracy')

# degree=[2,3,4,5,6]
# acc_score=[]
# for d in degree:
#     svc = SVC(kernel='poly', degree=d)
#     scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#     acc_score.append(scores.mean())
# print(acc_score)


# import matplotlib.pyplot as plt
# %matplotlib inline

# degree=[2,3,4,5,6]

# # plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
# plt.plot(degree,acc_score,color='r')
# plt.xlabel('degrees for SVC ')
# plt.ylabel('Cross-Validated Accuracy')


#performing SVM using best value of gamma and c
from sklearn.svm import SVC
svc= SVC(kernel='linear',C=0.45,gamma=0.02)
svc.fit(X_train,y_train)
y_predict=svc.predict(X_test)
accuracy_score= metrics.accuracy_score(y_test,y_predict)
print(accuracy_score)

#K-fold cross validation with k=10
from sklearn.cross_validation import cross_val_score
svc=SVC(kernel='linear',C=0.1)
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
print(scores)


from sklearn.svm import SVC
svc= SVC(kernel='rbf',gamma=0.01)
svc.fit(X_train,y_train)
y_predict=svc.predict(X_test)
metrics.accuracy_score(y_test,y_predict)


svc=SVC(kernel='rbf',gamma=0.01)
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
print(scores)
print(scores.mean())

from sklearn.svm import SVC
svc= SVC(kernel='poly',degree=3)
svc.fit(X_train,y_train)
y_predict=svc.predict(X_test)
accuracy_score= metrics.accuracy_score(y_test,y_predict)
print(accuracy_score)

from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
steps = [('scaler', StandardScaler()), ('SVM', SVC())]
from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps)

model_svm = GridSearchCV(pipeline, tuned_parameters,cv=10,scoring='accuracy')
model_svm.fit(X_train, y_train)
print(model_svm.best_score_)
y_pred = model_svm.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 
