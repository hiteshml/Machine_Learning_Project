import numpy as np
from sklearn.svm import SVC 
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
df['bare_nuclei'] = pd.to_numeric(df.bare_nuclei)

x= np.array(df.drop(['class'],1))
y= np.array(df['class'])

x_train , x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2)
clf = svm.SVC(C=1,gamma='auto',probability=True,kernel='linear')
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
accuracy1 = clf.score(x_train, y_train)
print(accuracy, accuracy1)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1],[7,9,1,2,9,6,1,2,10]])
example_measures = example_measures.reshape(3,-1)
print('\n\n',clf.predict_proba(example_measures))
prediction = clf.predict(x_train[:20])
prediction1 = clf.predict(x_test[:20])
prediction2 = clf.predict(example_measures)
print('\n\n',prediction2)

plt.plot(x_train.min(axis=0), 'o', label='Min')
plt.plot(x_train.max(axis=0), 'v', label='Max')
plt.xlabel('Feature Index')
plt.ylabel('Feature magnitude of log scale')
plt.yscale('log')
plt.legend(loc='upper right')
plt.show()
