import numpy as np
from sklearn import preprocessing, model_selection, neighbors,svm
import pandas as pd
import matplotlib.pyplot as plt
desired_width=320

pd.set_option('display.width', desired_width)


pd.set_option('display.max_columns',22)

df = pd.read_csv('tic-tac-toe.data.txt')

t_l_s = {'x':0,'o':1,'b':2}
df.top_left_square = [t_l_s[item] for item in df.top_left_square]

t_m_s = {'x':0,'o':1,'b':2}
df.top_middle_square = [t_m_s[item] for item in df.top_middle_square]

t_r_s = {'x':0,'o':1,'b':2}
df.top_right_square = [t_r_s[item] for item in df.top_right_square]

m_l_s = {'x':0,'o':1,'b':2}
df.middle_left_square = [m_l_s[item] for item in df.middle_left_square]

m_m_s = {'x':0,'o':1,'b':2}
df.middle_middle_square = [m_m_s[item] for item in df.middle_middle_square]

m_r_s = {'x':0,'o':1,'b':2}
df.middle_right_square = [m_r_s[item] for item in df.middle_right_square]

b_l_s = {'x':0,'o':1,'b':2}
df.bottom_left_square = [b_l_s[item] for item in df.bottom_left_square]

b_m_s = {'x':0,'o':1,'b':2}
df.bottom_middle_square = [m_r_s[item] for item in df.bottom_middle_square]

b_r_s = {'x':0,'o':1,'b':2}
df.bottom_right_square = [m_r_s[item] for item in df.bottom_right_square]

full_data = df.values.tolist()

x = np.array(df.drop(['Class'], 1))

y = np.array(df['Class'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
clf = svm.SVC(gamma='scale',kernel='poly')
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)

print(accuracy)

example_measures = np.array([[[[0,0,1,0,1,1,0,2,1],[1,0,1,2,1,1,0,2,1],[1,0,1,2,1,1,0,2,1],[0,0,1,0,1,1,0,2,1]]]])
print(len(example_measures))
#example_measures = example_measures.reshape(len(example_measures), -1)
example_measures = example_measures.reshape(4,-1)
prediction = clf.predict(example_measures)
print(prediction)


