import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import numpy as np


# Read Data
df = pd.read_csv('diabetes.csv')

# Data
x = df.drop('Outcome',axis=1).values
# Resault

y = df['Outcome'].values

#split data
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.4)

# # Model
knn = KNeighborsClassifier(n_neighbors= 10)

# train
knn.fit (x_train,y_train)

#predict 
y_predict= knn.predict(x_test)

print(classification_report(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))
print(accuracy_score(y_test,y_predict)*100)


