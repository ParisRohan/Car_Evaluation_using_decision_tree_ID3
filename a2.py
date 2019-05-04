import numpy as np
import pandas as pd

DataSet=pd.read_csv("car.csv")
x=DataSet.iloc[:,:-1]
y=DataSet.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder 
LE=LabelEncoder()
x=x.apply(LabelEncoder().fit_transform)
print(x)

from sklearn.tree import DecisionTreeClassifier as DTC
tree = DTC(criterion='entropy')
model = tree.fit(x,y)

tree.fit(x.iloc[:,0:6],y)

X_in=np.array([3,2,1,0,1,2]) 	#to predict
y_pred=tree.predict([X_in])
print("Prediction:", y_pred)

import graphviz as gv
import sklearn.tree as tree

gv_comp_model = tree.export_graphviz(model,feature_names=["buying","maintenance","doors","persons","lug_boot","safety"],class_names=['unacc','acc','good','vgood'])

x1 = gv.Source(gv_comp_model)
x1.render("treePDF")



