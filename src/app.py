import pandas as pd 
import matplotlib.pyplot as plt 
from pickle import dump

from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from utils import db_connect

engine = db_connect()

# your code here
total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")

total_data=total_data.drop_duplicates().reset_index(drop=True)

x=total_data.drop("Outcome", axis =1)
y=total_data["Outcome"]

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size= 0.2 , random_state= 42)

selection_model= SelectKBest(k=7)
selection_model.fit(x_train,y_train)

selected_columns = x_train.columns[selection_model.get_support()]
x_train_sel=pd.DataFrame(selection_model.transform(x_train),columns= selected_columns)
x_test_sel=pd.DataFrame(selection_model.transform(x_test), columns= selected_columns)

x_train_sel["Outcome"] = y_train.values
x_test_sel["Outcome"] = y_test.values

x_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
x_test_sel.to_csv("../data/processed/clean_test.csv", index = False)

train_data=pd.read_csv("../data/processed/clean_train.csv")
test_data=pd.read_csv("../data/processed/clean_test.csv")

model= RandomForestClassifier(n_estimators=60, random_state=42)
model.fit(x_train,y_train)

fig, axis = plt.subplots(2, 2, figsize = (15, 15))

tree.plot_tree(model.estimators_[0], ax=axis[0,0],feature_names=list(x_train.columns), class_names=["0","1","2"], filled=True)
tree.plot_tree(model.estimators_[1], ax=axis[0,1],feature_names=list(x_train.columns), class_names=["0","1","2"], filled=True)
tree.plot_tree(model.estimators_[2], ax=axis[1,0],feature_names=list(x_train.columns), class_names=["0","1","2"], filled=True)
tree.plot_tree(model.estimators_[3], ax=axis[1,1],feature_names=list(x_train.columns), class_names=["0","1","2"], filled=True)

plt.show()

y_pred=model.predict(x_test)

y_pred

accuracy_score(y_test,y_pred)

dump(model, open("Random_Forest_Classifier_default_42.sav", "wb"))
