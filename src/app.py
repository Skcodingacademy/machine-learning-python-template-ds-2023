import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from pickle import dump
from utils import db_connect
engine = db_connect()

# your code here
total_data=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")

total_data=total_data.drop_duplicates().reset_index(drop=True)
# Feature selection

x= total_data.drop("Outcome", axis = 1)
y=total_data["Outcome"]

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)

selection_model=SelectKBest(k=7)
selection_model.fit(x_train,y_train)

selected_columns= x_train.columns[selection_model.get_support()]
x_train_sel=pd.DataFrame(selection_model.transform(x_train), columns = selected_columns)
x_test_sel = pd.DataFrame(selection_model.transform(x_test), columns= selected_columns)

x_train_sel["Outcome"] = y_train.values
x_test_sel["Outcome"]= y_test.values

x_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
x_test_sel.to_csv("../data/processed/clean_test.csv", index = False)

# Tree model

train_data=pd.read_csv("../data/processed/clean_train.csv")
test_data=pd.read_csv("../data/processed/clean_test.csv")

plt.figure(figsize=(12,6))

pd.plotting.parallel_coordinates(total_data,"Outcome", color=("#E58139", "#39E581", "#8139E5"))

plt.show()

x_train=train_data.drop(["Outcome"],axis=1)
y_train=train_data["Outcome"]
x_test=test_data.drop(["Outcome"], axis=1)
y_test=test_data["Outcome"]

model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)

fig= plt.figure(figsize=(15,15))

tree.plot_tree(model, feature_names= list(x_train.columns), class_names=["0","1","2"], filled = True)

plt.show()

y_pred= model.predict(x_test)

accuracy_score(y_test , y_pred)

# Model optimization

hyperparams={
    "criterion":["gini","entropy"],
    "max_depth":[None,5,10,20],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4]

}

grid=GridSearchCV(model,hyperparams,scoring="accuracy", cv = 10)


def warn(*args, **kwargs):
    pass
import warnings
warnings.Warn=warn
grid.fit(x_train, y_train)

print(f"best hyperparams: {grid.best_params_}")

model= DecisionTreeClassifier(criterion= "entropy", max_depth = 5, min_samples_leaf = 4, min_samples_split= 2, random_state = 42)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy_score(y_test,y_pred)

dump(model,open("../models/tree_classifier_crit-entro_maxdepth-5_minleaf-4_minsplit2_42.sav", "wb"))