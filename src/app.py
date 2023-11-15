import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import Lasso
from pickle import dump
from utils import db_connect
engine = db_connect()

# your code here
total_data=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv")

total_data=total_data.drop_duplicates().reset_index(drop = True)

data_types = total_data.dtypes
numeric_columns=[c for c in list (data_types[data_types != "object"].index) if c != "Heart disease_number"]

scaler = StandardScaler()
norm_features = scaler.fit_transform(total_data[numeric_columns])

total_data_scal = pd.DataFrame(norm_features , index=total_data.index, columns= numeric_columns)
total_data_scal["Heart disease_number"]= total_data["Heart disease_number"]

x= total_data_scal.drop(columns=["Heart disease_number"])
y= total_data_scal["Heart disease_number"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

train_indices=list(x_train.index)
test_indices=list(x_test.index)

k= int(len(x_train.columns)* 0.3)
selection_model= SelectKBest(score_func=f_regression, k=k)
selection_model.fit(x_train,y_train)
ix= selection_model.get_support()

x_train_sel = pd.DataFrame(selection_model.transform(x_train), columns=x_train.columns.values[ix])
x_test_sel=pd.DataFrame(selection_model.transform(x_test), columns=x_test.columns.values[ix])

x_train_sel["Heart disease_number"]= list(y_train)
x_test_sel["Heart disease_number"]=list(y_test)

x_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
x_test_sel.to_csv("../data/processed/clean_test.csv", index = False)

total_data= pd.concat([x_train_sel,x_test_sel])

# Logistic Regression model

train_data=pd.read_csv("../data/processed/clean_train.csv")
test_data=pd.read_csv("../data/processed/clean_test.csv")

x_train=train_data.drop(["Heart disease_number"], axis =1 )
y_train = train_data["Heart disease_number"]

x_test=test_data.drop(["Heart disease_number"], axis= 1 )
y_test= test_data["Heart disease_number"]

model= LogisticRegression()
model.fit(x_train,y_train)

print(f"intercep (a):{model.intercept_}")
print(f"coefficemnts : {model.coef_}")

y_pred = model.predict(x_test)

print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R2 score: {r2_score(y_test, y_pred)}")

# Model optimization

alpha = 1.0
Lasso_model = Lasso(alpha= alpha)

Lasso_model.fit(x_train,y_train)

score=Lasso_model.score(x_test,y_test)
print("coefficents:", Lasso_model.coef_)
print("R2 score:", score)

dump(Lasso_model, open("../models/lasso_alpha-1.0.sav", "wb"))

