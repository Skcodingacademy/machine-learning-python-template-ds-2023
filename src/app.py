
from pickle import dump 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import db_connect
engine = db_connect()

# your code here



total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv",sep = ";")

total_data.head()

#saving raw data 
total_data.to_csv("../data/raw/total_data.csv", index = False)

#dealing with duplicates

total_data = total_data.drop_duplicates().reset_index(drop = True)

total_data.head()

#looking for nulls 
total_data.isnull().sum()

# min-max scaler 


total_data["job_n"]=pd.factorize(total_data["job"])[0]
total_data["marital_n"]=pd.factorize(total_data["marital"])[0]
total_data["education_n"]=pd.factorize(total_data["education"])[0]
total_data["default_n"]=pd.factorize(total_data["default"])[0]
total_data["housing_n"]=pd.factorize(total_data["housing"])[0]
total_data["loan_n"]=pd.factorize(total_data["loan"])[0]
total_data["contact_n"]=pd.factorize(total_data["contact"])[0]
total_data["month_n"]=pd.factorize(total_data["month"])[0]
total_data["day_of_week_n"]=pd.factorize(total_data["day_of_week"])[0]
total_data["poutcome_n"]=pd.factorize(total_data["poutcome"])[0]
total_data["y_n"]=pd.factorize(total_data["y"])[0]
num_variables= ["job_n","marital_n","education_n","default_n","housing_n","loan_n","contact_n","month_n","day_of_week_n","poutcome_n",
                "age","duration","campaign","pdays","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y_n"]

scaler=MinMaxScaler()
scal_features= scaler.fit_transform(total_data[num_variables])
total_data_scal=pd.DataFrame(scal_features,index=total_data.index,columns=num_variables)
total_data_scal.head()


x = total_data_scal.drop("y_n",axis=1)
y = total_data_scal["y_n"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)

selection_model= SelectKBest(chi2, k=5)
selection_model.fit(x_train,y_train)
ix=selection_model.get_support()
x_train_sel=pd.DataFrame(selection_model.transform(x_train),columns=x_train.columns.values[ix])
x_test_sel=pd.DataFrame(selection_model.transform(x_test),columns=x_test.columns.values[ix])


x_train_sel.head()


x_train_sel["y_n"]=list(y_train)
x_test_sel["y_n"]=list(y_test)

x_train_sel.to_csv("../data/processed/clean_train.csv", index=False)
x_test_sel.to_csv("../data/processed/clean_test.csv", index=False)

# Logistic Regression MOdel

train_data=pd.read_csv("../data/processed/clean_train.csv")
test_data=pd.read_csv("../data/processed/clean_test.csv")


x_train=train_data.drop(["y_n"],axis=1)
y_train=train_data["y_n"]
x_test=test_data.drop(["y_n"], axis=1)
y_test=test_data["y_n"]

model= LogisticRegression()
model.fit(x_train,y_train)

y_pred =model.predict(x_test)
y_pred

accuracy_score(y_test,y_pred)

#Model optimization


hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1","l2","elasticnet",None],
    "solver": ["newton-cg", "lbfgs","liblinear", "sag", "saga"]
}

grid=GridSearchCV(model,hyperparams,scoring="accuracy",cv = 10)

grid

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn=warn

grid.fit(x_train, y_train)

print(f"Best hyperparameters:{grid.best_params_}")

model=LogisticRegression(C=0.1, penalty="l2", solver="liblinear")
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_pred

accuracy_score(y_test,y_pred)


dump(model, open("../models/logistic_regression_C-0.1_penalty-l2_solver-liblinear_42.sav", "wb"))