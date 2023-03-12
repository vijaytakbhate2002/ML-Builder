from flask import Flask,render_template,redirect,request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,BaggingClassifier,BaggingRegressor
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier 
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso
from model import Builder
import jinja2
import joblib

class_algos = [DecisionTreeClassifier(),LogisticRegression(),KNeighborsClassifier(),SVC(),RandomForestClassifier(),BaggingClassifier()]
reg_algos = [DecisionTreeRegressor(),KNeighborsRegressor(),LinearRegression(),Ridge(),Lasso(),SVR(),RandomForestRegressor(),BaggingRegressor()]
start = False

app = Flask(__name__)
df = None
@app.route("/")
def Home():
    return render_template("index.html")

def colshandle(lis):
    new_col = []
    for col in lis:
        new_col.append(col.strip())
    return new_col

@app.route('/upload', methods=['GET','POST'])
def upload():
    csvfile = request.files['datafile']
    if csvfile:
        global df
        df = pd.read_csv(csvfile)
        global cols
        cols = colshandle(df.columns)
        df.columns = cols
        return render_template("index.html",cols=cols,start=True)
    else:
        return redirect("/")

@app.route('/target_col/<string:col>')
def target_selector(col):
    global target_name
    target_name = col
    return render_template("index.html", tar_col=target_name)

def problem_type_finder(df,target_name):
    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(exclude=[np.number])
    if target_name in num_df:
        problem_type = "reg"
    else:
        problem_type = "class"
    return problem_type

@app.route('/build_model', methods=['GET','POST'])
def build_model():
    if request.method == 'POST':
        target_name = request.form['target_name']
        problem_type = problem_type_finder(df, target_name)
        if problem_type == 'class':
            algo = class_algos[0]
        else:
            algo = reg_algos[0]
        build = Builder(df,algo=algo,problem_type=problem_type,target_name=target_name,cross_scores=False)
        dic,model = build.fit_make()
        dum=joblib.dump(model,"static/mlmodel.pkl")
        dic["X_train"].to_csv("static/X_train.csv")
        dic['y_train'].to_csv("static/y_train.csv")
        dic['X_test'].to_csv("static/X_test.csv")
        dic['y_test'].to_csv("static/y_test.csv")

        del(dic["X_train"])
        del(dic["y_train"])
        del(dic["X_test"])
        del(dic["y_test"])
    return render_template("index.html",dictionary=dic,download_link=True)

@app.route("/customize", methods=['GET','POST'])
def customize():
    return "Customize"

if __name__ == "__main__":
    app.run(debug=True,port=8000)