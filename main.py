from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
import pickle
app=Flask(__name__)
import pandas as pd

@app.route("/",methods=['POST','GET'])
@cross_origin()
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST','GET'])
@cross_origin()
def submit1():
    #HTML---> .py
    if request.method == "POST":
        try:
            rm=float(request.form["rm"])
            ptratio=float(request.form["ptratio"])
            lstat=float(request.form["lstat"])
            indus=float(request.form["indus"])
            #df = pd.DataFrame([rm, lstat, ptratio, indus])
            filename = 'Regression_Assignment_model.pickle'
            loadedmodel = pickle.load(open(filename, "rb"))
            predicted_price = loadedmodel.predict([[rm, lstat, ptratio, indus]])
            print('prediction is', predicted_price)
            #return render_template('results.html', prediction=round(100 * predicted_price[0]))
            return render_template('results.html', prediction=predicted_price[0])
            #6.57 4.98 15.3 2.31
        except Exception as e:
            print('Exception is ',e)
            return 'Something is wrong'
    else:
        #.py-->html
         return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)