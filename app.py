#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
import joblib


# In[2]:


app = Flask(__name__, template_folder = "templates")
# "__" to get main


# In[3]:


from flask import request, render_template
from keras.models import load_model
from sklearn import preprocessing


@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        print(income, age, loan)
        logreg = joblib.load("logreg")
        cart = joblib.load("cart")
        randomforest = joblib.load("randomforest")
        xgb = joblib.load("xgb")
        nn = joblib.load("nn")
        pred_logreg = logreg.predict([[float(income),float(age),float(loan)]])
        pred_cart = cart.predict([[float(income),float(age),float(loan)]])
        pred_randomforest = randomforest.predict([[float(income),float(age),float(loan)]])
        pred_xgb = xgb.predict([[float(income),float(age),float(loan)]])
        input_data = [[float(income),float(age),float(loan)]]
        normalize = joblib.load("normalize")
        input_scaled = normalize.transform(input_data)
        print(input_scaled)
        pred_nn = nn.predict(input_scaled)
        s_logreg = "Default" if pred_logreg > 0 else "Not Default"
        s_cart = "Default" if pred_cart > 0 else "Not Default"
        s_randomforest = "Default" if pred_randomforest > 0 else "Not Default"
        s_xgb = "Default" if pred_xgb > 0 else "Not Default"
        s_nn = "Default" if pred_nn > 0 else "Not Default"
        s = "According to the model the customer will: Logistic Regression - " + s_logreg + ", CART - " + s_cart + ", RandomForest - " + s_randomforest + ", XGBoost - " + s_xgb + ", Neural Net - "+ s_nn  
        return(render_template("index.html", result=s))
    else:
        return(render_template("index.html", result="2"))


# In[ ]:


if __name__ == "__main__":
    app.run()

# in cloud, this means: if this is my file, then run


# In[ ]:





# In[ ]:




