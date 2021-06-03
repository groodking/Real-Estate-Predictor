import numpy as np
from flask import Flask,render_template,request
import pickle
app=Flask(__name__)
model=pickle.load(open('rfreg.pkl','rb'))

@app.route('/')
def home():
    return render_template("Estate.html")

@app.route('/predict',methods=['POST'])
def predict():
    features=[float(x) for x in request.form.values()]
    final_feat=[np.array(features)]
    predection=model.predict(final_feat)
    output=predection[0]
    return render_template("Estate.html",pred_text="$ {}".format(output))
if __name__=="__main__":
    app.run(debug=True)