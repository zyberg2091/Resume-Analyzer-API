
import pickle

import os
import numpy as np
from flask import jsonify
from src.source import Feature_Matrix, Resume_Extractor, patterns
from flask import Flask, render_template

app=Flask(__name__, template_folder='template')


def process_file(resumelist):
    features,matcher=patterns()  #examples
    resume_obj=Resume_Extractor(resumelist)
    matches,doclist=resume_obj(matcher)


    arr_obj=Feature_Matrix(len(resumelist),len(features))
    x_data,y_data=arr_obj.feature_gen(matches,doclist,features)

    return x_data,y_data


UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ALLOWED_EXTENSIONS = {'pdf'}
# def allowed_file(filename):
#    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/uploader',methods=['POST'])
def upload_file():
    
    f = request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))

    return 'file uploaded successfully'





@app.route("/predict",methods=['GET'])
def predict():
    response={}
    resumelist=[os.path.join(UPLOAD_FOLDER,file) for file in os.listdir(UPLOAD_FOLDER)]
    x_data,y_data=process_file(resumelist)

    y_pred,class_names,scores=[],[],[]
    with open('assets/normalizer.pkl','rb') as f:
        normalizer=pickle.load(f)

    x_data=normalizer.transform(x_data)
    
    with open('assets/encoder.pkl','rb') as f:
        encoder=pickle.load(f)

    with open('models/classification_model.pkl','rb') as f:
        class_model=pickle.load(f)

    with open('models/regression_model.pkl','rb') as f:
        reg_model=pickle.load(f)

    
    y_pred.append([class_model.predict(x_data),reg_model.predict(x_data)])
    classes=y_pred[0][0]
    score_val=np.ravel(y_pred[0][1])


    for class_val,score in zip(classes,score_val):
        class_names.append(encoder.inverse_transform([class_val]))
        scores.append(score)

    for name,label,score in zip(resumelist,class_names,scores):
        response[name] = [str(label[0]),str(score*100)]


    return jsonify(response)

    

if __name__=='__main__':
    app.run(debug=False)



    





