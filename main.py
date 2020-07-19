import pickle
from flask import Flask, render_template, request
app = Flask(__name__)


file= open(r'D:\Python\Machine Learning\Covid Detector\model.pkl', 'rb')
model= pickle.load(file)
file.close()

@app.route('/')
@app.route('/intro.html')
def index():
    #return f'Hello, World! {ProbInfc}%'
    return render_template('intro.html') 

@app.route('/home.html', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        print(request.form)
        DataDict= request.form
        BodyTemp= int(DataDict['BodyTemp'])
        Age= int(DataDict['Age'])
        BodyPain= int(DataDict['BodyPain'])
        DryCough= int(DataDict['DryCough'])
        DiffBreath= int(DataDict['DiffBreath'])
        inputFeature_tmp= [BodyTemp, BodyPain, DryCough, DiffBreath, Age]
        inputFeature= [inputFeature_tmp]
        ProbInfc = model.predict_proba(inputFeature)[0][1]
        ProbInfc*=100
        ProbInfc= round(ProbInfc, 2)
        return render_template('result.html', Probinf=ProbInfc)
    return render_template('home.html')

@app.route('/result.html')
def result():
    return render_template('result.html')

@app.route('/about.html')
def about_me():
    return render_template('about.html')    


if __name__ == "__main__":
    app.run(debug=True)
       