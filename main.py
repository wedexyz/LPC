
from flask import Flask, request
import librosa
import sounddevice as sd
import  threading, webbrowser
from flask import Flask, render_template, request, render_template
import numpy as np
import webbrowser
import sounddevice as sd
import librosa
import joblib
from lpc import *


offset=0.6
duration=3

app = Flask(__name__ ,template_folder='template')
def static_file(path):
    return app.send_static_file(path)

@app.route("/", methods=['GET', 'POST'])
def index():
    offset=0.6
    duration=3
    output = ''
    output2 = ''
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1 :
            data, sampling_rate =librosa.load(file1,duration=duration, offset=offset)
            sd.play(data, sampling_rate)
            lpccs = lpcc(data,16)
            joblib.dump(lpccs,'keluaran.pkl')
            output  = joblib.load('keluaran.pkl')
            return render_template('index.html',output=output,output2=output2 )
        
        elif file2:
            data2, sampling_rate = librosa.load(file2,duration=duration, offset=offset)
            sd.play(data2, sampling_rate)
            lpccs = lpcc(data2,16)
            joblib.dump(lpccs,'keluaran2.pkl')
            output  = joblib.load('keluaran.pkl')
            output2  = joblib.load('keluaran2.pkl')
            point1 = np.array(joblib.load('keluaran.pkl'))
            point2 = np.array(joblib.load('keluaran2.pkl'))
            dist = np.linalg.norm(point1 - point2)
            print(dist)
            return render_template('index.html',output=output,output2=output2 ,output3= dist)
        
        
    return render_template('index.html' )
   

if __name__ == "__main__":
    port = 8001#+ random.randint(0, 999)
    url = "http://127.0.0.1:{0}/".format(port)
    threading.Timer(1.5, lambda: webbrowser.open(url) ).start()
    app.run(host='127.0.0.1',threaded=False,port=8001)