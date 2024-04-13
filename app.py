
from flask import Flask, render_template, request, session, redirect, url_for, Response, g
from flask_caching import Cache
import mysql.connector
import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tensorflow import keras
import pickle
from keras.models import load_model
import json
import random
from skimage import transform
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
import requests


config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'face_emotion',
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'febin123'
cache = Cache(app, config={'CACHE_TYPE': 'redis',
              'CACHE_REDIS_URL': 'redis://localhost:6379/0'})
UPLOAD_FOLDER = "static/uploads"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = load_model('models/cnn_new.h5')
cvn = pickle.load(open('models/cv.pkl', 'rb'))
model2 = keras.models.load_model('models/txt_emo.h5')

counter = 0
current_frame = None
query = {}
emo = []
emo_r = -1
emo_f = []
emo_t = []
songs = None


@app.route("/")
def homepage():
    return render_template('index.html')


# LOGIN
@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == 'POST':
        query = "select * from login where email='"+request.form['txt_email']+"'"
        res = select_records(query)
        print(res)
        if res != []:
            if res[0][1] == request.form['txt_pass']:
                session['user_id'] = res[0][0]
                if res[0][2] == 2:
                    return '''
                    <script>
                    alert('Welcome Back!');
                    window.location.href = '/user/';
                    </script>
                    '''
                else:
                    return '''
                    <script>
                    alert('Welcome Back Admin!');
                    window.location.href = '/admin/';
                    </script>
                    '''
            else:
                return '''
                <script>
                alert('Invalid credential!');
                window.location.href = '/';
                </script>
                '''
        else:
            return '''
                <script>
                alert('Email not registered!');
                window.location.href = '/';
                </script>
                '''
    else:
        return render_template('/login.html')
    
    
# REGISTER
@app.route("/register", methods=["POST", "GET"])
def register():
    if request.method == 'POST':
        txt_name = request.form['txt_name']
        txt_email = request.form['txt_email']
        txt_number = request.form['txt_number']
        txt_pass = request.form['txt_pass']
        txt_cpass = request.form['txt_cpass']

        # Check if passwords match
        if txt_pass != txt_cpass:
            return '''
                <script>
                alert('Passwords do not match!');
                window.location.href = '/register';
                </script>
                '''

        query = "select * from login where email='"+request.form['txt_email']+"'"
        res = select_records(query)
        print(res)
        if not res:
            query = "INSERT INTO user (name,email,contact) VALUES (%s,%s,%s)"
            data = (txt_name, txt_email, txt_number)
            insert_record(query, data)

            query = "INSERT INTO login (email,password) VALUES (%s,%s)"
            data = (txt_email, txt_pass)
            insert_record(query, data)

            return '''
                <script>
                alert('Registered successfully!');
                window.location.href = '/';
                </script>
                '''
        else:
            return '''
                <script>
                alert('Email already registered!');
                window.location.href = '/';
                </script>
                '''
    else:
        return render_template('index.html')


#logout   
@app.route('/user/logout')
def logout():
    if 'user_id' in session.keys():
        session.pop('user_id', None)
        return render_template('/index.html')
    else:
        return render_template('/index.html')

#admin
@app.route('/admin/') 
def admin_dashboard():
    if 'user_id' in session.keys() and session.get('user_id')=="admin@gmail.com":
        user_id = session.get('user_id')
        return render_template('admin/homepage.html', email=user_id)
    else:
        return render_template('/index.html')  

@app.route('/admin/users') 
def admin_users():
    if 'user_id' in session.keys() and session.get('user_id')=="admin@gmail.com":
        user_id = session.get('user_id')
        query = "select * from user"
        res = select_records(query)
        return render_template('admin/users.html', email=user_id, res=res)
    else:
        return render_template('/index.html')  

@app.route('/admin/music') 
def admin_music():
    if 'user_id' in session.keys() and session.get('user_id')=="admin@gmail.com":
        user_id = session.get('user_id')
        query = "select * from music"
        res = select_records(query)
        return render_template('admin/music.html', email=user_id, res=res)
    else:
        return render_template('/index.html')     

#user
@app.route('/user/') 
def user_dashboard():
    if 'user_id' in session.keys():
        user_id = session.get('user_id')
        query1 = "select name from user where email='" + user_id + "'"
        res = select_records(query1)
        return render_template('user/homepage.html', firstName=res[0][0],email=user_id)
        
    else:
        return render_template('/index.html')
    


@app.route('/user/chatbot') 
def chatbot():
    if 'user_id' in session.keys():
        user_id = session.get('user_id')
        query1 = "select name from user where email='" + user_id + "'"
        res = select_records(query1)
        return render_template('user/chatbot.html', firstName=res[0][0],email=user_id)
    else:
        return render_template('/index.html')

@app.route('/chat2', methods=['POST', 'GET'])
def chat2():
    global counter, current_frame, emo, emo_r, emo_f, emo_t, songs, model
    emo_f = []
    if request.method == 'POST':
        print("counter", counter)
        if counter > 2:
            
            for x in os.listdir('static/output'):
                pt = os.path.join(os.getcwd(), 'static/output', x)
                image = load(pt)
                pred = model.predict(image)
                c = np.argmax(pred)
                emo.append(c)
                emo_f.append(c)
            ecount = []
            if 0 in emo:
                ecount.append(emo.count(0))
            else:
                ecount.append(0)
            if 1 in emo:
                ecount.append(emo.count(1))
            else:
                ecount.append(0)
            if 2 in emo:
                ecount.append(emo.count(2))
            else:
                ecount.append(0)
            emo_r = np.argmax(ecount)
            print(emo_f)
            print(emo_t)
            print(emo)
            print(ecount)
            print(emo_r)
            # songs = search_song(emo_r)
            return url_for('result')
        else:
            output = os.path.join(os.getcwd(), 'static/output')
            pt = output+"/"+str(counter)+".png"
            print(pt)
            save_frame(current_frame, pt)
            counter += 1
            r = txt_to_emo(request.form['message'])
            emo.append(r)
            emo_t.append(r)
            print(emo_t)
            response = chatbot_res(request.form['message'])
            return response
    else:
        return render_template('index.html')
    
@app.route('/result')
def result():
    global emo_r, query, songs, emo_f, emo_t
    if songs == None:
        return render_template("index.html")
    else:
        output = os.path.join(os.getcwd(), 'static/output')
        dir_f = sorted(os.listdir(output))
        print(emo_r)
        return render_template("result.html", emo_r=emo_r, songs=songs, emo_f=emo_f, emo_t=emo_t, dir_f=dir_f)


def save_frame(frame, path):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        face_filepath = path
        cv2.imwrite(face_filepath, face_img)


# chatbot
def chatbot_res(msg):
    userText = msg
    data = json.dumps({"sender": "Rasa", "message": userText})
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post(
        'http://localhost:5005/webhooks/rest/webhook', data=data, headers=headers)
    response = response.json()
    return str(response[0]['text'])


# text to emotion


def txt_to_emo(txt):
    global cvn, model2
    txt = [preprocess(x) for x in [txt]]
    data_cv2 = cvn.transform(txt).toarray()
    preds = model2.predict(data_cv2)
    preds_class = np.argmax(preds)
    return preds_class

# cam utils
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def get_camera():
    with app.app_context():
        # Check if the camera has already been opened
        if 'camera' not in g:
            # Open the default camera (usually 0)
            g.camera = cv2.VideoCapture(0)

        return g.camera


def generate_frames():
    global current_frame
    camera = get_camera()
    while True:
        success, frame = camera.read()
        current_frame = frame
        if not success:
            break
        else:
            framef = detect_faces(frame)
            ret, buffer = cv2.imencode('.jpg', framef)
            framef = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + framef + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def preprocess(line):
    ps = PorterStemmer()
    # leave only characters from a to z
    review = re.sub('[^a-zA-Z]', ' ', line)
    review = review.lower()  # lower the text
    review = review.split()  # turn string into list of words
    # apply Stemming
    review = [ps.stem(word) for word in review if not word in stopwords.words(
        'english')]  # delete stop words like I, and ,OR   review = ' '.join(review)
    # trun list into sentences
    return " ".join(review)


# cnn

def load(filename):
    np_image = Image.open(filename)
    # np_image = np.array(np_image).astype('float32')/255
    np_image = np.array(np_image.convert('L')).astype('float32') / 255
    np_image = transform.resize(np_image, (48, 48, 1))
    np_image = np.expand_dims(np_image, axis=0)

    # np_image = transform.resize(np_image, (48, 48, 1))
    # np_image = img_to_array(np_image)
    # gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    # normalized_image = gray_image / 255.0
    # input_image = np.expand_dims(gray_image, axis=-1)
    # input_image = np.reshape(input_image, (1, 48, 48, 1))

    return np_image


# util functions

def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (128, 128, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

# mysql helper functions


def insert_record(query, data):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    cursor.execute(query, data)

    cnx.commit()
    cursor.close()
    cnx.close()


def update_record(query, data):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    cursor.execute(query, data)

    cnx.commit()
    cursor.close()
    cnx.close()


def select_records(query):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    cursor.execute(query)

    rows = cursor.fetchall()

    cursor.close()
    cnx.close()
    return rows


# @app.errorhandler(404)
# def page_not_found(error):
#     return render_template('404.html'),404
