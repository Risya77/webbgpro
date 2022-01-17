
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from flask_mysqldb import MySQL
from MySQLdb.cursors import DictCursor

#from skimage import io, color, filters, util 
import cv2
import numpy as np
import sys
import cv2
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']=''
app.config['MYSQL_DB']='edetect'
#app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
mysql=MySQL(app)
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('pict.html')
 
@app.route('/pdetec', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        classifier =load_model(r'trained_model.h5')
        emotion_labels = ['angry','happy','neutral','sad']
        faceCascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

        image = cv2.imread('static/uploads/'+filename) #url_for('static', filename='uploads/' + filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
        flash('Found {0} faces!'.format(len(faces)))  
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            
            prediksi = cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            # print("[INFO] Object found. Saving locally.")
            gambar=cv2.imwrite('static/Hasil/faces.jpg', prediksi)
            gbr=cv2.imread ('static/Hasil/faces.jpg',0)
            
            
        smt = request.form['semester']
        mk = request.form['mk']
        jml=len(faces)
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO hsil VALUES (null, %s,%s,%s,%s)", (int(smt),str(mk),int(jml),str(gbr)))
        mysql.connection.commit()   
        return render_template('pict.html', filename=filename)
    
    
    
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
# @app.route('/display/<filename>')
# def display_image(filename):
#     #print('display_image filename: ' + filename)
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/hasil/<filename>')
def pdetect(filename):
    # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # filename='static/uploads/'+filename
    # img=ROOT_DIR + filename
    classifier =load_model(r'trained_model.h5')
    emotion_labels = ['angry','happy','neutral','sad']
    faceCascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

    image = cv2.imread('static/uploads/'+filename) #url_for('static', filename='uploads/' + filename))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    #flash('Found {0} faces!'.format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            #prediksi = cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            # print("[INFO] Object found. Saving locally.")
            # gambar=cv2.imwrite('static/Hasil/faces.jpg', image)
            # gbr=cv2.imread ('static/Hasil/faces.jpg',0)
            
            # smt = request.form['semester']
            # # mk = request.form['mk']
            # jml=len(faces)
            # cur = mysql.connection.cursor()
            # cur.execute("INSERT INTO hsil VALUES (null, null,null, null,%s)", (str(label)))
            # mysql.connection.commit()       
        else:
            cv2.putText(image,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
    
    cv2.imshow("Emotion Detection", image)
    cv2.waitKey(0)

 
if __name__ == "__main__":
    app.run()