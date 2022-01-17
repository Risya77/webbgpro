import os
from flask_cors import CORS
from flask import Flask,request, render_template, jsonify,redirect,session,flash,url_for
from functools import wraps
from chat import get_response
from flask_mysqldb import MySQL
from MySQLdb.cursors import DictCursor
#deteksi muka by pic
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
#deteksi muka real time
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import urllib.request
from werkzeug.utils import secure_filename
 


app=Flask(__name__,template_folder="templates")

 
#UPLOAD_FOLDER = 'application\static\uploads\'

CORS(app)
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']=''
app.config['MYSQL_DB']='edetect'
#app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.config['UPLOAD_FOLDER'] = 'D:/EmotionDetect/bgproject/application/static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
mysql=MySQL(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET'])

####### LOGIN #########
@app.route('/login',methods=['POST','GET'])
def login():
    
    status=True
    if request.method=='POST':
        email=request.form["email"]
        pwd=request.form["upass"]
        cur=mysql.connection.cursor(DictCursor)
        cur.execute("select * from users where email=%s and password=%s",(email,pwd))
        data=cur.fetchone()
        if data:
            session['logged_in']=True
            session['username']=data["name"]
            flash('Login Successfully','success')
            return redirect('index')
        else:
            flash('Invalid Login. Try Again','danger')
    return render_template("login.html")
  
#check if user logged in
def is_logged_in(f):
	@wraps(f)
	def wrap(*args,**kwargs):
		if 'logged_in' in session:
			return f(*args,**kwargs)
		else:
			flash('Unauthorized, Please Login','danger')
			return redirect(url_for('login'))
	return wrap
  

@app.route("/index")
def index():
    return render_template("index.html")

#logout
@app.route("/logout")
@is_logged_in
def logout():
	session.clear()
	flash('You are now logged out','success')
	return redirect(url_for('login'))
    

###### DATA DOSEN ########
@app.route("/data",methods=['GET'])
@is_logged_in
def data():
    cur = mysql.connection.cursor()
    cur.execute("SELECT*FROM dosen")
    row = cur.fetchall()
    cur.close()
    return render_template('data_dosen.html', dosen=row)

@app.route('/simpan',methods=["POST"])
def simpan():
    nip = request.form['nip']
    nama = request.form['nama']
    email = request.form['email']
    smt = request.form['semester']
    mk = request.form['mk']
    password = request.form['password']   
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO dosen VALUES (null, %s, %s,%s, %s,%s, %s)", (str(nip), str(nama),str(email), str(smt),str(mk), str(password)))
    mysql.connection.commit()
    return redirect(url_for('data'))

@app.route('/update', methods=["POST"])
def update():
    id_data = request.form['id']
    nama = request.form['nama']
    email = request.form['email']
    smt = request.form['semester']
    mk = request.form['mk']
    password = request.form['password']
    cur = mysql.connection.cursor()
    cur.execute("UPDATE dosen SET nama=%s,email=%s,semester=%s,mk=%s,password=%s WHERE id=%s", (nama,email,smt,mk,password, id_data))
    mysql.connection.commit()
    return redirect(url_for('data'))

@app.route('/hapus/<string:id_data>', methods=["GET"])
def hapus(id_data):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM dosen WHERE id=%s", (id_data,))
    mysql.connection.commit()
    return redirect(url_for('data'))

###### DATA User ########

###### DATA Hasil ########

###### Deteksi Open Kamera ########
@app.route("/detection", methods=['GET','POST'])

def detection():
    face_classifier = cv2.CascadeClassifier(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\haarcascade_frontalface_default.xml')
    classifier =load_model(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\trained_model.h5')
    emotion_labels = ['angry','happy','neutral','sad']
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render_template("index.html")
###### End Deteksi Open Kamera ########
   

###### Deteksi Get Picture ########
@app.route('/pict')
def pict():
    return render_template('pict.html')


@app.route("/pdetec", methods=['POST'])
@is_logged_in
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
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        #print('upload_image filename: ' + filename)
        classifier =load_model(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\trained_model.h5')
        emotion_labels = ['angry','happy','neutral','sad']
        faceCascade = cv2.CascadeClassifier(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\haarcascade_frontalface_default.xml')

        image = cv2.imread('D:/EmotionDetect/bgproject/application/static/uploads/'+filename) #url_for('static', filename='uploads/' + filename))
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
                cv2.imwrite('D:/EmotionDetect/bgproject/application/Hasil/faces.jpg', prediksi)
        gbr=cv2.imread ('D:/EmotionDetect/bgproject/application/Hasil/faces.jpg',0)    
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

@app.route('/hasil/<filename>')
def pdetect(filename):
    # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # filename='static/uploads/'+filename
    # img=ROOT_DIR + filename
    classifier =load_model(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\trained_model.h5')
    emotion_labels = ['angry','happy','neutral','sad']
    faceCascade = cv2.CascadeClassifier(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\haarcascade_frontalface_default.xml')

    image = cv2.imread('D:/EmotionDetect/bgproject/application/static/uploads/'+filename) #url_for('static', filename='uploads/' + filename))
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


# def pdetec():
    
#     classifier = load_model(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\trained_model.h5')
#     detector = cv2.CascadeClassifier(r'D:\EmotionDetect\bgproject\pendukung\checkpoint\haarcascade_frontalface_default.xml')
#     class_labels = ['angry','happy','neutral','sad']

#     class GUI(tk.Frame):

#         def __init__(self, master=None):
#             tk.Frame.__init__(self, master)
#             w,h = 500, 400
#             master.minsize(width=w, height=h)
#             master.maxsize(width=w, height=h)
#             self.pack()
            
#             self.file = tk.Button(self, text='Browse', command=self.choose)
#             self.choose = tk.Label(self, text="Choose file").pack()
            
#             self.imageFile = Image.open("R.png")
#             self.image = ImageTk.PhotoImage(self.imageFile)
#             self.label = tk.Label(image=self.image)
#             fontStyle = tkFont.Font(family="Lucida Grande", size=20)
#             self.emotion = tk.Label(self, text="Detected emotion", font=fontStyle)
        
#             self.file.pack()
#             self.label.pack()
#             self.emotion.pack(side = tk.BOTTOM)

#         def choose(self):
#             self.cur = mysql.connection.cursor()
#             self.filename = filedialog.askopenfilename(initialdir = "/",
#                                               title = "Select a File",
#                                               filetypes = [("Images",".jpg .jpeg .png")]
#                                               )
#             #imageFile = Image.open(self.filename)
#             #wpercent = (300/float(imageFile.size[0]))
#             #hsize = int((float(imageFile.size[1])*float(wpercent)))
#             #imageFile = imageFile.resize((300,hsize), Image.ANTIALIAS)
            
#             converted = cv2.imread(self.filename)
#             gray = cv2.cvtColor(converted,cv2.COLOR_BGR2GRAY)
#             faces = detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
#             self.emotion.configure(text='Found {0} faces!'.format(len(faces)))
            

            
#             for (x,y,w,h) in faces:
#                 cv2.rectangle(converted,(x,y),(x+w,y+h),(0, 255, 255), 2)
#                 roi_gray = gray[y:y+h,x:x+w]
#                 roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

#                 if np.sum([roi_gray])!=0:
#                     roi = roi_gray.astype('float')/255.0
#                     roi = img_to_array(roi)
#                     roi = np.expand_dims(roi,axis=0)
                    
#                     preds = classifier.predict(roi)[0]
#                     label1=class_labels[preds.argmax()]
#                     label_position = (x,y)
#                     cv2.putText(converted, label1, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#                     prediksi = cv2.putText(converted, label1, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#                     print("[INFO] Object found. Saving locally.")
#                     gambar=cv2.imwrite('application\static\img\hasil\faces.jpg', prediksi)
#                     gbr=cv2.imread ('application\static\img\hasil\faces.jpg',0)
#                     jml=len(faces)
#                     # happy=len(class_labels[preds.argmax()]=='happy')
#                     # sad=len(class_labels[preds.argmax()]=='sad')
#                     # angry=len(class_labels[preds.argmax()]=='angry')
#                     # neutral=len(class_labels[preds.argmax()]=='neutral')
#                     cur = mysql.connection.cursor()
#                     cur.execute("INSERT INTO hasil VALUES (null, %s, %s, %s)", (str(gbr),int(jml),str(label1)))
#                     mysql.connection.commit()
                
                    
#                     self.hasil = cv2.imshow("hasil",converted)
#                     self.image = ImageTk.PhotoImagePIL.Image.fromarray(self.hasil)
#                     self.label.configure(image=self.hasil)
#                     self.label.image=self.hasil
#                     # self.jml=print('terdapat {0} anak'.format(len(faces)))
#                     # self.gambar=cv2.imwrite('hasil.jpg',converted)
#                     # self.happy=print('terdapat {0} anak yang senang'.format(len(class_labels[preds.argmax()]=='happy')))
#                     # self.sad=print('terdapat {0} anak yang sedih'.format(len(class_labels[preds.argmax()]=='sad')))
#                     # self.angry=print('terdapat {0} anak yang marah'.format(len(class_labels[preds.argmax()]=='angry')))
#                     # self.neutral=print('terdapat {0} anak yang netral'.format(len(class_labels[preds.argmax()]=='neutral')))
#                     # self.cur.execute("INSERT INTO hasil VALUES (null, %s, %s, %s, %s, %s, %s)", (str(self.gambar),str(self.jml), str(self.happy),str(self.sad),str(self.angry),str(self.neutral)))
#                     # mysql.connection.commit()
                    
#                 else:
#                     self.emotion.configure(text='No face detected')
                    

#     root = tk.Tk()
#     root.title('Emotion detector')
#     app = GUI(master=root)
#     app.mainloop()
#     root.quit()
#     #return redirect("/simpan")
#     return render_template("index.html")

###### End Deteksi Get Picture ######## 


    

###### Chatboot ######## 
@app.route("/chatbot", methods=['GET','POST'])
def chatbot():
    return render_template ("base.html")

@app.route("/predict", methods=['GET','POST'])
def predict():
    text =  request.get_json().get('message')
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)
###### End Chatboot ######## 

@app.route("/about", methods=['GET'])
def about():
    return render_template ("about.html")



if __name__ == '__main__':
    app.secret_key='secret123'
    app.run(host='0.0.0.0', port=5000, debug=True)
