from flask import Flask, render_template,request, url_for, redirect
from flask_mysqldb import MySQL

app = Flask(__name__)
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']=''
app.config['MYSQL_DB']='edetect'
mysql = MySQL(app)

@app.route('/')
def home():
    cur = mysql.connection.cursor()
    cur.execute("SELECT*FROM dosen")
    data = cur.fetchall()
    cur.close()
    return render_template('home.html', dosen=data)

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
    return redirect(url_for('home'))

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
    return redirect(url_for('home'))

@app.route('/hapus/<string:id_data>', methods=["GET"])
def hapus(id_data):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM dosen WHERE id=%s", (id_data,))
    mysql.connection.commit()
    return redirect(url_for('home'))

@app.route('/about-us')
def aboutUs():
    return render_template('about-us.html')

@app.route('/contact-us')
def contactUs():
    return render_template('contact-us.html')

if __name__=='__main__':
    app.run(debug=True)