# import os
import sqlite3
import hashlib
import secrets
import requests
# from minio import Minio
from flask import Flask, render_template, request, redirect, url_for, session, g

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.database = "users.db"

# TODO: Minio Service
"""
minio_access_key = os.getenv("MINIO_ACCESS_KEY")
minio_secret_key = os.getenv("MINIO_SECRET_KEY")
minio_client = Minio(
    "host", 
    access_key=minio_access_key,
    secret_key=minio_secret_key,
    secure=False
)
"""

# Login code based on: https://gist.github.com/jironghuang/24e0577e58844882604c0013407bf606


def init_db():
    with app.app_context():
        db = get_db()
        cur = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
        table_exists = cur.fetchone()
        if not table_exists:
            with app.open_resource('init.sql', mode='r') as f:
                db.cursor().executescript(f.read())
            db.commit()


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.database)
    return g.db


@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'db'):
        g.db.close()


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        cur = db.execute('SELECT password FROM users WHERE username = ?', [username])
        user = cur.fetchone()
        if user and user[0] == hashlib.sha256(password.encode()).hexdigest():
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('auth.html', mode='login', error='Invalid username or password')
    return render_template('auth.html', mode='login')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None

        cur = db.execute('SELECT id FROM users WHERE username = ?', [username])
        if cur.fetchone() is not None:
            error = 'Username is already taken.'

        if error is None:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)', [username, hashed_password])
            db.commit()
            return redirect(url_for('login'))

        return render_template('auth.html', mode='register', error=error)

    return render_template('auth.html', mode='register')


@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html')
    return redirect(url_for('login'))


@app.route('/submit', methods=['POST'])
def submit():
    if 'username' in session:
        input_text = request.form['text']
        min_len = int(request.form.get('min_len', 0))
        max_len = int(request.form.get('max_len', 128))
        beams = int(request.form.get('beams', 1))
        sample = request.form.get('sample', 'False').lower() == 'true'

        # 'http://127.0.0.1:5001/predict'
        response = requests.post('http://model_service:5001/predict', json={
            'text': input_text,
            'params': {
                'min_len': min_len,
                'max_len': max_len,
                'beams': beams,
                'sample': sample
            }
        })
        summary = response.json()['summary']
        summary = '\n'.join(['>' + chunk for chunk in summary.split('\n')])
        return render_template('index.html', input_text=input_text, output_text=summary)
    return redirect(url_for('login'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
