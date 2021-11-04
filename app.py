# -*- coding: utf-8 -*-
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/dashboard/')
def dashboard():
    return render_template("dashboard.html")
	
@app.route('/api/meteo/')
def meteo():
    pass # ligne temporaire

if __name__ == "__main__":
    app.run(debug=True)