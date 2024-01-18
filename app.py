#-*- coding=utf-8 -*-
from flask import Flask,request,render_template #อิมพอร์ตโมดูล
app = Flask(__name__)

import ProgramSum as model


title = ''
news_th =''

# app = Flask(__name__,static_folder = '/static')

# @app.route('/')
# def index():  #def  เป็นคำสำคัญสำหรับการสร้างฟังก์ชัน
#    return render_template('index.html') #เรนเดอร์ไฟล์ที่ชื่อ index ที่อยู่ในไดเร้กทอรี่ที่ชื่อ templates


# if __name__ == '__main__':
#    app.run()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        title = request.form['title']
        news_th = request.form['text']
        
        re = model.result(title,news_th)
        return render_template('index.html', sentiment=re)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)