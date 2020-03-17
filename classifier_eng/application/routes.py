from flask import Flask
from flask import request, jsonify
from spam_classifier import classify
from application import app
app = Flask(__name__)

@app.route('/hello_user', methods=['POST'])
def hello_user():
    data = request.json
    user = data['user']
    return f'hello, {user}'

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/classify_text', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get('text') 
#Метод возвращает None, если запрашиваемого ключа нет
    if text is None:
        params = ', '.join(data.keys()) 
#Преобразуем все полученные параметры в строку
        return jsonify({'message': f'Parametr "{params}" is invalid'}), 400 
#Ранее мы не указывали код ответа HTTP явно,
#но на самом деле Flask выполнял эту работу за нас. 
#По умолчанию возвращается 200
    else:
        result = classify(text)
        return jsonify({'result': result})
