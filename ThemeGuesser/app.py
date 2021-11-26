from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'hello'

app.run('0.0.0.0', port=5000, debug=True)