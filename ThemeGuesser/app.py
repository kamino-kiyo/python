import sys
from flask import Flask


sys.path.append('ThemeGuesser')
import trim


app = Flask(__name__)

@app.route('/')
def hello():
    return 'trim!'

app.run('0.0.0.0', port=5000, debug=True)