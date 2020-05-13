from flask import Flask
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
