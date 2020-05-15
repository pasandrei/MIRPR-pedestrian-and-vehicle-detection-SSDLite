from flask import request, render_template, jsonify
from flask import Flask
from flask_cors import CORS
import numpy as np
import cv2
from custom_inference import run
import json
from utils.box_computations import corners_to_wh
import time

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
inferer = run.Custom_Infernce()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('file-upload.html')


@app.route('/process_image', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp

    files = request.files.getlist('files[]')
    nms_thresh, conf, device = float(request.form['nms_thresh']), float(request.form['conf_thresh']), request.form['device']
    print(nms_thresh, conf, device)

    errors = {}
    success = False

    file = files[0]
    if file and allowed_file(file.filename):
        filestr = file.read()
        npimg = np.fromstring(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        start = time.time()
        boxes = inferer.run_inference(img, modify_image=False,
                                      custom_settings=(nms_thresh, conf, device))
        boxes = corners_to_wh(boxes)
        total = time.time() - start
        total = "{:.3f}".format(total)

        json_dump = json.dumps({'boxes': boxes}, cls=NumpyEncoder)
        boxes = json.loads(json_dump)

        success = True
    else:
        errors[file.filename] = 'File type is not allowed'

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 206
        return resp
    if success:
        resp = jsonify({'data': boxes, 'time_taken': total})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    app.run()
