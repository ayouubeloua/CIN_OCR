import argparse
import traceback
from flask import Flask, request, jsonify
import os
import normal_ocr
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, instance_relative_config=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
HOST_PORT = 5006


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[len(filename.rsplit('.', 1)) - 1].lower() in ALLOWED_EXTENSIONS


@app.route('/magic_link', methods=['POST'])
def magic_link():
    if request.method != 'POST':
        return 'Forbidden'
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            extension = file.filename.split('.')[1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'card.' + extension))
            return jsonify(normal_ocr.do_magic('uploads/card.' + extension))


def main():
    parser = argparse.ArgumentParser(description='Id Card Recognition')
    parser.add_argument('-p', '--port',type=int, default=HOST_PORT, help='port number')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    args = parser.parse_args()
    if args.debug:
        app.debug = True
    app.run(host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    try:
        main()
    except:
        traceback.print_exc()
