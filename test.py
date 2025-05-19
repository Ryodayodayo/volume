from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
import os
from app import process_audio

app = Flask(__name__)
CORS(app)  

TEST_FOLDER = "test"
TEST_PROCEED_FOLDER = "test_proceed"
os.makedirs(TEST_FOLDER, exist_ok=True)
os.makedirs(TEST_PROCEED_FOLDER, exist_ok=True)

@app.route("/test", methods=["POST"])
def test_upload():
    normalize = float(request.form.get("normalize"))
    ratio = float(request.form.get("ratio"))
    threshold = float(request.form.get("threshold"))

    file = request.files["file"]
    filepath = os.path.join(TEST_FOLDER, file.filename)
    file.save(filepath)
    output_filename = "proceed_" + file.filename
    output_path = os.path.join(TEST_PROCEED_FOLDER, output_filename)
    file_url = f'/test_proceed/{output_filename}'
    graph_filename = process_audio(filepath, output_path, threshold, ratio, normalize)

    print(f"[DEBUG] 受け取ったファイル名: {file.filename}")
    print(f"[DEBUG] 受け取った数値: {normalize}")
    print(f"[DEBUG] 受け取った数値: {ratio}")
    print(f"[DEBUG] 受け取った数値: {threshold}")

    return jsonify({ "filename": file.filename, "file_url": file_url})

@app.route('/test_proceed/<output_filename>')
def get_processed_file(output_filename):
    return send_from_directory('test_proceed', output_filename)

@app.route('/result/<filename>')
def result():
    return 0


@app.route('/download/<filename>')
def download_file():
    return 0

if __name__ == "__main__":
    app.run(debug=True)