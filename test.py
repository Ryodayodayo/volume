from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
import os
from app import process_audio
from testApp import process_audio_advanced

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
    attack = float(request.form.get("attack"))
    release = float(request.form.get("release"))
    knee=5

    vocal = request.files["vocal"]
    vocal_filepath = os.path.join(TEST_FOLDER, vocal.filename)
    vocal.save(vocal_filepath)
    output_vocal_filename = "proceed_" + vocal.filename
    output_vocal_path = os.path.join(TEST_PROCEED_FOLDER, output_vocal_filename)
    vocal_file_url = f'/test_proceed/{output_vocal_filename}'
    previous_vocal_file_url = f'/test/{vocal.filename}'
    #graph_filename = process_audio(filepath, output_path, threshold, ratio, normalize)

    
    graph_filename = process_audio_advanced(vocal_filepath, output_vocal_path, threshold, ratio,attack, release,knee, normalize)

    image_url = f'/static/{graph_filename}'

    print(f"[DEBUG] 受け取ったファイル名: {vocal.filename}")
    print(f"[DEBUG] 受け取った数値: {normalize}")
    print(f"[DEBUG] 受け取った数値: {ratio}")
    print(f"[DEBUG] 受け取った数値: {threshold}")

    return jsonify({ 
        "vocal" : {
            "filename": vocal.filename,
            "file_url": vocal_file_url,
            "image_url" : image_url,
            "previous_file_url": previous_vocal_file_url
        }
     })

@app.route('/test/<filename>')
def get_original_file(filename):
    return send_from_directory("test", filename)

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