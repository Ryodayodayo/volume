from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
import os
from testApp import process_audio_advanced, process_mastering_audio, process_mix_audio
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  

TEST_FOLDER = "test"
TEST_PROCESSED_FOLDER = "test_processed"
BEFORE = "before"
os.makedirs(TEST_FOLDER, exist_ok=True)
os.makedirs(TEST_PROCESSED_FOLDER, exist_ok=True)
os.makedirs(BEFORE, exist_ok=True)

@app.route("/")
def home():
    return "こんにちは"

@app.route("/test", methods=["POST"])
def test_upload():
    logging.info("データ受け取った")

    normalize = float(request.form.get("normalize"))
    ratio = float(request.form.get("ratio"))
    threshold = float(request.form.get("threshold"))
    attack = float(request.form.get("attack"))
    release = float(request.form.get("release"))
    knee=5

    vocal = request.files["vocal"]
    vocal_filepath = os.path.join(TEST_FOLDER, vocal.filename)
    vocal.save(vocal_filepath)
    output_vocal_filename = "processed_" + vocal.filename
    output_vocal_path = os.path.join(TEST_PROCESSED_FOLDER, output_vocal_filename)
    vocal_file_url = f'/test_processed/{output_vocal_filename}'
    previous_vocal_file_url = f'/test/{vocal.filename}'
    #graph_filename = process_audio(filepath, output_path, threshold, ratio, normalize)

    logging.info(f"受け取ったファイル名: {vocal.filename}")

    
    graph_filename = process_audio_advanced(vocal_filepath, output_vocal_path, threshold, ratio,attack, release,knee, normalize)

    image_url = f'/static/{graph_filename}'

    logging.info("グラフ画像URL発行")

    print(f"[DEBUG] 受け取ったファイル名: {vocal.filename}")
    print(f"[DEBUG] 受け取った数値: {normalize}")
    print(f"[DEBUG] 受け取った数値: {ratio}")
    print(f"[DEBUG] 受け取った数値: {threshold}")

    inst = request.files["inst"]
    inst_filepath = os.path.join(TEST_FOLDER, inst.filename)
    inst.save(inst_filepath)
    output_inst_filename = "processed_" + inst.filename
    output_inst_path = os.path.join(TEST_PROCESSED_FOLDER, output_inst_filename)
    inst_file_url = f"/test_processed/{output_inst_filename}"

    logging.info("instの音声処理開始")

    #instの音声処理
    process_mastering_audio(inst_filepath, output_inst_path, normalize)

    logging.info("instの音声処理完了")

    # ミックス処理
    output_mix_filename = "mix_" + vocal.filename
    output_mix_path = os.path.join(TEST_PROCESSED_FOLDER, output_mix_filename)
    mix_file_url = f"/test_processed/{output_mix_filename}"

    process_mix_audio(output_inst_path, output_vocal_path, output_mix_path, vocal_ratio=1.0, inst_ratio=1.0, offset_ms = 0)

    logging.info("mix処理完了")



    return jsonify({ 
        "vocal" : {
            "filename": output_vocal_filename,
            "file_url": vocal_file_url,
            "file_path" : output_vocal_path,
            "image_url" : image_url,
            "previous_file_url": previous_vocal_file_url
        },
        "inst" : {
            "filename" : inst.filename,
            "file_path" : inst_filepath,
            "file_url" : inst_file_url,
        },
        "mix" : {
            "filename": output_mix_filename,
            "file_path" : output_mix_path,
            "file_url": mix_file_url,
        }
     })


@app.route('/test/<filename>')
def get_original_file(filename):
    return send_from_directory("test", filename)

@app.route('/test_processed/<output_filename>')
def get_processed_file(output_filename):
    return send_from_directory('test_processed', output_filename)

@app.route('/test_processed/<filename>')
def get_mix_file(filename):
    return send_from_directory('test_processed', filename)

@app.route('/result/<filename>')
def result():
    return 0

@app.route('/api/apply_mix', methods=["POST"])
def apply_volume_mix():
    output_inst_path = request.form.get('output_inst_path')
    output_vocal_path = request.form.get('output_vocal_path')
    output_mix_path = request.form.get('output_mix_path')
    vocal_ratio = float(request.form.get('vocal_gain'))
    inst_ratio = float(request.form.get('inst_gain'))
    offset_ms = float(request.form.get('offset_ms'))
    

    # 音量を適用してミックスし、ファイル保存
    process_mix_audio(output_inst_path, output_vocal_path, output_mix_path, vocal_ratio, inst_ratio, offset_ms)
    
    return jsonify({"message": "音量バランス、オフセット受け取り完了"})


@app.route('/download/<filename>')
def download_file():
    return 0

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
