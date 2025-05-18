from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  

TEST_FOLDER = "test"
os.makedirs(TEST_FOLDER, exist_ok=True)

@app.route("/test", methods=["POST"])
def test_upload():
    normalize = request.form.get("normalize")

    file = request.files["file"]
    file.save(os.path.join(TEST_FOLDER, file.filename))
    print(f"[DEBUG] 受け取ったファイル名: {file.filename}")
    print(f"[DEBUG] 受け取った数値: {normalize}")

    return jsonify({"message": "ノーラマイズ数値とファイルを保存しました"}),200

if __name__ == "__main__":
    app.run(debug=True)