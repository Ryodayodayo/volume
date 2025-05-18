from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
from app import process_audio

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#ファイルを保存するディレクトリ
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# フォルダがなかったら作る
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

#ルート設定
@app.route('/')
def home():
    return render_template('home.html')

#音声ファイルのアップロード
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'ファイルが選択されていません', 400
    
    file = request.files['file']

    if file.filename == '':
        return 'ファイルが選択されていません', 400
    
    else:
        #ファイルを保存
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        #処理を行う
        #処理後の音声を返す

        #processed/に保存するファイル名
        output_filename = "processed_" + file.filename
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)

        #取得した値を整理
        threshold = float(request.form['threshold'])
        ratio = float(request.form['ratio'])
        target_level = float(request.form['target_level'])


        #app.pyの関数を使って処理
        graph_filename = process_audio(filepath, output_path, threshold, ratio, target_level)

        # 結果ページへリダイレクト
        return redirect(url_for('result', filename=output_filename, graph_filename=graph_filename))

        """
        # 処理済みファイルを返す
        return send_from_directory(PROCESSED_FOLDER, output_filename, as_attachment=True)
        """
    
#結果ページ    
@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename, graph_filename = request.args.get('graph_filename'))

#ファイルダウンロード用のエンドポイント
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    
if __name__ == '__main__' :
    app.run(debug=True)