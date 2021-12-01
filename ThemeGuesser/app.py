# https://qiita.com/keimoriyama/items/7c935c91e95d857714fb
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
from image_process import canny
from datetime import datetime
import os
import string
import random
from werkzeug.utils import secure_filename


# 初期設定
app = Flask(__name__)

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])


# 拡張子の確認
def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ファイルのアップロード
@app.route('/', methods=['GET', 'POST'])
def upload():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        # ファイルのチェック
        if file and allwed_file(file.filename):
            # 危険な文字を削除（サニタイズ処理）
            filename = secure_filename(file.filename)
            # ファイルの保存
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # アップロード後のページに転送
            return redirect(url_for('uploaded_file', filename=filename))

    return '''
<html>
    <body>
        <form method = post enctype = multipart/form-data>
            <input type=file name = file>
            <input type = submit value = Upload>
        </form>
    </body>
</html>
'''
    
    
# アップロードされたファイルの処理
@app.route('/uploads/<filename>')
# ファイルを表示する
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)