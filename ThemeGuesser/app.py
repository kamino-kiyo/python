# ベース https://qiita.com/keimoriyama/items/7c935c91e95d857714fb
# 一部 https://qiita.com/redshoga/items/60db7285a573a5e87eb6

import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from keras import layers, models
import numpy as np
import os
from PIL import Image
import string
import tensorflow_hub as hub
from werkzeug.utils import secure_filename

from label import Label, binarization, SQUARE_LENGTH
from trim import trim_all_eyes


# 初期設定
app = Flask(__name__)

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'bmp'])

# 学習済みモデル
MODEL_NAME = "./model.h5"


# 拡張子の確認
def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 画像読み込み
def load_image(path):
    # cv2.imreadが日本語ファイル名に対応していないので、一旦Pillowで読み込んでから変換する
    # https://qiita.com/derodero24/items/f22c22b22451609908ee
    src = np.array(Image.open(path), dtype=np.uint8)
    if src.ndim == 2:  # モノクロ
        pass
    elif src.shape[2] == 3:  # カラー
        src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    elif src.shape[2] == 4:  # 透過
        src = cv2.cvtColor(src, cv2.COLOR_RGBA2BGRA)       
    return src


# ファイルのアップロード
@app.route('/', methods=['GET', 'POST'])
def upload():
    # 学習済みモデルの読み込み
    model = models.load_model(MODEL_NAME, custom_objects={"KerasLayer": hub.KerasLayer})
    
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
            
            # 画像読み込み
            src = load_image(UPLOAD_FOLDER + "/" + filename)
                
            # 片目ずつトリミング
            preds = np.empty(0)
            imgs = trim_all_eyes(src)
            print("imgs: " + str(len(imgs)))
            count = 0
            for img in imgs:
                split_path = os.path.splitext(UPLOAD_FOLDER + "/" + filename)
                save_path = split_path[0] + "_" + str(count) + split_path[1]

                # 保存
                cv2.imwrite(save_path, img)

                # リサイズ
                resize_img = np.array(Image.open(save_path).resize(size=(SQUARE_LENGTH,SQUARE_LENGTH)))
                
                # 判別
                pred = model.predict(np.array([resize_img / 255.]))
                preds = np.append(preds, pred)
                print("pred:")
                print(pred)
                
                count += 1
            
            print("-----------------------------")
            print("preds:")
            print(preds)
            print(binarization(preds))
            result = Label(binarization(preds)).name
            print(filename + ": " + result) 
            
            # 入力画像を判定結果を一緒に表示させるページに遷移
            return render_template('result.html', UPLOAD_FOLDER = UPLOAD_FOLDER, filename = filename, result = result)
        
    # GET時に表示されるページ
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