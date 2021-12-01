# ベース https://qiita.com/keimoriyama/items/7c935c91e95d857714fb
# 一部 https://qiita.com/redshoga/items/60db7285a573a5e87eb6

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
import os
from PIL import Image
import string
from werkzeug.utils import secure_filename

from trim import trim_all_eyes


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
            
            # cv2.imreadが日本語ファイル名に対応していないので、一旦Pillowで読み込んでから変換する
            # https://qiita.com/derodero24/items/f22c22b22451609908ee
            src = np.array(Image.open(UPLOAD_FOLDER + "/" + filename), dtype=np.uint8)
            if src.ndim == 2:  # モノクロ
                pass
            elif src.shape[2] == 3:  # カラー
                src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
            elif src.shape[2] == 4:  # 透過
                src = cv2.cvtColor(src, cv2.COLOR_RGBA2BGRA)       

            # 目元をトリミング（保存要る？）
            imgs = trim_all_eyes(src)
            count = 0
            for img in imgs:
                split_path = os.path.splitext(UPLOAD_FOLDER + "/" + filename)
                save_name = split_path[0] + "_" + str(count) + split_path[1]
                cv2.imwrite(save_name, img)
#                 img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                count += 1
              
            # 目の画像群に対してTFモデルを使って判定処理を通す（結果だけ表示するんだったら目画像の保存不要かも）
            # TODO: 画像を入力してTFモデル適用?する処理のpyファイルが要る
            
            # 入力画像を判定結果を一緒に表示させるページに遷移
           
            
            # これあとで消す（かこれを改造して判定結果表示させる）
            # アップロード後のページに転送
            return redirect(url_for('uploaded_file', filename=filename))

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
#     return '''
# {% block content %}

# <form action="/upload" method="post" enctype="multipart/form-data">
#   <input type="file" name="image" accept="image/png, image/jpeg">
#   <button type="submit">submit</button>
# </form>

# {% if uploads %}
#   {% for path in uploads %}
#     <div>
#       <img src="uploads/{{ path }}" style="margin-top: 10px; vertical-align: bottom; width: 200px;">
#       {{ path }}
#     </div>
#   {% endfor %}
# {% endif %}

# {% endblock %}
# '''

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)