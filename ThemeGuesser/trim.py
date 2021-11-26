#!/usr/bin/env python
# coding: utf-8
import cv2
import dlib
import glob
import numpy as np
from numpy import linalg as LA
from imutils import face_utils
import os
from PIL import Image

# フォルダ指定(末尾に/要る)
src_dir = "D:/python/Image/"
dst_dir = "D:/python/Image/test/"

# 目の位置（0.5で目尻目頭が真ん中に来る、0で下、1で上）
# TODO: 0.0でバグるはずなので弾きたい
ratio = 0.4

# 2次元平面ベクトルをdeg度回転させる行列
# https://qiita.com/harmegiddo/items/8f4b985e19bbc3c23c1f
def rotateMat(deg):
    theta = np.deg2rad(deg)
    return np.matrix([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],  
    ])


# 2次元平面ベクトルとx軸との角度
# https://www.mathpython.com/ja/numpy-vector-angle/
def angle_xaxis(v):
    u = np.array([1, 0])
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)

    return np.rad2deg(np.arccos(i / n))[0]


# 目尻と目頭の座標を使って画像から片目を切り出す（保存は外でやる）
def trim_eye(img, left, right):
    # 目の両端から縦に広げた正方形（quad: leftup rightup leftdown rightdown）
    v = np.array([right - left]) # 目の横幅（目尻と目頭のベクトル）
    h =  rotateMat(90) * v.reshape(2,1)  # vを90度回転して長さ半分（これを上下に足せば正方形になる）
    lu = np.array([left]).reshape([2,1]) + h * ratio
    ld = np.array([left]).reshape([2,1]) - h * (1 - ratio)
    ru = np.array([right]).reshape([2,1]) + h * ratio
    rd = np.array([right]).reshape([2,1]) - h * (1 - ratio)
    
    # quadからrect(?)を計算する
    center =  tuple(map(int, (lu + ld + ru + rd) / 4))
    s = round(np.linalg.norm(v, ord=2))
    size = tuple(map(int, (s, s)))
    angle = angle_xaxis(v) * np.sign(right[1] - left[1]) 
    h, w = img.shape[:2]  # 画像の高さ、幅
    
    # 画像を回転する
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    
    # 切り抜く
    return cv2.getRectSubPix(rotated, size, center)


# 画像1枚渡すと片目ずつの画像群を返す処理
def trim_all_eyes(src):    
    # --------------------------------
    # 1.顔ランドマーク検出の前準備
    # https://qiita.com/mimitaro/items/bbc58051104eafc1eb38
    # --------------------------------
    # 顔検出ツールの呼び出し
    face_detector = dlib.get_frontal_face_detector()

    # 顔のランドマーク検出ツールの呼び出し
    predictor_path = '../dlib/shape_predictor_68_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    
    # 処理高速化のためグレースケール化(任意)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # --------------------------------
    # 2.顔のランドマーク検出
    # --------------------------------
    # 顔検出
    # ※2番めの引数はupsampleの回数。基本的に1回で十分。
    faces = face_detector(src_gray, 1)
    
    eyes = []
    
    # 検出した全顔に対して処理
    for face in faces:
        # 顔のランドマーク検出
        landmark = face_predictor(src_gray, face)
        # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
        landmark = face_utils.shape_to_np(landmark)

        # 目のトリミング
        left_eye = trim_eye(src, landmark[36], landmark[39])    
        right_eye = trim_eye(src, landmark[42], landmark[45])
        eyes.append(left_eye)
        eyes.append(right_eye)
    return eyes


# 画像全部見る
for file in glob.glob(src_dir + "/**/*", recursive=True):
    if os.path.splitext(file)[1] in [".jpg", ".png", ".bmp"]:
        print("--------------------------------")
        
        # cv2.imreadが日本語ファイル名に対応していないので、一旦Pillowで読み込んでから変換する
        # https://qiita.com/derodero24/items/f22c22b22451609908ee
        src = np.array(Image.open(file), dtype=np.uint8)
        if src.ndim == 2:  # モノクロ
            pass
        elif src.shape[2] == 3:  # カラー
            src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        elif src.shape[2] == 4:  # 透過
            src = cv2.cvtColor(src, cv2.COLOR_RGBA2BGRA)       

        # 保存先パス周りの整備(フォルダなかったら作る)
        dst_path = dst_dir + file.replace(os.sep,"/").replace(src_dir,"")
        print(dst_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        split_dst_path = os.path.splitext(dst_path)
        print(split_dst_path)
        
        # 目をトリミングして保存
        eye_count = 0
        # この中で顔検出ツール呼び出しやってるの無駄だけど一旦気付かなかったフリをする（TODO: ？）
        eyes = trim_all_eyes(src)
        for eye in eyes:
            # 保存
            # TODO: (放置でもいいかも)日本語だけは文字化けするので元々のファイル名を維持できない
            each_dst_path = split_dst_path[0] + "_" + str(eye_count) + split_dst_path[1]
            cv2.imwrite(each_dst_path, eye)
            eye_count += 1
            
            print(each_dst_path)
            
print("end")