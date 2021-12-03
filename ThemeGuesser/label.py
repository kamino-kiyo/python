from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import numpy as np


# データを正方形にリサイズするときの辺の長さ
SQUARE_LENGTH = 64
# これラベル関係ないし、ファイル名変えたい


# ラベル付け(0春 1夏 2冬)
# TODO: ? フォルダ名に数字入れておくとかのほうがいいかもしれない？
# TODO: 3つ以上に増やす場合は、class_modeやlossをbinaryからcategricalに変える
class Label(Enum):
    Ariel = 0
    Other = 1

# Label[name].value
# Label[value].name  
   

# 複数の判定値から1つの機種名を出す（二項分類の場合にしか使えなさそう）
# 四捨五入まわり　https://note.nkmk.me/python-round-decimal-quantize/
def binarization(results):
    return Decimal(str(np.mean(results))).quantize(Decimal("0"), rounding=ROUND_HALF_UP)