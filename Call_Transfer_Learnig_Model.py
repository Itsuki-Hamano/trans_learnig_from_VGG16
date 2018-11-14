# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:33:04 2018

@author: IstukiHamano
"""

from keras.models import model_from_json
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
import numpy as np

model_file_name="origin_model_from_VGG16.json"#作成したモデル（層の作り）を出力するファイル名
weight_file_name="Transfer_Learning_Model_Weight"#学習済みモデルの重みを出力するファイル名


#画像の整形
img_path='train/Danger_Road/1. 20070429_patto.jpg'#対象とする画像データ
img=image.load_img(img_path,target_size=(224,224))#画像のリサイズ
#img.show()#リサイズ後、画像表示
x=image.img_to_array(img)#画像ベクトルに変換
x=np.expand_dims(x,axis=0)#画像ベクトルをnumpy配列に変換


#作成したモデルの読み込み
model=model_from_json(open('origin_from_VGG16.json').read())

#学習済み重みの読み込み
model.load_weights('Road_Transfer_Learning_Model.h5')

#model.summary()#モデルの層を表示

"""コンパイルしても結果変わらず
model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),#作成したモデルのコンパイル
             loss='categorical_crossentropy',#損失関数の設定
             metrics=['accuracy'])#
"""

#画像認識する部分
labels=np.array(["Danger_Road","Safe_Road"])
preds=model.predict(preprocess_input(x))
preds=np.array(preds[0])
results=np.vstack((labels,preds))#結果のラベルづけ
print(results)


"""
results=decode_predictions(preds,top=1)[0]#Danger,Safeそれぞれの確信度表示
for result in results:
    print(result)
""" 
    



