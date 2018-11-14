# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:20:18 2018

@author: IstukiHamano
"""

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense,GlobalAveragePooling2D,Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import CSVLogger

n_categories=2#分類したいカテゴリ数（Denger_Road,Safe_Road用）
batch_size=32
train_data_dir="train"#訓練データのあるディレクトリ
validation_data_dir="validation"#バリデーションで使うデータのあるディレクトリ
model_file_name="origin_model_from_VGG16.json"#作成したモデル（層の作り）を出力するファイル名
weight_file_name="Transfer_Learning_Model_Weight"#学習済みモデルの重みを出力するファイル名


#画像の前処理
train_images=ImageDataGenerator(#訓練データの画像整形
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

validation_images=ImageDataGenerator(rescale=1.0/255)#バリデーションデータの画像整形

train_generator=train_images.flow_from_directory(#訓練画像データとカテゴリ（ファイル名）を紐づけ
        train_data_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
        )

validation_generator=validation_images.flow_from_directory(#バリデーション画像データとカテゴリ（ファイル名）を紐づけ
        validation_data_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
        )


base_model=VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))

base_model.summary()#VGG16の層を表示

#追加する層（prediction）を定義
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
prediction=Dense(n_categories,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=prediction)#VGG16の後ろにpredictionを追加する。

for layer in base_model.layers[:15]:#VGG16の分の層の重み（14層目まで）は変更しない設定
    layer.trainable=False

model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),#作成したモデルのコンパイル
             loss='categorical_crossentropy',#損失関数の設定
             metrics=['accuracy'])#

model.summary#作成した層を表示

#作成したモデル書き出し
model_json_str = model.to_json()
open(model_file_name, 'w').write(model_json_str)




#作成したモデルの学習
hist=model.fit_generator(train_generator,#訓練データ、日によってfitかfit_generator切り替える必要あるかも
               steps_per_epoch=2,
               epochs=2,#学習回数
               verbose=1,#ログをプログレスバーで出力、2の場合、エポック毎に1行のログを出力
               validation_data=validation_generator,#バリデーションデータ
               validation_steps=2,
               callbacks=[CSVLogger(weight_file_name+'.csv')]
               )

#作成した学習済みモデルの重みをファイルで出力
model.save(weight_file_name+'.h5')
