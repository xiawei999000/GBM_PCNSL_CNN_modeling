'''
@File        : CNN_training.py
@Author      : xiaw@sibet.ac.cn
@Date        : 2021/02/12
@Description :
Deep Learning for Automatic Differential Diagnosis of Primary Central Nervous System Lymphoma and
Glioblastoma: Multi-parametric MRI based Convolutional Neural Network Model

## script for CNN based model training and generate predictions.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.metrics import roc_curve, auc
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import DenseNet121
from tensorflow.python.keras.applications.densenet import preprocess_input
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
N_CLASSES = 2
TARGET_SIZE = (224, 224)

class DenseNetFineTune():
    def __init__(self):
        self.base_model = DenseNet121(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        self.predictions = Dense(N_CLASSES, activation='softmax')(GlobalAveragePooling2D()(self.base_model.output))
        self.model = Model(inputs=self.base_model.input, outputs=self.predictions)
        # print(self.model.summary())

def train(origin_model, fold, modal):
    # data generator with data argumentation
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05
    )
    # train and validation
    train_generator = data_generator.flow_from_directory(
        './data/fold_' + fold + '/' + modal + '/brain_tumor_train/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    val_generator = data_generator.flow_from_directory(
        './data/fold_' + fold + '/' + modal + '/brain_tumor_val/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    origin_model.compile(optimizer=Adam(lr=2e-5), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # callbacks
    output_model_file = './model/' + modal + '_' + fold + '_checkpoint-{epoch:02d}e-val_acc_{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(output_model_file, monitor='val_acc', verbose=1, save_best_only=True)

    lr_reducer = ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=3,
                                   verbose=1, mode='max', min_delta=1.e-5, cooldown=0, min_lr=0)

    EarlyStop = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')

    # begin training history_ft =
    origin_model.fit_generator(
        train_generator,
        epochs=50,
        callbacks=[checkpoint, lr_reducer, EarlyStop],  # tfboard_callback,
        validation_data=val_generator,
    )

def cal_y_test_and_y_score(model, src_path, p_label, result, fw):
    label = 0 if 'GBM' == p_label else 1
    for subroot, dirs, files in os.walk(src_path + p_label):
        if files is not None:
            for file in files:
                img_file_path = os.path.join(subroot, file)
                # logger.info('working on %s', img_file_path)
                img = Image.open(img_file_path)
                if img.size != TARGET_SIZE:
                    img = img.resize(TARGET_SIZE)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x_size = x.shape
                if x_size[3] == 1:
                    x = np.tile(x, 3)
                    x = preprocess_input(x)
                    preds = model.predict(x)
                    result.append([label, preds[0][1]])
                    fw.write('{}\t{}\t{}\t{}\n'.format(
                        img_file_path.split('/')[-1].split('_')[0],
                        img_file_path.split('/')[-1].split('_')[2].replace('.jpg', ''),
                        img_file_path.split('/')[-2],
                        preds[0][1]))

                if x_size[3] == 3:
                    x = preprocess_input(x)
                    preds = model.predict(x)
                    result.append([label, preds[0][1]])
                    fw.write('{}\t{}\t{}\t{}\n'.format(
                        img_file_path.split('/')[-1].split('_')[0],
                        img_file_path.split('/')[-1].split('_')[3].replace('.jpg', ''),
                        img_file_path.split('/')[-2],
                        preds[0][1]))


def model_test(model, src_path, fold, modal, procedure):
    result = []
    with open('./data/fold_' + fold + '_' + modal + '_' + procedure + '.txt', 'w') as fw:
        cal_y_test_and_y_score(model, src_path, 'GBM', result, fw)
        cal_y_test_and_y_score(model, src_path, 'PCNSL', result, fw)
    # print(result)

if __name__ == '__main__':

    # train
    for fold in ['1', '2', '3', '4', '5']:
        for modal in ['T1C', 'T2F', 'ADC', 'cross_modal']:
            # cross_modal means image fusion(IF)
            print('fold ', fold, 'and modal ', modal)
            densenet = DenseNetFineTune()
            train(densenet.model, fold, modal)

    # test and generate predictions for each dataset in each fold
    for fold in ['1', '2', '3', '4', '5']:
        for modal in ['T1C', 'T2F', 'ADC', 'cross_modal']:
            model_path = ''
            for path in os.listdir('./model/'):
                if modal + '_' + fold in path:
                    model_path = path
            model = load_model('./model/' + model_path,
                               custom_objects={'BatchNormalizationV1': BatchNormalization}, compile=False)
            for procedure in ['train', 'val', 'test']:
                src_path = './data/fold_' + fold + '/' + modal + '/brain_tumor_' + procedure + '/'
                print('')
                print('---fold:{}, modal:{}, model:{}, procedure:{}---'.format(fold, modal, model_path, procedure))
                model_test(model, src_path, fold, modal, procedure)
