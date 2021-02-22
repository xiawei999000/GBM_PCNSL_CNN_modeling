
'''
@File        : Densenet_classification.py
@Author      : xiaw@sibet.ac.cn
@Date        : 2021/02/12
@Description :
Deep Learning for Automatic Differential Diagnosis of Primary Central Nervous System Lymphoma and
Glioblastoma: Multi-parametric MRI based Convolutional Neural Network Model

## script for heatmap generation.
'''

import os
import numpy as np
from tensorflow.python.keras.models import load_model
import cv2
import time
from PIL import Image
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.densenet import preprocess_input
from tensorflow.python.keras.layers import BatchNormalization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
N_CLASSES = 2
TARGET_SIZE = (224, 224)

def cal_y_score_and_draw_heatmap(model, src_path, output_path):
    for subroot, dirs, files in os.walk(src_path):
        if files is not None:
            for file in files:
                img_file_path = os.path.join(subroot, file)
                img = Image.open(img_file_path)
                if img.size != TARGET_SIZE:
                    img = img.resize(TARGET_SIZE)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)

                x_size = x.shape
                if x_size[3] == 1:
                    x = np.tile(x, 3)
                    x = preprocess_input(x)

                if x_size[3] == 3:
                    x = preprocess_input(x)

                start_time = time.time()
                preds = model.predict(x)
                print("----img_file:{} predict_time:{}".format(img_file_path, str(time.time() - start_time)))
                label = 0 if preds[0][0] > preds[0][1] else 1
                heatmap = draw_heatmap(model, x, label)
                sample_back_to_original_pic(img_file_path, heatmap, output_path)


def draw_heatmap(model, x, label):
    # draw heat map
    preds = model.output[:, label]
    last_conv_layer = model.get_layer('conv5_block16_concat')
    grads = K.gradients(preds, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, (0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    conv_layer_output_value *= pooled_grads_value
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # plt.matshow(heatmap)
    return heatmap

def sample_back_to_original_pic(img_path, heatmap, output_path):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.3 + img
    # superimposed_img = heatmap
    cv2.imwrite(output_path + img_path.split("/")[-2] + "_" + img_path.split("/")[-1].replace("\\", "_"), superimposed_img)


if __name__ == '__main__':
        model = load_model('./model/T1C_2_checkpoint-34e-val_acc_0.88.hdf5',
                       custom_objects={'BatchNormalizationV1': BatchNormalization})
        src_path = './data/source/T1C/'  # folder store images
        output_path = './data/heat_map/T1C/' # folder save heat map
        cal_y_score_and_draw_heatmap(model, src_path, output_path)
