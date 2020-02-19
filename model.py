import tensorflow as tf
import tensorflow.keras.layers as layers
import IPython.display as display
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pathlib
import io

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 416
IMG_WIDTH = 416


def build(img_w, img_h, grid_w, grid_h, nb_boxes, nb_classes):
    inputs = tf.keras.Input(shape=(img_w, img_h, 3))
    x = layers.Conv2D(16, (1, 1))(inputs)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='sigmoid')(x)
    x = layers.Dense(
        grid_w * grid_h * (nb_boxes * 5 + nb_classes), activation='sigmoid')(x)
    outputs = layers.Reshape(
        (grid_w * grid_h, (nb_boxes * 5 + nb_classes)))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='YoloV3')


raw_image_dataset = tf.data.TFRecordDataset('./train.record')

# Create a dictionary describing the features.
image_feature_description = {"image/filename": tf.io.FixedLenFeature((), tf.string, ""),
                             'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                             'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
                             'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                             'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                             'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                             'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32)}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)



parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset = parsed_image_dataset.shuffle(500)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def encode_img(tensor):
    # resize the image to the desired size.
    # Use `convert_image_dtype` to convert to uint8.
    img = tf.image.convert_image_dtype(tensor, tf.uint8)
    # convert the 3D uint8 tensor to a compressed string
    return tf.image.encode_jpeg(img)

def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


# Testing Code to Draw the Bounding Boxes
for image_features in parsed_image_dataset.take(1):
    filename = image_features['image/filename'].numpy().decode('utf-8')
    #print(filename)
    tensor = process_path('./images/power_cell/' + filename)
    fh = open("imageToSave.jpeg", "wb")
    fh.write(encode_img(tensor).numpy())
    fh.close()
    pic = Image.open('imageToSave.jpeg')
    draw = ImageDraw.Draw(pic)
    xmins = image_features['image/object/bbox/xmin'].values.numpy()
    xmaxes = image_features['image/object/bbox/xmax'].values.numpy()
    ymins = image_features['image/object/bbox/ymin'].values.numpy()
    ymaxes = image_features['image/object/bbox/ymax'].values.numpy()

    for i in range(xmins.size):
        print(i)
        draw.rectangle([(xmins[i] * 416, ymins[i] * 416), (xmaxes[i] * 416, ymaxes[i] * 416)], outline=0xff0000,
                       width=3, fill=None)
    display.display(pic)

# I'm pretty sure this is the dataset now
# train_ds=prepare_for_training(labeled_ds)

# image_batch, label_batch=next(iter(train_ds))

# print(label_batch.numpy())

# show_batch(image_batch.numpy(), label_batch.numpy())

def calc_IOU(truth, pred):
    overlap_w = (truth.w + pred.w) / 2 - abs(truth.x - pred.x)
    overlap_h = (truth.h + pred.h) / 2 - abs(truth.y - pred.y)
    overlap_a = overlap_w * overlap_h
    truth_a = truth.w * truth.h
    pred_a = pred.w * pred.h
    union_a = truth_a + pred_a - overlap_a
    return overlap_a / union_a
