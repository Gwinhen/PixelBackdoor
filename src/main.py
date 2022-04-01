# coding: utf-8

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import argparse
import numpy as np
import keras
import os
import time

from keras import backend as K
from keras import optimizers
from keras.applications import resnet50
from keras.layers import Input, Lambda
from keras.models import Model

from dataset import ImageNetData
from inversion import PixelBackdoor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if('tensorflow' == K.backend()):
    import tensorflow as tf

    try:
        from tensorflow.python.util import module_wrapper as deprecation
    except ImportError:
        from tensorflow.python.util import deprecation_wrapper as deprecation
    deprecation._PER_MODULE_WARNING_LIMIT = 0
    tf.logging.set_verbosity(tf.logging.ERROR)


def normalization(X):
    MEAN = [103.939, 116.779, 123.68]
    red, green, blue = tf.split(X, 3, 3)
    X_bgr = tf.concat([
                blue  - MEAN[0],
                green - MEAN[1],
                red   - MEAN[2],
            ], 3)
    return X_bgr


def get_model(logits=False):
    if args.model == 'resnet50':
        model = resnet50.ResNet50(weights='imagenet')

    # use logits
    if logits:
        model.layers[-1].activation = keras.activations.linear

    # add preprocessing layer
    model.layers.pop(0)
    new_input = Input(shape=(224, 224, 3))
    x = model(Lambda(lambda x: normalization(x))(new_input))
    new_model = Model(inputs=new_input, outputs=x)

    return new_model


def get_data(loader, source, size=100):
    x_data = []
    y_data = []

    # get all inputs with source label
    for i in range(15):
        x_batch, y_batch = loader.get_next_batch()
        indices = np.where(y_batch == source)[0]
        if i == 0:
            x_data = x_batch[indices]
            y_data = y_batch[indices]
        else:
            x_data = np.concatenate((x_data, x_batch[indices]), axis=0)
            y_data = np.concatenate((y_data, y_batch[indices]), axis=0)
        if x_data.shape[0] >= size:
            break

    x_data = x_data[:size]
    y_data = y_data[:size]
    y_data = keras.utils.to_categorical(y_data, args.num_classes)
    print('data:', x_data.shape, y_data.shape)

    return x_data, y_data


def test():
    model = get_model()
    model.load_weights(f'ckpt/imagenet_{args.model}.h5')
    model.compile(optimizer=optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    val_data = ImageNetData(args.batch_size, 'val')
    val_samples = 50000

    acc = 0
    for step in range(val_samples // args.batch_size):
        x_batch, y_batch = val_data.get_next_batch()
        y_batch = keras.utils.to_categorical(y_batch, args.num_classes)

        values = model.test_on_batch(x_batch, y_batch)
        acc += values[1]

        if step % 10 == 0:
            print(step, values[1])

    acc /= step + 1
    print(f'val acc: {acc}')


def evaluate():
    source, target = list(map(int, args.pair.split('-')))

    # load data from ImageNet data generator
    # data_loader = ImageNetData(50, 'train')
    # x_val, y_val = get_data(data_loader, source)

    # load data from saved path
    x_val = np.load('data/imagenet_train_x.npy')
    y_val = np.load('data/imagenet_train_y.npy')

    print('generation set:', x_val.shape)

    model = get_model()
    model.compile(optimizer=optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # trigger inversion
    time_start = time.time()

    backdoor = PixelBackdoor(model, batch_size=args.batch_size)

    pattern = backdoor.generate((source, target),
                                x_val,
                                y_val,
                                attack_size=args.attack_size)

    time_end = time.time()
    print('='*50)
    print('Generation time: {:.4f} m'.format((time_end - time_start) / 60))
    print('='*50)

    np.save(f'trigger/pattern_{source}-{target}', pattern)
    size = np.count_nonzero(np.sum(np.abs(pattern), axis=2))
    print('trigger size:  ', size)

    # load data from ImageNet data generator
    # data_loader = ImageNetData(50, 'val')
    # x_val, y_val = get_data(data_loader, source)

    # load data from saved path
    x_val = np.load('data/imagenet_test_x.npy')
    y_val = np.load('data/imagenet_test_y.npy')

    x_adv = np.clip(x_val + pattern, 0, 255)
    y_adv = np.zeros((x_adv.shape[0], args.num_classes))
    y_adv[:, target] = 1

    score = model.evaluate(x_adv, y_adv, verbose=0)
    print(f'success rate: {score[1]:.4f}')



################################################################
############                  main                  ############
################################################################
def main():
    if args.phase == 'test':
        test()
    elif args.phase == 'evaluate':
        evaluate()
    else:
        print('Option [{}] is not supported!'.format(args.phase))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--gpu',   default='0',        help='gpu id')
    parser.add_argument('--phase', default='evaluate', help='phase of framework')
    parser.add_argument('--model', default='resnet50', help='model architecture')
    parser.add_argument('--pair',  default='33-104',   help='label pair')

    parser.add_argument('--seed',        default=1024, type=int, help='random seed')
    parser.add_argument('--batch_size',  default=32,   type=int, help='batch size')
    parser.add_argument('--num_classes', default=1000, type=int, help='number of classes')
    parser.add_argument('--attack_size', default=50,   type=int, help='number of samples for inversion')

    args = parser.parse_args()

    # set gpu usage
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)

    # set random seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # main function
    time_start = time.time()
    main()
    time_end = time.time()
    print('='*50)
    print('Running time: {:.4f} m'.format((time_end - time_start) / 60))
    print('='*50)
