# coding: utf-8

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import argparse
import numpy as np
import os
import random
import time
import torch

from torchvision import models as M
from torchvision import transforms as T

from dataset import ImageNetData
from inversion_torch import PixelBackdoor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_imagenet_networks = {
    'resnet50': M.resnet50(pretrained=True),
}


def get_model(network):
    model = _imagenet_networks[network]
    return model


def get_norm():
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std  = torch.FloatTensor([0.229, 0.224, 0.225])
    normalize = T.Normalize(mean, std)
    return normalize


def preprocess(inputs):
    inputs = np.transpose(inputs / 255.0, [0, 3, 1, 2])
    inputs = torch.FloatTensor(inputs)
    return inputs


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
    print('data:', x_data.shape, y_data.shape)

    return x_data, y_data


def test():
    model = get_model(args.model).to(args.device)
    model.eval()

    normalize = get_norm()

    val_data = ImageNetData(args.batch_size, 'val')
    val_samples = 50000

    acc = 0
    for step in range(val_samples // args.batch_size):
        x_batch, y_batch = val_data.get_next_batch()

        x_batch = normalize(preprocess(x_batch)).to(args.device)
        y_batch = torch.LongTensor(y_batch).to(args.device)

        pred = model(x_batch).argmax(dim=1)
        correct = (pred == y_batch).sum().item()
        acc += correct / y_batch.size(0)

        if step % 10 == 0:
            print(step, acc / (step + 1))

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
    y_val = np.argmax(y_val, axis=1)

    print('generation set:', x_val.shape)

    model = get_model(args.model).to(args.device)
    model.eval()

    x_val = preprocess(x_val)
    y_val = torch.LongTensor(y_val)

    normalize = get_norm()

    # trigger inversion
    time_start = time.time()

    backdoor = PixelBackdoor(model,
                             batch_size=args.batch_size,
                             normalize=normalize)

    pattern = backdoor.generate((source, target),
                                x_val,
                                y_val,
                                attack_size=args.attack_size)

    time_end = time.time()
    print('='*50)
    print('Generation time: {:.4f} m'.format((time_end - time_start) / 60))
    print('='*50)

    np.save(f'trigger/pattern_{source}-{target}', pattern.cpu().numpy())
    size = np.count_nonzero(pattern.abs().sum(0).cpu().numpy())
    print('trigger size:  ', size)

    # load data from ImageNet data generator
    # data_loader = ImageNetData(50, 'val')
    # x_val, y_val = get_data(data_loader, source)

    # load data from saved path
    x_val = np.load('data/imagenet_test_x.npy')
    y_val = np.load('data/imagenet_test_y.npy')
    y_val = np.argmax(y_val, axis=1)

    x_val = preprocess(x_val).to(args.device)
    x_adv = torch.clamp(x_val + pattern, 0, 1)
    x_adv = normalize(x_adv)

    pred = model(x_adv).argmax(dim=1)
    correct = (pred == target).sum().item()
    asr = correct / pred.size(0)
    print(f'success rate: {asr:.4f}')



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
    args.device = torch.device('cuda')

    # set gpu usage
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # main function
    time_start = time.time()
    main()
    time_end = time.time()
    print('='*50)
    print('Running time: {:.4f} m'.format((time_end - time_start) / 60))
    print('='*50)
