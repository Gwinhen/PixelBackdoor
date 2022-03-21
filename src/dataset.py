# coding: utf-8

import cv2
import multiprocessing

from tensorpack import imgaug, dataset
from tensorpack.dataflow import BatchData, MultiThreadMapData, RepeatedData

class ImageNetData:
    def __init__(self, batch_size, data_name='val'):
        self.batch_size = batch_size
        self.data_name  = data_name
        self.datadir    = 'data/ILSVRC/Data/CLS-LOC'
        assert self.datadir is not None

        self.size = {'train': 1281167, 'val': 50000, 'test': 100000}
        self.size = self.size[self.data_name]

        self.data = self.get_dataflow()
        self.data.reset_state()
        self.iter = iter(self.data)


    def data_augmentor(self, isTrain):
        """
        Augmentor used for BGR images in range [0,255].
        """
        if isTrain:
            """
            augmentors = [
                imgaug.GoogleNetRandomCropAndResize(),
                # It's OK to remove the following augs if your CPU is not fast enough.
                # Removing brightness/contrast/saturation does not have a significant
                # effect on accuracy.
                # Removing lighting leads to a tiny drop in accuracy.
                imgaug.RandomOrderAug(
                    [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                     imgaug.Contrast((0.6, 1.4), clip=False),
                     imgaug.Saturation(0.4, rgb=False),
                     # rgb-bgr conversion for the constants copied from fb.resnet.torch
                     imgaug.Lighting(0.1,
                                     eigval=np.asarray(
                                         [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                     eigvec=np.array(
                                         [[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]],
                                         dtype='float32')[::-1, ::-1]
                                     )]),
                imgaug.Flip(horiz=True),
            ]
            """
            augmentors = [
                imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
                imgaug.RandomCrop((224, 224)),
                imgaug.CenterCrop((224, 224)),
            ]
        else:
            augmentors = [
                imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
                imgaug.CenterCrop((224, 224)),
            ]
        return augmentors


    def get_dataflow(self):
        # augmentors = self.data_augmentor(True if self.data_name == 'train' else False)
        augmentors = self.data_augmentor(False) # this is for trigger inversion
        assert isinstance(augmentors, list)
        aug = imgaug.AugmentorList(augmentors)

        parallel = min(4, multiprocessing.cpu_count())

        # shuffle = True if self.data_name == 'train' else False
        shuffle = False # this is for trigger inversion
        ds = dataset.ILSVRC12Files(self.datadir, self.data_name, shuffle=shuffle)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            #from BGR to RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = aug.augment(im)
            return im, cls

        ds = MultiThreadMapData(ds, parallel, mapf,
                                buffer_size=min(300, ds.size()), strict=True)
        ds = BatchData(ds, self.batch_size, remainder=False)
        ds = RepeatedData(ds, num=-1)

        return ds


    def get_next_batch(self):
        batch = next(self.iter)

        if batch is None:
            self.data.reset_state()
            self.iter = iter(self.data)
            batch = next(self.iter)

        return batch
