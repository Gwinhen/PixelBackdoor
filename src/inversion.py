# coding: utf-8

import keras
import numpy as np
import sys

from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

class PixelBackdoor:
    def __init__(self,
                 model,                 # subject model for inversion
                 shape=(224, 224, 3),   # input shape
                 num_classes=1000,      # number of classes of subject model
                 steps=1000,            # number of steps for inversion
                 batch_size=32,         # batch size in trigger inversion
                 asr_bound=0.9,         # threshold for attack success rate
                 init_cost=1e-3,        # weight on trigger size loss
                 lr=1e-1,               # learning rate of trigger inversion
                 clip_max=255.0,        # maximum pixel value
                 use_tanh=False,        # use tanh on input variables
                 augment=False          # use data augmentation on inputs
        ):

        self.input_shape = shape
        self.num_classes = num_classes
        self.steps = steps
        self.batch_size = batch_size
        self.asr_bound = asr_bound
        self.init_cost = init_cost
        self.clip_max = clip_max
        self.augment = augment

        # use data augmentation
        if self.augment:
            self.datagen = ImageDataGenerator(horizontal_flip=True,
                                              width_shift_range=0.125,
                                              height_shift_range=0.125,
                                              fill_mode='constant',
                                              cval=0.0)

        # initialize pattern variables
        self.pattern_pos_var = K.variable(np.zeros(self.input_shape))
        self.pattern_neg_var = K.variable(np.zeros(self.input_shape))

        # map pattern variables to the valid range
        if use_tanh:
            self.pattern_pos =   (K.tanh(self.pattern_pos_var)\
                                    / (2 - K.epsilon()) + 0.5) * self.clip_max
            self.pattern_neg = - (K.tanh(self.pattern_neg_var)\
                                    / (2 - K.epsilon()) + 0.5) * self.clip_max
        else:
            self.pattern_pos =   K.clip(self.pattern_pos_var * self.clip_max,
                                    0.0, self.clip_max)
            self.pattern_neg = - K.clip(self.pattern_neg_var * self.clip_max,
                                    0.0, self.clip_max)

        # initialize loss weight
        self.cost_var = K.variable(self.init_cost)

        # input and output placeholders
        input_ph  = K.placeholder(model.input_shape)
        output_ph = K.placeholder(model.output_shape)

        # stamp trigger pattern
        input_adv = K.clip(input_ph + self.pattern_pos + self.pattern_neg,
                            0.0, self.clip_max)

        output_adv = model(input_adv)
        output_tru = output_ph

        accuracy = keras.metrics.categorical_accuracy(output_adv, output_tru)
        loss_ce  = keras.losses.categorical_crossentropy(output_adv, output_tru)

        # loss for the number of perturbed pixels
        reg_pos  = K.max(K.tanh(self.pattern_pos_var / 10)\
                             / (2 - K.epsilon()) + 0.5, axis=2)
        reg_neg  = K.max(K.tanh(self.pattern_neg_var / 10)\
                            / (2 - K.epsilon()) + 0.5, axis=2)
        loss_reg = K.sum(reg_pos) + K.sum(reg_neg)

        # total loss
        loss = loss_ce + loss_reg * self.cost_var

        self.optimizer = optimizers.Adam(lr=lr,
                                         beta_1=0.5,
                                         beta_2=0.9)
        # parameters to optimize
        updates = self.optimizer.get_updates(params=[self.pattern_pos_var,
                                                     self.pattern_neg_var],
                                             loss=loss)
        self.train = K.function([input_ph, output_ph],
                                [loss_ce, loss_reg, loss, accuracy],
                                updates=updates)

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        # store best results
        pattern_best     = np.zeros(self.input_shape)
        pattern_pos_best = np.zeros(self.input_shape)
        pattern_net_best = np.zeros(self.input_shape)
        reg_best   = float('inf')
        pixel_best = float('inf')

        # hyper-parameters to dynamically adjust loss weight
        patience = 10
        cost_up_counter = 0
        cost_down_counter = 0
        cost = self.init_cost
        K.set_value(self.cost_var, cost)

        # initialize patterns with random values
        for i in range(2):
            init_pattern = np.random.random(self.input_shape) * self.clip_max
            init_pattern = np.clip(init_pattern, 0.0, self.clip_max)
            init_pattern = init_pattern / self.clip_max
            if i == 0:
                K.set_value(self.pattern_pos_var, init_pattern)
            else:
                K.set_value(self.pattern_neg_var, init_pattern)

        # reset optimizer states
        K.set_value(self.optimizer.iterations, 0)
        for w in self.optimizer.weights:
            K.set_value(w, np.zeros(K.int_shape(w)))

        # select inputs for label-specific or universal attack
        if source < self.num_classes:
            indices = np.where(np.argmax(y_set, axis=1) == source)[0]
            if indices.shape[0] > attack_size:
                indices = indices[:attack_size]
            else:
                attack_size = indices.shape[0]
            x_set = x_set[indices]
            y_set = np.zeros((x_set.shape[0], self.num_classes))
            y_set[:, target] = 1
        else:
            if x_set.shape[0] > attack_size:
                x_set = x_set[:attack_size]
                y_set = y_set[:attack_size]
            y_set[...] = 0
            y_set[:, target] = 1
            source = self.num_classes

        # avoid having the number of inputs smaller than batch size
        if attack_size < self.batch_size:
            self.batch_size = attack_size

        # use data augmentation
        if self.augment:
            self.datagen.fit(x_set)
            dataflow = self.datagen.flow(x_set,
                                         y_set,
                                         batch_size=self.batch_size)

        # start generation
        for step in range(self.steps):
            # shuffle training samples
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                # get a batch data
                if self.augment:
                    x_batch, y_batch = dataflow.next()
                else:
                    x_batch = x_set[idx*self.batch_size:(idx+1)*self.batch_size]
                    y_batch = y_set[idx*self.batch_size:(idx+1)*self.batch_size]

                (loss_ce_value,
                    loss_reg_value,
                    loss_value,
                    acc_value) = self.train([x_batch, y_batch])

                # record loss and accuracy
                loss_ce_list.extend( list(loss_ce_value.flatten()))
                loss_reg_list.extend(list(loss_reg_value.flatten()))
                loss_list.extend(    list(loss_value.flatten()))
                acc_list.extend(     list(acc_value.flatten()))

            # calculate average loss and accuracy
            avg_loss_ce  = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss     = np.mean(loss_list)
            avg_acc      = np.mean(acc_list)

            # remove small pattern values
            threshold = self.clip_max / 255.0
            pattern_pos_cur = K.eval(self.pattern_pos)
            pattern_neg_cur = K.eval(self.pattern_neg)
            pattern_pos_cur[(pattern_pos_cur < threshold)\
                                & (pattern_pos_cur > -threshold)] = 0
            pattern_neg_cur[(pattern_neg_cur < threshold)\
                                & (pattern_neg_cur > -threshold)] = 0
            pattern_cur = pattern_pos_cur + pattern_neg_cur

            # count current number of perturbed pixels
            pixel_cur = np.count_nonzero(np.sum(np.abs(pattern_cur), axis=2))

            # record the best pattern
            if avg_acc >= self.asr_bound and avg_loss_reg < reg_best\
                    and pixel_cur < pixel_best:
                reg_best = avg_loss_reg
                pixel_best = pixel_cur

                pattern_pos_best = K.eval(self.pattern_pos)
                pattern_pos_best[pattern_pos_best < threshold] = 0
                init_pattern = pattern_pos_best / self.clip_max
                K.set_value(self.pattern_pos_var, init_pattern)

                pattern_neg_best = K.eval(self.pattern_neg)
                pattern_neg_best[pattern_neg_best > -threshold] = 0
                init_pattern = - pattern_neg_best / self.clip_max
                K.set_value(self.pattern_neg_var, init_pattern)

                pattern_best = pattern_pos_best + pattern_neg_best

            # helper variables for adjusting loss weight
            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            # adjust loss weight
            if cost_up_counter >= patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = 1e-3
                else:
                    cost *= 1.5
                K.set_value(self.cost_var, cost)
            elif cost_down_counter >= patience:
                cost_down_counter = 0
                cost /= 1.5 ** 1.5
                K.set_value(self.cost_var, cost)

            # periodically print inversion results
            if step % 10 == 0:
                size = np.count_nonzero(np.sum(np.abs(pattern_best), axis=2))
                size = 1024 if size == 0 else size
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                    .format(step, avg_acc, avg_loss)
                                 + 'ce: {:.2f}, reg: {:.2f}, reg_best: {:.2f}, '\
                                    .format(avg_loss_ce, avg_loss_reg, reg_best)
                                 + 'size: {:d}   '\
                                    .format(size))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rtrigger size of pair {:d}-{:d}: {:d}\n'.format(
                         source,
                         target,
                         np.count_nonzero(np.sum(np.abs(pattern_best), axis=2))))
        sys.stdout.flush()

        return pattern_best
