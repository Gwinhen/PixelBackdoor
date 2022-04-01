# coding: utf-8

import numpy as np
import sys
import torch

from torchvision import transforms as T

class PixelBackdoor:
    def __init__(self,
                 model,                 # subject model for inversion
                 shape=(3, 224, 224),   # input shape
                 num_classes=1000,      # number of classes of subject model
                 steps=1000,            # number of steps for inversion
                 batch_size=32,         # batch size in trigger inversion
                 asr_bound=0.9,         # threshold for attack success rate
                 init_cost=1e-3,        # weight on trigger size loss
                 lr=0.1,                # learning rate of trigger inversion
                 clip_max=1.0,          # maximum pixel value
                 normalize=None,        # input normalization
                 augment=False          # use data augmentation on inputs
        ):

        self.model = model
        self.input_shape = shape
        self.num_classes = num_classes
        self.steps = steps
        self.batch_size = batch_size
        self.asr_bound = asr_bound
        self.init_cost = init_cost
        self.lr = lr
        self.clip_max = clip_max
        self.normalize = normalize
        self.augment = augment

        # use data augmentation
        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                # T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.input_shape[1], scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5
        self.pattern_shape = self.input_shape

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        # store best results
        pattern_best     = torch.zeros(self.pattern_shape).to(self.device)
        pattern_pos_best = torch.zeros(self.pattern_shape).to(self.device)
        pattern_neg_best = torch.zeros(self.pattern_shape).to(self.device)
        reg_best = float('inf')
        pixel_best  = float('inf')

        # hyper-parameters to dynamically adjust loss weight
        cost = self.init_cost
        cost_up_counter   = 0
        cost_down_counter = 0

        # initialize patterns with random values
        for i in range(2):
            init_pattern = np.random.random(self.pattern_shape) * self.clip_max
            init_pattern = np.clip(init_pattern, 0.0, self.clip_max)
            init_pattern = init_pattern / self.clip_max

            if i == 0:
                pattern_pos_tensor = torch.Tensor(init_pattern).to(self.device)
                pattern_pos_tensor.requires_grad = True
            else:
                pattern_neg_tensor = torch.Tensor(init_pattern).to(self.device)
                pattern_neg_tensor.requires_grad = True

        # select inputs for label-specific or universal attack
        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        # avoid having the number of inputs smaller than batch size
        if attack_size < self.batch_size:
            self.batch_size = attack_size

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        [pattern_pos_tensor, pattern_neg_tensor],
                        lr=self.lr, betas=(0.5, 0.9)
                    )

        # start generation
        self.model.eval()
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
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                # map pattern variables to the valid range
                pattern_pos =   torch.clamp(pattern_pos_tensor * self.clip_max,
                                            min=0.0, max=self.clip_max)
                pattern_neg = - torch.clamp(pattern_neg_tensor * self.clip_max,
                                            min=0.0, max=self.clip_max)

                # stamp trigger pattern
                x_adv = torch.clamp(x_batch + pattern_pos + pattern_neg,
                                    min=0.0, max=self.clip_max)
                x_adv = self.normalize(x_adv)

                # use data augmentation
                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                loss_ce  = criterion(output, y_batch)

                # loss for the number of perturbed pixels
                reg_pos  = torch.max(torch.tanh(pattern_pos_tensor / 10)\
                                 / (2 - self.epsilon) + 0.5, axis=0)[0]
                reg_neg  = torch.max(torch.tanh(pattern_neg_tensor / 10)\
                                / (2 - self.epsilon) + 0.5, axis=0)[0]
                loss_reg = torch.sum(reg_pos) + torch.sum(reg_neg)

                # total loss
                loss = loss_ce.mean() + loss_reg * cost

                loss.backward()
                optimizer.step()

                # record loss and accuracy
                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            # calculate average loss and accuracy
            avg_loss_ce  = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss     = np.mean(loss_list)
            avg_acc      = np.mean(acc_list)

            # remove small pattern values
            threshold = self.clip_max / 255.0
            pattern_pos_cur = pattern_pos.detach()
            pattern_neg_cur = pattern_neg.detach()
            pattern_pos_cur[(pattern_pos_cur < threshold)\
                                & (pattern_pos_cur > -threshold)] = 0
            pattern_neg_cur[(pattern_neg_cur < threshold)\
                                & (pattern_neg_cur > -threshold)] = 0
            pattern_cur = pattern_pos_cur + pattern_neg_cur

            # count current number of perturbed pixels
            pixel_cur = np.count_nonzero(
                            np.sum(np.abs(pattern_cur.cpu().numpy()), axis=0)
                        )

            # record the best pattern
            if avg_acc >= self.asr_bound and avg_loss_reg < reg_best\
                    and pixel_cur < pixel_best:
                reg_best = avg_loss_reg
                pixel_best = pixel_cur

                pattern_pos_best = pattern_pos.detach()
                pattern_pos_best[pattern_pos_best < threshold] = 0
                init_pattern = pattern_pos_best / self.clip_max
                with torch.no_grad():
                    pattern_pos_tensor.copy_(init_pattern)

                pattern_neg_best = pattern_neg.detach()
                pattern_neg_best[pattern_neg_best > -threshold] = 0
                init_pattern = - pattern_neg_best / self.clip_max
                with torch.no_grad():
                    pattern_neg_tensor.copy_(init_pattern)

                pattern_best = pattern_pos_best + pattern_neg_best

            # helper variables for adjusting loss weight
            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            # adjust loss weight
            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = self.init_cost
                else:
                    cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier_down

            # periodically print inversion results
            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, reg_best: {:.2f}, '\
                                 .format(avg_loss_ce, avg_loss_reg, reg_best)\
                                 + 'size: {:.0f}  '.format(pixel_best))
                sys.stdout.flush()

        size = np.count_nonzero(pattern_best.abs().sum(0).cpu().numpy())
        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rtrigger size of pair {:d}-{:d}: {:d}\n'\
                         .format(source, target, size))
        sys.stdout.flush()

        return pattern_best
