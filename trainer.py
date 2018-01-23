import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.optim import SGD

import os
import time
import shutil

from tqdm import trange
from utils import AverageMeter
from model import RecurrentAttention
from tensorboard_logger import configure, log_value


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
        else:
            self.test_loader = data_loader
        self.num_classes = 10
        self.num_channels = 1

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.saturate_epoch = config.saturate_epoch
        self.init_lr = config.init_lr
        self.min_lr = config.min_lr
        self.decay_rate = (self.init_lr - self.min_lr) / (self.saturate_epoch)
        self.momentum = config.momentum
        self.lr = self.init_lr

        # misc params
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.
        self.counter = 0
        self.patience = config.patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.model_name = 'ram_{}_{}x{}_{}'.format(
            config.num_glimpses, config.patch_size,
            config.patch_size, config.glimpse_scale
        )

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size, self.num_patches, self.glimpse_scale,
            self.num_channels, self.loc_hidden, self.glimpse_hidden,
            self.std, self.hidden_size, self.num_classes,
        )

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # initialize optimizer
        self.optimizer = SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum,
        )

    def clear(self):
        """
        Reset the hidden state of the core network and
        a few bookeeping variables.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        self.h_t = torch.zeros(self.batch_size, self.hidden_size)
        self.h_t = Variable(self.h_t)

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        for epoch in trange(self.start_epoch, self.epochs):

            # decay learning rate
            if epoch < self.saturate_epoch:
                self.anneal_learning_rate(epoch)

            # train for 1 epoch
            self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_acc = self.validate(epoch)

            # check for improvement
            is_best = valid_acc > self.best_valid_acc
            if not is_best:
                self.counter += 1
            if self.counter > self.patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                 'best_valid_acc': self.best_valid_acc}, is_best
            )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        for i, (x, y) in enumerate(self.train_loader):
            x, y = Variable(x), Variable(y)

            self.batch_size = x.shape[0]
            self.clear()

            # initialize location vector
            l_t = torch.Tensor(self.batch_size, 2).uniform_(-1, 1)
            l_t = Variable(l_t)

            # extract the glimpses
            sum_grad_log_pi = 0.
            for t in range(self.num_glimpses - 1):

                # forward pass through model
                self.h_t, mu, l_t = self.model(x, l_t, self.h_t)

                # compute gradient of log of policy and accumulate
                grad_log_pi = (mu-l_t) / (self.std*self.std)
                sum_grad_log_pi += torch.sum(grad_log_pi, dim=1)

            # last iteration
            self.h_t, mu, l_t, b_t, log_probas = self.model(
                x, l_t, self.h_t, last=True
            )

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted == y).float()

            # compute losses for differentiable modules
            self.loss_action = F.nll_loss(log_probas, y)
            self.loss_baseline = F.mse_loss(b_t, R)

            # compute reinforce loss
            adjusted_reward = R - b_t.detach()
            self.reinforce_loss = torch.mean(
                adjusted_reward*sum_grad_log_pi
            )

            # compute composite loss
            self.loss = self.loss_action + \
                self.loss_baseline + self.reinforce_loss

            # compute accuracy
            total = len(y)
            correct = R.sum()
            self.acc = 100 * (correct / total)

            # store
            losses.update(self.loss.data[0], x.size()[0])
            accs.update(self.acc.data[0], x.size()[0])

            # compute gradients and update SGD
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            # print to screen
            if i % self.print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Train Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Train Acc: {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(self.train_loader),
                        batch_time=batch_time, loss=losses, acc=accs)
                )

        # log to tensorboard
        if self.use_tensorboard:
            log_value('train_loss', losses.avg, epoch)
            log_value('train_acc', accs.avg, epoch)

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        for i, (x, y) in enumerate(self.valid_loader):
            x = Variable(x, volatile=True)
            y = Variable(y, volatile=True)

            self.batch_size = x.shape[0]
            self.clear()

            # initialize location vector
            l_t = torch.Tensor(self.batch_size, 2).uniform_(-1, 1)
            l_t = Variable(l_t)

            # extract the glimpses
            sum_grad_log_pi = 0.
            for t in range(self.num_glimpses - 1):

                # forward pass through model
                self.h_t, mu, l_t = self.model(x, l_t, self.h_t)

                # compute gradient of log of policy and accumulate
                grad_log_pi = (mu-l_t) / (self.std*self.std)
                sum_grad_log_pi += torch.sum(grad_log_pi, dim=1)

            # last iteration
            self.h_t, mu, l_t, b_t, log_probas = self.model(
                x, l_t, self.h_t, last=True
            )

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted == y).float()

            # compute losses for differentiable modules
            self.loss_action = F.nll_loss(log_probas, y)
            self.loss_baseline = F.mse_loss(b_t, R)

            # compute reinforce loss
            adjusted_reward = R - b_t
            self.reinforce_loss = torch.mean(
                adjusted_reward*sum_grad_log_pi
            )

            # compute composite loss
            self.loss = self.loss_action + \
                self.loss_baseline + self.reinforce_loss

            # compute accuracy
            total = len(y)
            correct = R.sum()
            self.acc = 100 * (correct / total)

            # store
            losses.update(self.loss.data[0], x.size()[0])
            accs.update(self.acc.data[0], x.size()[0])

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            # print to screen
            if i % self.print_freq == 0:
                print('Valid: [{0}/{1}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Valid Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Valid Acc: {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, len(self.valid_loader), batch_time=batch_time,
                        loss=losses, acc=accs))

        print('[*] Valid Acc: {acc.avg:.3f}'.format(acc=accs))

        # log to tensorboard
        if self.use_tensorboard:
            log_value('val_loss', losses.avg, epoch)
            log_value('val_acc', accs.avg, epoch)

        return accs.avg

    def anneal_learning_rate(self, epoch):
        """
        This function linearly decays the learning rate
        to a predefined minimum over a set amount of epochs.
        """
        self.lr -= self.decay_rate

        # log to tensorboard
        if self.use_tensorboard:
            log_value('learning_rate', self.lr, epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )
            print("[*] ==== Best Valid Acc Achieved ====")

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['state_dict'])

        print(
            "[*] Loaded {} checkpoint @ epoch {} \
            with best valid acc of {:.3f}".format(
                filename, ckpt['epoch'], ckpt['best_valid_acc'])
        )
