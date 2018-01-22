import os
import time
import torch
import shutil

from tqdm import trange
from torch.autograd import Variable
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
        self.num_classes = 10
        self.num_channels = 1

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
        self.batch_size = config.batch_size
        self.model_name = 'ram_{}_{}x{}_{}'.format(
            config.num_glimpses, config.patch_size,
            config.patch_size, config.glimpse_scale
        )

        # build RAM model
        self.model = RecurrentAttention(
            self.loc_hidden, self.glimpse_hidden, self.patch_size,
            self.num_patches, self.glimpse_scale, self.num_channels,
            self.hidden_size, self.num_classes, self.std,
        )

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # switch to train mode for dropout
        self.model.train()

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
        for i, (img, target) in enumerate(self.train_loader):
            img_var = Variable(img)
            target_var = Variable(target)

            self.locs = []
            for t in range(self.num_glimpses - 1):
                if t == 0:
                    # initialize location and hidden state vectors
                    l_t = Variable(
                        torch.Tensor(
                            self.batch_size, 2
                        ).uniform_(-1, 1)
                    )
                    h_t = Variable(
                        torch.zeros(self.batch_size, self.hidden_size)
                    )

                # forward pass through model
                h_t, l_t = self.model(img_var, l_t, h_t)

                # bookeeping for later plotting
                self.locs.append(l_t)

            # last iteration
            probas = self.model(img_var, l_t, h_t, last=True)

            # to be continued


    def anneal_learning_rate(self, epoch):
        """
        This function linearly decays the learning rate
        to a predefined minimum over a set amount of epochs.
        """
        self.lr -= self.decay

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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
