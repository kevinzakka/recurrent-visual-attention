import torch
from torch import nn
from tqdm import trange
from torch.autograd import Variable

from model import RecurrentAttention


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
        self.num_glimpses = config.num_glimpses
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
        else:
            self.test_loader = data_loader

        # build densenet model
        self.model = RecurrentAttention(
            config.loc_hidden, config.glimpse_hidden, config.patch_size,
            config.num_patches, config.glimpse_scale, self.num_channels,
            config.hidden_size, self.num_classes, config.std,
        )

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.saturate_epoch = config.saturate_epoch
        self.init_lr = config.init_lr
        self.min_lr = config.min_lr
        self.decay_rate = (self.init_lr - self.min_lr) / (self.saturate_epoch)
        self.momentum = config.momentum
        self.lr = self.init_lr

        # other params
        self.best_valid_acc = 0.
        self.counter = 0
        self.patience = config.patience

        # # define optimizer
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.init_lr,
        #         momentum=self.momentum, weight_decay=self.weight_decay)

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

            is_best = valid_acc > self.best_valid_acc

            if not is_best:
                self.counter += 1
            if self.counter > self.patience:
                print("[!] No improvement in a while, stopping training.")
                return

            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_valid_acc': self.best_valid_acc
                }, is_best
            )

    def train_one_epoch(self, epoch):
        pass

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
