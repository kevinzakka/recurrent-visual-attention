import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# glimpse network params
glimpse_arg = add_argument_group('Glimpse Network Params')
glimpse_arg.add_argument('--patch_size', type=int, default=8,
                         help='size of extracted patch at highest res')
glimpse_arg.add_argument('--glimpse_scale', type=int, default=2,
                         help='scale of successive patches')
glimpse_arg.add_argument('--num_patches', type=int, default=1,
                         help='# of downscaled patches per glimpse')
glimpse_arg.add_argument('--loc_hidden', type=int, default=128,
                         help='hidden size of loc fc')
glimpse_arg.add_argument('--glimpse_hidden', type=int, default=128,
                         help='hidden size of glimpse fc')


# core network params
core_arg = add_argument_group('Core Network Params')
core_arg.add_argument('--num_glimpses', type=int, default=6,
                      help='# of glimpses, i.e. BPTT iterations')
core_arg.add_argument('--hidden_size', type=int, default=256,
                      help='hidden size of rnn')


# reinforce params
reinforce_arg = add_argument_group('Reinforce Params')
reinforce_arg.add_argument('--std', type=float, default=0.17,
                           help='gaussian policy standard deviation')


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_size', type=float, default=0.15,
                      help='Proportion of training set used for validation')
data_arg.add_argument('--batch_size', type=int, default=100,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=4,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')
data_arg.add_argument('--show_sample', type=str2bool, default=False,
                      help='Whether to visualize a sample grid of the data')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--momentum', type=float, default=0.5,
                       help='Nesterov momentum value')
train_arg.add_argument('--epochs', type=int, default=2000,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=0.01,
                       help='Initial learning rate value')
train_arg.add_argument('--min_lr', type=float, default=0.00001,
                       help='Min learning rate value')
train_arg.add_argument('--saturate_epoch', type=int, default=800,
                       help='Epoch at which decayed lr will reach min_lr')
train_arg.add_argument('--patience', type=int, default=100,
                       help='Max # of epochs to wait for no validation improv')


# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=False,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data',
                      help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                      help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=False,
                      help='Whether to use tensorboard for visualization')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--print_freq', type=int, default=10,
                      help='How frequently to print training details')
misc_arg.add_argument('--plot_freq', type=int, default=10,
                      help='How frequently to plot glimpses')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
