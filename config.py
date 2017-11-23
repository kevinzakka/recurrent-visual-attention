import argparse
from utils import str2bool


arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# network params
net_arg = add_argument_group('Model Params')

# data params
data_arg = add_argument_group('Data Params')

# training params
train_arg = add_argument_group('Training Params')

# other params
misc_arg = add_argument_group('Misc.')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
