import sys
from bayes_opt import BayesianOptimization
import torch

import utils
import data_loader

from trainer import Trainer
from config import get_config


def main(config):
    utils.prepare_dirs(config)

    if config.is_train_table and config.mem_based_inference:
        print("Error: You can't have is_train_table and mem_based_inference both True")
        sys.exit(1)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}

    # instantiate data loaders
    if config.is_train:
        dloader = data_loader.get_train_valid_loader(
            config.data_dir,
            config.batch_size,
            config.random_seed,
            config.valid_size,
            config.shuffle,
            config.show_sample,
            **kwargs,
        )
    elif config.is_train_table:
        dloader = data_loader.get_train_table_loader(
            config.data_dir, config.batch_size, **kwargs,
        )
    else:
        dloader = data_loader.get_test_loader(
            config.data_dir, config.batch_size, **kwargs,
        )

    trainer = Trainer(config, dloader)

    # either train
    if config.is_train:
        utils.save_config(config)
        trainer.train()
    # or load a pretrained model and test
    elif config.is_train_table:
        trainer.prepare_training_table()
    elif config.mem_based_inference:
        if config.bo:
            if config.all_coefficients:
                trainer.BO_all_coeff()
            else:
                trainer.BO_not_all_coeff()
        else:
            trainer.memory_based_inference_not_all_coeff()
    else:
        import time
        start_test = time.time()
        trainer.test()
        end_test = time.time()
        print("Test time: ", end_test-start_test)


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
