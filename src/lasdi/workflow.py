import numpy as np
import yaml
import torch
import argparse
import sys
from .enums import *
from .gplasdi import BayesianGLaSDI
from .latent_space import Autoencoder
from .physics.burgers1d import Burgers1D

trainer_dict = {'gplasdi': BayesianGLaSDI}

latent_dict = {'ae': Autoencoder}

physics_dict = {'burgers1d': Burgers1D}

parser = argparse.ArgumentParser(description = "",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('config_file', metavar='string', type=str,
                    help='config file to run LasDI workflow.\n')

def main():
    args = parser.parse_args(sys.argv[1:])
    print("config file: %s" % args.config_file)

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    trainer = initialize_trainer(config)

    if ('restart_file' in config):
        restart_file = np.load(config['restart_file'], allow_pickle=True).item()
        next_step = restart_file['next_step']
        result = restart_file['result']
    else:
        next_step = NextStep.Initial
        result = Result.Unexecuted

    if (result is Result.Fail):
        raise RuntimeError('Previous step has failed. Stopping the workflow.')
    elif (result is Result.Complete):
        print("Workflow is finished.")
        return result

    result = step(trainer, next_step, config)

    return result

def step(trainer, next_step, config):
    # TODO(kevin): implement save/load workflow.
    continue_workflow = True

    if (next_step is NextStep.Initial):

        result, next_step = initial_step(trainer, config)
        if (continue_workflow):
            result = step(trainer, next_step, config)

    elif (next_step is NextStep.Train):

        result, next_step = trainer.train()
        if (result is Result.Complete):
            return result
        else:
            assert(next_step is NextStep.RunSample)
            result = step(trainer, next_step, config)

    elif (next_step is NextStep.RunSample):

        trainer.sample_fom()
        # TODO(kevin): currently no offline fom simulation. skip CollectSample.
        result, next_step = Result.Success, NextStep.Train
        result = step(trainer, next_step, config)

    elif (next_step is NextStep.CollectSample):
        import warnings
        warnings.warn("Collecting sample from offline FOM simulation is not implemented yet")

    else:
        raise RuntimeError("Unknown next step!")

    return result

def initialize_trainer(config):
    '''
    Initialize a LaSDI class with a latent space model according to config file.
    Currently only 'gplasdi' is available.
    '''

    latent_space = initialize_latent_space(config)
    physics = initialize_physics(config)

    trainer_type = config['lasdi']['type']
    assert(trainer_type in config['lasdi'])
    assert(trainer_type in trainer_dict)

    trainer = trainer_dict[trainer_type](latent_space, physics, config['lasdi'][trainer_type])

    return trainer

def initialize_latent_space(config):
    '''
    Initialize a latent space model according to config file.
    Currently only 'ae' (autoencoder) is available.
    '''

    latent_type = config['latent_space']['type']

    assert(latent_type in config['latent_space'])
    assert(latent_type in latent_dict)
    latent_cfg = config['latent_space'][latent_type]

    # TODO(kevin): we might want to pass the config dict directly, for generalization.
    fom_dim = latent_cfg['fom_dimension']
    hidden_units = latent_cfg['hidden_units']
    latent_dim = latent_cfg['latent_dimension']
    latent_space = latent_dict[latent_type](fom_dim, hidden_units, latent_dim)

    return latent_space

def initialize_physics(config):
    '''
    Initialize a physics FOM model according to config file.
    Currently only 'burgers1d' is available.
    '''

    physics_cfg = config['physics']
    physics_type = physics_cfg['type']
    physics = physics_dict[physics_type](physics_cfg)

    return physics

def initial_step(trainer, config):
    from os.path import dirname
    from pathlib import Path

    print("Collecting initial training data..")

    # TODO(kevin): this is a temporary placeholder until parameter class is generalized.
    a_train = np.array([trainer.a_min, trainer.a_max])
    w_train = np.array([trainer.w_min, trainer.w_max])
    a_train, w_train = np.meshgrid(a_train, w_train)
    param_train = np.hstack((a_train.reshape(-1, 1), w_train.reshape(-1, 1)))
    n_train = param_train.shape[0]

    trainer.param_train = param_train
    trainer.X_train = trainer.physics.generate_solutions(param_train)
    trainer.n_init = n_train
    trainer.n_train = n_train

    data_train = {'param_train' : param_train, 'X_train' : trainer.X_train, 'n_train' : n_train}
    train_filename = config['initial_train']['train_data']
    Path(dirname(train_filename)).mkdir(parents=True, exist_ok=True)
    np.save(train_filename, data_train)

    # TODO(kevin): this is a temporary placeholder until parameter class is generalized.
    X_test = trainer.physics.generate_solutions(trainer.param_grid)
    n_test = trainer.param_grid.shape[0]

    data_test = {'param_grid' : trainer.param_grid, 'X_test' : X_test, 'n_test' : n_test}
    test_filename = config['initial_train']['test_data']
    Path(dirname(test_filename)).mkdir(parents=True, exist_ok=True)
    np.save(test_filename, data_test)

    next_step = NextStep.Train
    result = Result.Success

    return result, next_step

if __name__ == "__main__":
    main()