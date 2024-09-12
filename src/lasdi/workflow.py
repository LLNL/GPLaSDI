import numpy as np
import yaml
import torch
import argparse
import sys
from .enums import *
from .gplasdi import BayesianGLaSDI
from .latent_space import Autoencoder
from .latent_dynamics.sindy import SINDy
from .physics.burgers1d import Burgers1D
from .param import ParameterSpace
from .inputs import InputParser

trainer_dict = {'gplasdi': BayesianGLaSDI}

latent_dict = {'ae': Autoencoder}

ld_dict = {'sindy': SINDy}

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
        cfg_parser = InputParser(config, name='main')

    use_restart = cfg_parser.getInput(['workflow', 'use_restart'], fallback=False)
    if (use_restart):
        restart_filename = cfg_parser.getInput(['workflow', 'restart_file'], datatype=str)
        from os.path import dirname
        from pathlib import Path
        Path(dirname(restart_filename)).mkdir(parents=True, exist_ok=True)

    import os
    if (use_restart and (os.path.isfile(restart_filename))):
        # TODO(kevin): in long term, we should switch to hdf5 format.
        restart_file = np.load(restart_filename, allow_pickle=True).item()
        current_step = restart_file['next_step']
        result = restart_file['result']
    else:
        restart_file = None
        current_step = NextStep.Initial
        result = Result.Unexecuted
    
    trainer, param_space, physics, latent_space, latent_dynamics = initialize_trainer(config, restart_file)

    result, next_step = step(trainer, current_step, config, use_restart)

    if (result is Result.Fail):
        raise RuntimeError('Previous step has failed. Stopping the workflow.')
    elif (result is Result.Success):
        print("Previous step succeeded. Preparing for the next step.")
        result = Result.Unexecuted
    elif (result is Result.Complete):
        print("Workflow is finished.")

    # save restart (or final) file.
    import time
    date = time.localtime()
    date_str = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}"
    date_str = date_str.format(month = date.tm_mon, day = date.tm_mday, year = date.tm_year, hour = date.tm_hour + 3, minute = date.tm_min)
    if (use_restart):
        # rename old restart file if exists.
        if (os.path.isfile(restart_filename)):
            old_timestamp = restart_file['timestamp']
            os.rename(restart_filename, restart_filename + '.' + old_timestamp)
        save_file = restart_filename
    else:
        save_file = 'lasdi_' + date_str + '.npy'
    
    save_dict = {'parameters': param_space.export(),
                 'physics': physics.export(),
                 'latent_space': latent_space.export(),
                 'latent_dynamics': latent_dynamics.export(),
                 'trainer': trainer.export(),
                 'timestamp': date_str,
                 'next_step': next_step,
                 'result': result, # TODO(kevin): do we need to save result?
                 }
    
    np.save(save_file, save_dict)

    return result

def step(trainer, next_step, config, use_restart=False):

    if (next_step is NextStep.Initial):

        result, next_step = initial_step(trainer, config)

    elif (next_step is NextStep.Train):

        result, next_step = trainer.train()

    elif (next_step is NextStep.RunSample):

        trainer.sample_fom()
        # TODO(kevin): currently no offline fom simulation. skip CollectSample.
        result, next_step = Result.Success, NextStep.Train
        # result = step(trainer, next_step, config)

    elif (next_step is NextStep.CollectSample):
        import warnings
        warnings.warn("Collecting sample from offline FOM simulation is not implemented yet")
        result, next_step = Result.Success, NextStep.RunSample

    else:
        raise RuntimeError("Unknown next step!")
    
    # if fail or complete, break the loop regardless of use_restart.
    if ((result is Result.Fail) or (result is Result.Complete)):
        return result, next_step
    
    # continue the workflow if not using restart.
    if (not use_restart):
        result, next_step = step(trainer, next_step, config)

    return result, next_step

def initialize_trainer(config, restart_file=None):
    '''
    Initialize a LaSDI class with a latent space model according to config file.
    Currently only 'gplasdi' is available.
    '''

    # TODO(kevin): load parameter train space from a restart file.
    param_space = ParameterSpace(config)
    if (restart_file is not None):
        param_space.load(restart_file['parameters'])

    physics = initialize_physics(param_space, config)
    latent_space = initialize_latent_space(physics, config)
    if (restart_file is not None):
        latent_space.load(restart_file['latent_space'])

    # do we need a separate routine for latent dynamics initialization?
    ld_type = config['latent_dynamics']['type']
    assert(ld_type in config['latent_dynamics'])
    assert(ld_type in ld_dict)
    latent_dynamics = ld_dict[ld_type](latent_space.n_z, physics.nt, config['latent_dynamics'])
    if (restart_file is not None):
        latent_dynamics.load(restart_file['latent_dynamics'])

    trainer_type = config['lasdi']['type']
    assert(trainer_type in config['lasdi'])
    assert(trainer_type in trainer_dict)

    trainer = trainer_dict[trainer_type](physics, latent_space, latent_dynamics, config['lasdi'][trainer_type])
    if (restart_file is not None):
        trainer.load(restart_file['trainer'])

    return trainer, param_space, physics, latent_space, latent_dynamics

def initialize_latent_space(physics, config):
    '''
    Initialize a latent space model according to config file.
    Currently only 'ae' (autoencoder) is available.
    '''

    latent_type = config['latent_space']['type']

    assert(latent_type in config['latent_space'])
    assert(latent_type in latent_dict)
    
    latent_cfg = config['latent_space'][latent_type]
    latent_space = latent_dict[latent_type](physics, latent_cfg)

    return latent_space

def initialize_physics(param_space, config):
    '''
    Initialize a physics FOM model according to config file.
    Currently only 'burgers1d' is available.
    '''

    physics_cfg = config['physics']
    physics_type = physics_cfg['type']
    physics = physics_dict[physics_type](param_space, physics_cfg)

    return physics

def initial_step(trainer, config):
    from os.path import dirname
    from pathlib import Path

    print("Collecting initial training data..")

    trainer.X_train = trainer.physics.generate_solutions(trainer.param_space.train_space)

    data_train = {'param_train' : trainer.param_space.train_space,
                  'X_train' : trainer.X_train,
                  'n_train' : trainer.param_space.n_train}
    train_filename = config['initial_train']['train_data']
    Path(dirname(train_filename)).mkdir(parents=True, exist_ok=True)
    np.save(train_filename, data_train)

    if (trainer.param_space.test_space is not None):
        X_test = trainer.physics.generate_solutions(trainer.param_space.test_space)

        data_test = {'param_grid' : trainer.param_space.test_space,
                     'X_test' : X_test,
                     'n_test' : trainer.param_space.n_test}
        test_filename = config['initial_train']['test_data']
        Path(dirname(test_filename)).mkdir(parents=True, exist_ok=True)
        np.save(test_filename, data_test)

    next_step = NextStep.Train
    result = Result.Success

    return result, next_step

if __name__ == "__main__":
    main()