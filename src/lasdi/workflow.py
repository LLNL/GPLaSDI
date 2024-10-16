# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import  yaml
import  numpy                   as      np
import  torch
import  argparse
import  sys
from    .enums                  import  NextStep, Result
from    .gplasdi                import  BayesianGLaSDI
from    .latent_space           import  Autoencoder
from    .latent_dynamics        import  LatentDynamics
from    .latent_dynamics.sindy  import  SINDy
from    .physics                import  Physics
from    .physics.burgers1d      import  Burgers1D
from    .param                  import  ParameterSpace



# -------------------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------------------

# Dictionaries 
trainer_dict    = {'gplasdi'    : BayesianGLaSDI}
latent_dict     = {'ae'         : Autoencoder}
ld_dict         = {'sindy'      : SINDy}
physics_dict    = {'burgers1d'  : Burgers1D}

# Set up the argument parser. We add one argument, "config_file". This should be a .yml file that 
# specifies how to run a particular LaSDI experiment.
parser = argparse.ArgumentParser(description        = "",
                                 formatter_class    = argparse.RawTextHelpFormatter)

parser.add_argument('config_file', 
                    metavar     = 'string', 
                    type        = str,
                    help        = 'config file to run LasDI workflow.\n')



# -------------------------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------------------------

def main() -> Result:
    # Fetch the arguments and report what we loaded.
    args = parser.parse_args(sys.argv[1:])
    print("config file: %s" % args.config_file)

    # Now, open the configuration file
    with open(args.config_file, 'r') as f:
        config : dict = yaml.safe_load(f)

    # Use the loaded settings to initialize a trainer object.
    trainer = initialize_trainer(config)

    # Determine what the next step to simulate is. If we are loading from a restart, then the 
    # restart should have logged then next step. Otherwise, the next step is initialization.
    if ('restart_file' in config):
        restart_file    = np.load(config['restart_file'], allow_pickle = True).item()
        next_step       = restart_file['next_step']
        result          = restart_file['result']
    else:
        next_step   = NextStep.Initial
        result      = Result.Unexecuted

    # Make sure that the saved simulation didn't fail. If it did, we can not proceed. Likewise, if
    # it has completed, then there is nothing to do.
    if (result is Result.Fail):
        raise RuntimeError('Previous step has failed. Stopping the workflow.')
    elif (result is Result.Complete):
        print("Workflow is finished.")
        return result

    # Now that we have set up everything, run the next step.
    result = step(trainer, next_step, config)

    # All done!
    return result;



def step(trainer : BayesianGLaSDI, next_step : NextStep, config : dict) -> Result:
    """
    

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------
    
    trainer: A Trainer class object that we use when training the model for a particular instance
    of the settings.

    next_step: a NextStep object (a kind of enumeration) specifying what the next step is. 

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models.

    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A Returns object (a kind of enumeration) that indicates what happened during the current step.
    """

    # TODO(kevin): implement save/load workflow.
    continue_workflow = True

    # What we do depends entirely on what the next step is... we handle each case separately.
    if (next_step is NextStep.Initial):
        # If the next step is "Initial", then we need to generate the fom solution for each 
        # combination of parameter values in the testing/training set (and save those fom solutions 
        # to file). We do this using the "initial_step" method.
        result, next_step = initial_step(trainer, config)

        # If we should continue training, the go onto the next step!
        if (continue_workflow):
            result = step(trainer, next_step, config)
    
    elif (next_step is NextStep.Train):
        # If our next step is to train, then let's train!
        result, next_step = trainer.train()

        # Check what to do next. Either we are done or will have generated new parameter 
        # combinations to train at. In the latter case, we need to generate the fom solutions for 
        # the new parameter values (this is the "RunSample" element of NextStep).
        if (result is Result.Complete):
            return result
        else:
            assert(next_step is NextStep.RunSample)
            result = step(trainer, next_step, config)

    elif (next_step is NextStep.RunSample):
        # If we have trained but generated new parameter values to train at, then we need to 
        # generate the fom solutions for these new parameter values. We do this using the 
        # sample_fom method.
        trainer.sample_fom()
        
        # TODO(kevin): currently no offline fom simulation. skip CollectSample.
        
        # Update the Results, NextStep methods. Now that we have data for the new fom solutions, we
        # need to train at the new parameter values.
        result, next_step = Result.Success, NextStep.Train
        result = step(trainer, next_step, config)
    
    elif (next_step is NextStep.CollectSample):
        import warnings
        warnings.warn("Collecting sample from offline FOM simulation is not implemented yet")

    else:
        raise RuntimeError("Unknown next step!")

    # All done!
    return result



def initialize_trainer(config : dict) -> BayesianGLaSDI:
    '''
    Initialize a LaSDI class with a latent space model according to config file.
    Currently only 'gplasdi' is available.

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models. We expect this dictionary 
    to contain the following keys (if a key is within a dictionary that is specified by another 
    key, then we tab the sub-key relative to the dictionary key): 
        - physics           (used by "initialize_physics")
            - type
        - latent_dynamics   (how we parameterize the latent dynamics; e.g. sindy)
            - type
        - lasdi
            - type
    
            
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A "BayesianGLaSDI" object that has been initialized using the settings in config/is ready to 
    begin training.
    '''

    # TODO(kevin): load parameter train space from a restart file.

    # Set up a parameter space object. This will keep track of all parameter combinations we want
    # to try during testing and training. We load the set of possible parameters and their possible
    # values using the configuration file.
    param_space : ParameterSpace = ParameterSpace(config)
    
    # Get the "physics" object we need to generate the dataset.
    physics         : Physics           = initialize_physics(param_space, config)

    # Determine what kind of model we want to use to learn the latent dynamics.
    latent_space    : torch.nn.Module   = initialize_latent_space(physics, config)

    # do we need a separate routine for latent dynamics initialization?
    ld_type = config['latent_dynamics']['type']
    assert(ld_type in config['latent_dynamics'])
    assert(ld_type in ld_dict)
    latent_dynamics : LatentDynamics = ld_dict[ld_type](latent_space.n_z, physics.nt, config['latent_dynamics'])

    # Fetch the trainer type. Note that only "gplasdi" is allowed.
    trainer_type : dict = config['lasdi']['type']
    assert(trainer_type in config['lasdi']) 
    assert(trainer_type in trainer_dict)

    # Initialize the trainer object. 
    trainer : BayesianGLaSDI = trainer_dict[trainer_type](physics, latent_space, latent_dynamics, config['lasdi'][trainer_type])

    # All done!
    return trainer



def initialize_latent_space(physics : Physics, config : dict) -> torch.nn.Module:
    '''
    Initialize a latent space model according to config file.
    Currently only 'ae' (autoencoder) is available.
    

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    physics: A "Physics" object that allows us to generate the dataset. It corresponds to a 
    specific equation that we then solve. 

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models. We expect this dictionary 
    to contain the following keys (if a key is within a dictionary that is specified by another 
    key, then we tab the sub-key relative to the dictionary key): 
        - latent_space
            - type
    
            
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A torch.nn.Module object that is designed to act as the trainable model in the gplasdi 
    framework. This model should have a latent space of some form. This latent space is where we
    '''

    # First, determine what model we are using in the latent dynamics. Make sure the user 
    # included all the information that is necessary to initialize the corresponding dynamics.
    latent_type : str = config['latent_space']['type']
    assert(latent_type in config['latent_space'])
    assert(latent_type in latent_dict)
    
    # Next, initialize the latent space.
    latent_cfg      : dict              = config['latent_space'][latent_type]
    latent_space    : torch.nn.Module   = latent_dict[latent_type](physics, latent_cfg)

    # All done!
    return latent_space



def initialize_physics(param_space : ParameterSpace, config: dict) -> Physics:
    '''
    Initialize a physics FOM model according to config file.
    Currently only 'burgers1d' is available.

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    param_space: The "ParameterSpace" object we loaded using the settings in "config"

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models. We expect this dictionary 
    to contain the following keys (if a key is within a dictionary that is specified by another 
    key, then we tab the sub-key relative to the dictionary key): 
        - physics 
            - type


            
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A "Physics" object initialized using the parameters in the config['physics'] dictionary. 
    '''

    # First, determine what kind of "physics" object we want to load.
    physics_cfg     : dict      = config['physics']
    physics_type    : str       = physics_cfg['type']

    # Next, initialize the "physics" object we are using to build the simulations.
    physics         : Physics   = physics_dict[physics_type](param_space, physics_cfg)

    # All done!
    return physics



def initial_step(trainer : BayesianGLaSDI, config : dict) -> tuple[Result, NextStep]:
    """
    This function generates the data for the testing and training sets. It also saves this data
    to file (so that we can access it in future steps). How does this work? The testing set and 
    training set consist of sets of combinations of parameter values. For each combination of 
    parameter values in the training/testing sets, we solve the underlying physics using that 
    combination of parameters. We then store these results in a pair of gigantic arrays (one for
    the testing set, another for the training one). We save these arrays + the set of combinations
    of parameter testing/training values to file.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    trainer: A trainer type object which represents a particular equation and stores the sets of 
    combinations of parameter values for the testing and training sets. This object also has a way 
    to solve the underlying physics for a particular combination of parameter values. This should
    be an object returned by the "initialize_physics" function.

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models.
    

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A two element tuple. Both are enumerations. The first is a Returns type object which specifies
    if we were able to generate the data successfully or not. The second is a NextStep object which
    specifies what the next step in the workflow is.
    """

    # Imports
    from os.path import dirname
    from pathlib import Path

    print("Collecting initial training data..")

    # Use the physics object to generate the training dataset. To do this, we use a numerical
    # solver to solve the underlying equation. This function returns a Nd + 2 dimensional 
    # array, where Nd is the number of spatial dimensions in the problem domain to the equation 
    # that trainer.physics represents. It should have shape [Np, Nt, Nx[0], ... , Nx[Nd - 1]], 
    # where Np is the number of distinct parameter combinations in the training set, Nt is the 
    # number of time steps per fom solution, and Nx[0], ... , Nx[Nd - 1] represent the number of 
    # steps along the spatial axes. 
    trainer.X_train = trainer.physics.generate_solutions(trainer.param_space.train_space)

    # Save the fom solutions for each parameter combination + the parameter values in each 
    # combination to file.
    data_train = {'param_train' : trainer.param_space.train_space,
                  'X_train'     : trainer.X_train,
                  'n_train'     : trainer.param_space.n_train}
    train_filename : str = config['initial_train']['train_data']
    Path(dirname(train_filename)).mkdir(parents = True, exist_ok = True)
    np.save(train_filename, data_train)

    # If we have a test set, then that set consists of a set of parameter values. We need to 
    # solve the underlying physics for each one of those sets of parameter values.
    if (trainer.param_space.test_space is not None):
        # Generate the training fom solutions. As with the training set, this returns a Nd + 2 
        # dimensional array of shape [Np, Nt,  Nx[0], ... , Nx[Nd - 1]], except here Np represents
        # the number of distinct parameter combinations in the testing set.
        X_test = trainer.physics.generate_solutions(trainer.param_space.test_space)

        # Save the results for each parameter combination + the parameter values themselves to 
        # file.
        data_test = {'param_grid'   : trainer.param_space.test_space,
                     'X_test'       : X_test,
                     'n_test'       : trainer.param_space.n_test}
        test_filename = config['initial_train']['test_data']
        Path(dirname(test_filename)).mkdir(parents = True, exist_ok = True)
        np.save(test_filename, data_test)

    # Finally, update the next step and results.
    next_step   : NextStep  = NextStep.Train
    result      : Result    = Result.Success

    # All done!
    return result, next_step



if __name__ == "__main__":
    main()