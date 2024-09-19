#!/usr/bin/bash

# First stage will be RunSamples, which will produce parameter point files in hdf5 format.
# Each lasdi command will save a restart file, which will be read on the next lasdi command.
# So all stages are run by the same command, directed differently by the restart file.
lasdi burgers1d.offline.yml

# Run/save FOM solution with offline FOM solver.
burgers1d burgers1d.offline.yml

# Collect FOM solution.
lasdi burgers1d.offline.yml

# Train latent dynamics model.
lasdi burgers1d.offline.yml

for k in {0..8}
do
    # Pick a new sample from greedy sampling
    lasdi burgers1d.offline.yml

    # A cycle of RunSamples/offline FOM/CollectSamples
    lasdi burgers1d.offline.yml
    burgers1d burgers1d.offline.yml
    lasdi burgers1d.offline.yml

    # Train latent dynamics model.
    lasdi burgers1d.offline.yml
done
