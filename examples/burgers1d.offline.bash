#!/usr/bin/bash

check_result () {
  # $1: Result output of the previous command ($?)
  # $2: Name of the previous command
  if [ $1 -eq 0 ]; then
      echo "$2 succeeded"
  else
      echo "$2 failed"
      exit -1
  fi
}

# First stage will be PickSample, which will produce parameter point files in hdf5 format.
# Each lasdi command will save a restart file, which will be read on the next lasdi command.
# So all stages are run by the same command, directed differently by the restart file.
lasdi burgers1d.offline.yml
check_result $? initial-picksample

# Run/save FOM solution with offline FOM solver.
burgers1d burgers1d.offline.yml
check_result $? initial-runsample

# Collect FOM solution.
lasdi burgers1d.offline.yml
check_result $? initial-collect

# Train latent dynamics model.
lasdi burgers1d.offline.yml
check_result $? initial-train

for k in {0..8}
do
    # Pick a new sample from greedy sampling
    lasdi burgers1d.offline.yml
    check_result $? pick-sample

    # A cycle of offline FOM/CollectSamples
    burgers1d burgers1d.offline.yml
    check_result $? run-sample

    lasdi burgers1d.offline.yml
    check_result $? collect-sample

    # Train latent dynamics model.
    lasdi burgers1d.offline.yml
    check_result $? train
done
