FROM ubuntu:22.04

ENV ENVDIR=env

# install sudo
RUN apt-get -yq update && apt-get -yq install sudo

WORKDIR /$ENVDIR

# install packages
RUN sudo apt-get install -yq git
RUN sudo apt-get install --no-install-recommends -yq make gcc gfortran libssl-dev cmake
RUN sudo apt-get install -yq libopenblas-dev libmpich-dev libblas-dev liblapack-dev libscalapack-mpi-dev libhdf5-mpi-dev hdf5-tools
# RUN sudo apt-get install -yq vim
RUN sudo apt-get install -yq git-lfs
# RUN sudo apt-get install -yq valgrind
RUN sudo apt-get install -yq wget
# RUN sudo apt-get install -yq astyle

# install python
RUN sudo apt-get install -yq python3
RUN sudo apt-get install -yq python3-dev
RUN sudo apt-get install -yq python3-pip
RUN sudo apt-get install python-is-python3
RUN sudo python -m pip install --upgrade pip
# RUN sudo python -m pip install sphinx sphinx-autoapi sphinx_rtd_theme
RUN sudo pip3 install torch==2.0.1 numpy==1.23.0 scikit-learn==1.3 scipy==1.10 pyyaml==6.0 matplotlib==3.8.0 argparse==1.1 h5py pytest pytest-cov
#
RUN sudo apt-get clean -q

# create and switch to a user
ENV USERNAME=test
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN useradd --no-log-init -u 1001 --create-home --shell /bin/bash $USERNAME
RUN adduser $USERNAME sudo
USER $USERNAME
WORKDIR /home/$USERNAME
