FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG USERNAME
ARG PASSWORD

RUN apt-get update && apt-get install -y wget

RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install -c conda-forge pytorch-lightning=1.8.6
RUN conda install pytorch3d=0.7.0 -c pytorch3d
RUN pip install igfold

# Optional install pyRosetta otherwise OpenMM will be used.
 
WORKDIR /pyrosetta
# Install PyRosetta from source
RUN if [[ -n "$USERNAME" ]]; then wget https://$USERNAME:$PASSWORD@graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python38.ubuntu/PyRosetta4.Release.python38.ubuntu.release-337.tar.bz2;
# Extract
RUN if [[ -n "$USERNAME" ]]; then tar -xvjf PyRosetta4.Release.python38.ubuntu.release-337.tar.bz2 && \
    rm PyRosetta4.Release.python38.ubuntu.release-337.tar.bz2 && \
    cd PyRosetta4.Release.python38.ubuntu.release-337 && \
    pip install pyrosetta;
 
