FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG USERNAME
ARG PASSWORD

RUN apt-get update && apt-get install -y wget

RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install pytorch3d=0.7.0 -c pytorch3d
RUN pip install pytorch-lightning==1.8.6 igfold
RUN pip install Flask==2.2.2 gunicorn==20.1.0

# Optional install pyRosetta otherwise OpenMM will be used.
 
WORKDIR /pyrosetta
# Install PyRosetta from source
RUN if [[ -n $USERNAME  &&  -n $PASSWORD ]]; \ 
    then wget https://$USERNAME:$PASSWORD@graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python38.ubuntu/PyRosetta4.Release.python38.ubuntu.release-337.tar.bz2 ; \
    else echo 'passing'; fi

# Extract
RUN if [[ -n $USERNAME && -n $PASSWORD ]] ; \ 
    then tar -xvjf PyRosetta4.Release.python38.ubuntu.release-337.tar.bz2 && \
    rm PyRosetta4.Release.python38.ubuntu.release-337.tar.bz2 && \
    cd PyRosetta4.Release.python38.ubuntu.release-337 && \
    pip install pyrosetta ;\
    else echo 'passing'; fi
 
WORKDIR /app
COPY app.py /app/app.py
EXPOSE 5000
CMD ["gunicorn","--bind", "0.0.0.0:5000", "--workers", "4","--timeout","300","app:app"]
