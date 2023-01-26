# IgFold

Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies. Official repository for [IgFold](https://github.com/Graylab/IgFold) 

## Build

Without pyRosetta

`docker build -t igfold .`

With pyRosetta. You need to provide the proper credentials.

`docker build -t igfold --build-arg USERNAME=[value] --build-arg PASSWORD=[value] .`

# Usage

The container provides a REST interface to generate a PDB file from heavy and light chains. At a first usage, the generation of a PDB file may take longer as IgFold needs to download the models.