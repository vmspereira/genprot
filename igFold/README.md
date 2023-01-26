# IgFold

Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies. Official repository for [IgFold](https://github.com/Graylab/IgFold) 

## Build

Without pyRosetta

`docker build -t igfold .`

With pyRosetta. You need to provide the proper credentials.
`docker build -t igfold --build-arg USERNAME=[value] --build-arg PASSWORD=[value] .`