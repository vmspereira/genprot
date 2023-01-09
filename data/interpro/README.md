
Interpro sequences download and alignment tool
==============================================

The docker file allows to easily download, align and trim by length amino acid sequences from Interpro, as well as generate HMM profiles, given an Interpro assertion identifier.

## Build the image and use the image

```
docker build -t interpro .
docker run -v [PATH]/data:/data interpro [ASSERTION_ID] 
```
Optionally, you may add a sequence maximum length:

```
docker run -v [PATH]/data:/data interpro [ASSERTION_ID] [MAX_LENGTH]
```
  
