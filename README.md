# e-NABLE multi-class classifier

This repo contains a TensorFlow multi-class classifier which classifies Google posts from e-NABLE communitites into the following classes.
* Reporting
* Device
* Delivery
* Progress
* Becoming member
* Attempt Action
* Activity
* Other

## Setting up the environment

### Prequisites
* Keras
* pandas
* numpy

To create the environment with the above prequisites run
```bash
conda env create -f environment.yml
```

**Note**: This installs the GPU version of TensorFlow. If it's the CPU version that desire change '_tensorfow-gpu_' to '_tensorflow_' in the environment.yml file.

### Required files
* Glove 100d embeddings
To download and extract glove.6B.100d.txt file
```bash
sudo wget http://nlp.stanford.edu/data/glove.6B.zip && unzip -j glove.6B.zip glove.6B.100d.txt && sudo rm glove.6B.zip
```

## Running the code

```bash
$ source activate enable-multiclassifier-env

$ makefile
```