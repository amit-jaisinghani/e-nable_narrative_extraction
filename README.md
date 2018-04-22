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
* TensorFlow
* pandas
* numpy

To create the environment with the above prequisites run
```bash
conda env create -f environment.yml
```

**Note**: This installs the GPU version of TensorFlow. If it's the CPU version that desire change '_tensorfow-gpu_' to '_tensorflow_' in the environment.yml file.

## Running the code

```bash
$ source activate enable-multiclassifier-env

$ python main.py
```

## Summary