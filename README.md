# seal-inference-experiment
Privacy preserving CNN inference over homomorphic encryption (using [Microsoft SEAL version 3.6.6](https://github.com/microsoft/SEAL/tree/3.6.6))

Supports both latency-oriented single image inference (channel-wise packing) and throughput-oriented multiple images inference (batch-axis packing).

## Directory structure
```
seal-inference-experiment/
├── Dockerfile
├── build_docker.sh
│
├── bin/                  # Generated executable files
│
├── datasets/
│   ├── cifar-10/
│   └── mnist/
│
├── include/              # Library header files
│
├── secrets/              # Generated keys and parameter of SEAL (by gen_keys.cpp)
│
├── src/                  # Codes for secure inference (C++)
│   ├── cnn/
│   ├── gen_keys.cpp
│   ├── main.cpp
│   └── utils/
│
└── train_model/          # Codes for training model w/ PyTorch (Python)
    ├── cifar-10/
    │   ├── *.py
    │   └── saved_models  # Model structure(.json) and parameters(.h5)
    ├── mnist/
    │   ├── *.py
    │   └── saved_models  # Model structure(.json) and parameters(.h5)
    ├── poetry.lock
    ├── pyproject.toml
    └── utils/
```

## Dataset
- MNIST (http://yann.lecun.com/exdb/mnist/)
- CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html)

## Build
You can build program for secure inference using Docker
```
$ bash build_docker.sh
```

You can install required packages for training through [Poetry](https://github.com/python-poetry/poetry)
```
$ cd train_model
$ poetry install
```

## Run example
1. Start docker container
```
$ docker run --privileged -it seal-inference-experiment /bin/bash
```
2. Generate keys (gen_keys.cpp)
```
$ ./bin/gen_keys -N 16384 -L 5 --q0 50 --qi 30 --ql 60 --prefix N16384_L5_50-30-60 --dataset mnist
```
3. Execute secure inference program (main.cpp)
```
$ OMP_NUM_THREADS=72 ./bin/main -P N16384_L5_50-30-60 -D mnist -M 3layer_cnn-square-BN --model-params 3layer_cnn-square-BN-99.35_200epoch-1223_1032-params.h5 -A square --fuse-layer --mode single --images 20
```

