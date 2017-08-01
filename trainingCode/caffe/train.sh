#! /bin/bash

CAFFE=/home/chenqi/workspace/caffe/python

source /home/chenqi/.py2env/bin/activate

python $CAFFE/train.py \
    --solver solver_DilatedNet.prototxt \
    --gpus 0 1 2 3
    # --timing
