#!/bin/bash
python train.py --optim Adam --pre_trained --epochs 100
cp ../tmp/* ../result/pre_trained/Adam/ -r
python train.py --optim SGD --pre_trained --epochs 100
cp ../tmp/* ../result/pre_trained/SGD/ -r
python train.py --optim RMSprop --pre_trained --epochs 100
cp ../tmp/* ../result/pre_trained/RMSprop/ -r
python train.py --optim Adam --epochs 1000
cp ../tmp/* ../result/no_pre_trained/Adam/ -r
python train.py --optim SGD --epochs 1000
cp ../tmp/* ../result/no_pre_trained/SGD/ -r
python train.py --optim RMSprop --epochs 1000
cp ../tmp/* ../result/no_pre_trained/RMSprop/ -r