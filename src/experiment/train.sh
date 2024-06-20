#!/bin/sh

# preprocess data
python ./explain.py -p --pfiles complete_origin.csv,complete_origin ../datasets/mnist/10,10/1,3/
python ./explain.py -p --pfiles complete_origin.csv,complete_origin ../datasets/mnist/10,10/1,7/
python ./experiment/process.py

# train BT models
python ./explain.py -o ./btmodels/mnist/10,10/1,3/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/mnist/10,10/1,3/train_origin_data.csv
python ./explain.py -o ./btmodels/mnist/10,10/1,7/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/mnist/10,10/1,7/train_origin_data.csv
python ./explain.py -o ./btmodels/pneumoniamnist/10,10/ --testsplit 0 -t -n 25 -d 3 ../datasets/pneumoniamnist/10,10/train_origin.csv
python ./explain.py -o ./btmodels/sarcasm/ --testsplit 0 -t -n 25 -d 5 ../datasets/sarcasm/sarcasm_process_train.csv
python ./explain.py -o ./btmodels/disaster-tweets/ --testsplit 0 -t -n 40 -d 4 ../datasets/disaster-tweets/disaster-tweets_process_train.csv
