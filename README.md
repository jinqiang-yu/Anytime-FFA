# Anytime Approximate FFA

This repository contains the implementation used in our SAT-24 paper~(Anytime Approximate Formal Feature Attribution). The implementation is an anytime approach to computing approximate formal feature attribution (FFA), by starting with contrastive explanation (CXp) enumeration, and then dynamically switching to abductive explanation (AXp) enumeration when the rate of AXp discovery by CXp enumeration drops. In doing so, the approach is able to quickly get accurate approximations, but also arrive to the full set of AXp’s quicker than pure CXp enumeration. 


## Instruction <a name="instrt"></a>
Before using the implementation, we need to extract the datasets stored in ```datasets.tar.xz```. To extract the datasets, please ensure ```tar``` is installed and run:
```
$ tar -xvf datasets.tar.xz
```

## Table of Content
* **[Required Packages](#require)** 
* **[Usage](#usage)**
	* [Prepare a Dataset](#prepare)
	* [Generate a Boosted Tree](#bt)
	* [Enumerate Abductive Explanations (AXp's) with switch](#enum)
* **[Reproducing Experimental Results](#expr)**

## Required Packages <a name="require"></a>
The implementation is written as a set of Python scripts. The python version used in the experiments is 3.8.5. Some packages are required. To install requirements:
```
$ pip install -r requirements.txt
```

## Usage <a name="usage"></a>

First, change to the source directory
```
$ cd src/
```

### Preparing a dataset <a name="prepare"></a>  <a name="prepare"></a>

The implementation can address datasets in the CSV format. Before enumerating abductive explanations (AXp's) and generating feature attribution, we may need to prepare the datasets to train a BT model. Note that in our experiments we do not need this step.

1. Assume a target dataset is stored in ```somepath/dataset.csv```
2. Create an extra file named ```somepath/dataset.csv.catcol``` containing the indices of the categorical columns ofthe target dataset. For example, if columns ```0```, ```3```, and ```6``` are categorical features, the file should be as follow:
	```
	0
	3
	6
	```
3. With the two files above, we can run:
```
$ python explain.py -p --pfiles dataset.csv,somename somepath/
```
to create a new dataset file `somepath/somename_data.csv` with the categorical features properly addressed.

### Training a gradient boosted tree model  <a name="bt"></a>
A gradient boosted tree model is required before generating a decision set. Run the following command to train a BT model:
```
$ python ./explain.py -o ./btmodels/pneumoniamnist/10,10/ --testsplit 0 -t -n 25 -d 3 ../datasets/pneumoniamnist/10,10/train_origin.csv
```
Here, a boosted tree consisting of 25 trees per class is trained, where the maximum depth of each tree is 3. ```../datasets/pneumoniamnist/10,10/train_origin.csv
 ``` is the dataset to be trained. The value of ```--testsplit``` ranges from 0.0 to 1.0. In this command line, the given dataset is split into 100% to train and 0% to test. ```./btmodels/pneumoniamnist/10,10/``` is the output path to store the trained model. In this example, the generated model is saved in ```./btmodels/pneumoniamnist/10,10/train_origin/train_origin_nbestim_25_maxdepth_3_testsplit_0.0.mod.pkl```


### Enumerating Abductive Explanations (AXp's) with Switch  <a name="enum"></a>
To enumerate abductive or contrastive explanations for BTs, run:
```
$ python -u ./explain.py -e mx --am1 -E -T 1 -z -vvv -R lin --sort abs --htype sat --hsolver formal_mgh --xtype con --switch 1 --sliding 50 --stype ls --gap 2 --diff 1 --unit-mcs --use-mhs --xnum all --cut <int> --explains <dataset.csv> <model.pkl> 
```

Here, parameter ```--cut``` is optional, where the value of ```--cut``` indicate the instance index to enumeration explanations. ```--sliding``` indicates the size of the window. ```--gap``` and ```--diff``` denote the paremeters $\alpha$ in condition (6) and $\varepsilon$ in condition (7) in the submission. By default, all instances in the dataset are considered. ```<dataset.csv>``` and ```<model.pkl>``` specify the dataset and BT model.

For example:

```
$ python -u ./explain.py -e mx --am1 -E -T 1 -z -vvv -R lin --sort abs --htype sat --hsolver formal_mgh --xtype con --switch 1 --sliding 50 --stype ls --gap 2 --diff 1 --unit-mcs --use-mhs --xnum all --cut 17 --explains ../datasets/pneumoniamnist/10,10/test_origin.csv --explain_ formal ./btmodels/pneumoniamnist/10,10/train_origin/train_origin_nbestim_25_maxdepth_3_testsplit_0.0.mod.pkl 
```

The command above will enumerate AXp's with switch for the 18th instance in *PneumoniaMNIST* dataset.


## Reproducing  Experimental Results <a name="expr"></a>
Experimental results can be reproduced by the following script:

```
$ cd ./src/ & ./experiment/repro_exp.sh
```

Since the runtime required to acquire exact FFA is large, running the experiments will take a while.
