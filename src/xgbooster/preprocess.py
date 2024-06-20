#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## preprocess.py
##
##

#
#==============================================================================
import json
import numpy as np
from xgbooster import XGBooster
import math
import pandas as pd
import numpy as np
import sklearn
import pickle
import collections

#
#==============================================================================
def preprocess_dataset(raw_data_path, files):
    print("preprocess dataset from ", raw_data_path)
    files = files.split(",")

    data_file = files[0]
    dataset_name = files[1]

    try:
        data_raw = pd.read_csv(raw_data_path + data_file, sep=',', na_values=  [''])
        try:
            catcols = pd.read_csv(raw_data_path + data_file + ".catcol", header = None)
        except:
            # no categorical features
            catcols = pd.DataFrame([[]])
        categorical_features = np.concatenate(catcols.values).tolist()

        for i in range(len(data_raw.values[0])):
            if i in categorical_features:
                data_raw.fillna('',inplace=True)
            else:
                data_raw.fillna(0,inplace=True)
        dataset_all = data_raw
        dataset = dataset_all.values.copy()

    except Exception as e:
        print("Please provide info about categorical columns/original datasets or omit option -p", e)
        exit()

    # move categrorical columns forward

    feature_names = dataset_all.columns
    #print(feature_names)

    ##############################
    extra_info = {}
    categorical_names = {}
    #print(categorical_features)
    dataset_new = dataset_all.values.copy()
    for feature in categorical_features:
        #print("feature", feature)
        #print(dataset[:, feature])
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(dataset[:, feature])
        categorical_names[feature] = le.classes_
        dataset_new[:, feature] = le.transform(dataset[:, feature])

    ###################################3
    # target as categorical
    labels_new = []

    le = sklearn.preprocessing.LabelEncoder()
    le.fit(dataset[:, -1])
    dataset_new[:, -1]= le.transform(dataset[:, -1])
    class_names = le.classes_
    ######################################33


    if (False):
        dataset_new = np.delete(dataset_new, -1, axis=1)
        oneencoder = sklearn.preprocessing.OneHotEncoder()
        oneencoder.fit(dataset_new[:, categorical_features])
        #print(oneencoder.categories_)
        n_transformed_features = sum([len(cats) for cats in oneencoder.categories_])
        #print(n_transformed_features)
        #print(dataset_new.shape)
        X = dataset_new[:,categorical_features][0]
        #print(X)
        x = np.expand_dims(X, axis=0)
        #print("x", x, x.shape)
        y = dataset_new[0].copy()
        #print(y.shape, oneencoder.transform(x).shape)
        y[categorical_features] = oneencoder.transform(x).toarray()

        #print("y", y, y.shape)

        z = oneencoder.inverse_transform(y)
        #print(z.shape)
        exit()

    ###########################################################################3
    extra_info = {"categorical_features": categorical_features,
                  "categorical_names": categorical_names,
                  "feature_names": feature_names,
                  "class_names": class_names}

    new_file_train = raw_data_path + dataset_name + '_data.csv'
    df = pd.DataFrame(data=dataset_new)
    df.columns = list(feature_names)
    df.to_csv(new_file_train, mode = 'w', index=False)
    print("new dataset", new_file_train)


    f =  open(raw_data_path + dataset_name + '_data.csv.pkl', "wb")
    pickle.dump(extra_info, f)
    f.close()

def discretize_dataset(raw_data_path, files, options):

    #iscomplete = False if '_train' in files else True

    #if iscomplete:
    #    train1 = options.files[1]
    #    intvs = collections.defaultdict(lambda: set())
    #    for i in range(1, 6):
    #        options.files[1] = train1.replace('_train1', '_train{0}'.format(i))
    #        xgb = XGBooster(options, from_model=options.files[1])
    #        # encode it and save the encoding to another file
    #        xgb.encode(test_on=options.explain)
    #        for f in xgb.intvs:
    #            intvs[f] = intvs[f].union(xgb.intvs[f][:-1])
    #    intvs = {f: sorted(intvs[f]) + ['+'] for f in intvs}
    #else:
    #    xgb = XGBooster(options, from_model=options.files[1])
    #    xgb.encode(test_on=options.explain)
    #    intvs = xgb.intvs

    xgb = XGBooster(options, from_model=options.files[1])
    xgb.encode()#test_on=options.explain)
    intvs = xgb.intvs
    try:
        numeric_features = set(range(xgb.nb_features)).difference(set(xgb.categorical_names.keys()))
    except:
        numeric_features = set(range(xgb.nb_features))

    #files = files.split(",")

    #print('xgb.intvs:')
    #print(xgb.intvs)
    #print()
    #print('xgb.categorical_names:')
    #print(xgb.categorical_names)
    #print()
    #print('xgb.categorical_features:')
    #print(xgb.categorical_features)
    #print()
    #print('xgb.feature_names:')
    #print(xgb.feature_names)
    #print('xgb.target_name:')
    #print(xgb.target_name)
    #print()
    #print('xgb.categorical_names.keys():')
    #print(xgb.categorical_names.keys())

    #data_file = files[0]
    dataset_name = files

    columns = xgb.feature_names + [xgb.class_name]
    #print(data_raw.columns)

    keep = {int(f.split('_')[0][1:]) for f in xgb.intvs}
    keep.add(xgb.nb_features)
    remove = set(range(xgb.nb_features+1)).difference(keep)

    new_dataset_info = {'feature_names': columns,
                        'expanded_feature_names': [],
                        'features': {},
                        'numerical_names': {}
                        }

    new_dataset_info['feature_names'] = [new_dataset_info['feature_names'][i] for i in range(xgb.nb_features+1) if i not in remove]

    ii = 0
    #for i in range(len(dataset[0])):
    for i in range(xgb.nb_features+1):
        if i not in keep:
            continue

        new_dataset_info['features'][ii] = []

        if i in numeric_features:
            new_dataset_info['numerical_names'][ii] = []

            intervals = intvs['f{0}'.format(i)][:-1]
            for inv_id, thredshole in enumerate(intervals):
                if inv_id == 0:
                    name = '{0} < {1}'.format(columns[i], thredshole)
                else:
                    name = '{0} <= {1} < {2}'.format(intervals[inv_id-1], columns[i], thredshole)

                new_dataset_info['expanded_feature_names'].append(name)
                new_dataset_info['features'][ii].append(len(new_dataset_info['expanded_feature_names']) - 1)
                new_dataset_info['numerical_names'][ii].append(name)

            name = '{0} >= {1}'.format(columns[i], intervals[-1])
            new_dataset_info['expanded_feature_names'].append(name)
            new_dataset_info['features'][ii].append(len(new_dataset_info['expanded_feature_names']) - 1)
            new_dataset_info['numerical_names'][ii].append(name)
        else:
            new_dataset_info['expanded_feature_names'].append(columns[i])
            new_dataset_info['features'][ii].append(len(new_dataset_info['expanded_feature_names']) - 1)
        ii += 1
    # print('intvs:')
    # print(xgb.intvs)
    # print()
    # print('xgb.categorical_names:')
    # print(xgb.categorical_names)
    # print()
    # print('xgb.categorical_features:')
    # print(xgb.categorical_features)
    # print()
    # print('xgb.feature_names:')
    # print(xgb.feature_names)
    # print('xgb.target_name:')
    # print(xgb.target_name)
    # print()
    print(raw_data_path)
    print(dataset_name)
    dataset = pd.read_csv(raw_data_path + dataset_name + '.csv', sep=',', na_values=['']).to_numpy()

    #if len(xgb.X_test) > 0:
    #    # test data info in xgb
    #    datasets['test'] =  np.append(xgb.X_test, np.array([[y] for y in xgb.Y_test]), axis=1)
    #    datasets['complete'] = np.append(xgb.X, np.array([[y] for y in xgb.Y]), axis=1)

    new_dataset = []
    for row in dataset:
        new_row = []
        wght = 1
        #else:
        #    wght = xgb.wghts[tuple(row)]

        for i in range(len(row[:-1])):
            if i in remove:
                continue
            val = row[i]
            if i in numeric_features:
                intervals = intvs['f{0}'.format(i)][:-1]
                num_row = []
                for inv_id, thredshole in enumerate(intervals):
                    if inv_id == 0:
                        #name = '{0} < {1}'.format(columns[i], thredshole)
                        match = val < thredshole
                    else:
                        #name = '{0} <= {1} < {2}'.format(intervals[inv_id - 1], columns[i], thredshole)
                        match = intervals[inv_id-1] <= val < thredshole
                    num_row.append(1 if match else 0)

                # for val >= max thredshole
                num_row.append(1 if val >= intervals[-1] else 0)
                assert sum(num_row) == 1
                new_row.extend(num_row)
            else:
                new_row.append(val)
        # print('xgb.target_name:')
        # print(xgb.target_name)
        new_row.append(row[-1])
        new_dataset.extend([new_row] * wght)

    new_file = raw_data_path + dataset_name + '_discrete.csv'
    df = pd.DataFrame(data=new_dataset, columns=new_dataset_info['expanded_feature_names'])
    df.to_csv(new_file, mode='w', index=False)
    print("new dataset", new_file)

    # also a pickle file to store features
    f = open(new_file + '.pkl', "wb")
    pickle.dump(new_dataset_info, f)
    f.close()