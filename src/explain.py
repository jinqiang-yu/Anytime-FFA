#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## explain.py
##

#
#==============================================================================
from __future__ import print_function
from data import Data
from options import Options
import os
import sys
from xgbooster import XGBooster, preprocess_dataset, discretize_dataset
import random

#
#==============================================================================
#==============================================================================
if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)
    random.seed(1234)
    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    if (options.preprocess_categorical):
        if options.encode and len(options.files) > 1:
            discretize_dataset(options.files[0], options.preprocess_categorical_files, options)
        else:
            preprocess_dataset(options.files[0], options.preprocess_categorical_files)

        exit()

    if options.files:
        xgb = None

        if options.train:
            if not os.path.isdir(options.output):
                os.makedirs(options.output)
            data = Data(filename=options.files[0], mapfile=options.mapfile,
                    separator=options.separator,
                    use_categorical = options.use_categorical)

            xgb = XGBooster(options, from_data=data)
            train_accuracy, test_accuracy, model = xgb.train()

        if 'mnist' in options.files[0].lower():
            options.isimg = True
            if '10,10' in options.files[0]:
                options.shape = (10, 10)
            else:
                options.shape = (28, 28)
        else:
            options.isimg = False

        # reading all samples
        if options.explains:
            with open(options.explains, 'r') as fp:
                lines = fp.readlines()[1:]
            lines = list(map(lambda l: l[: l.rfind(',')], lines))
        else:
            lines = [options.explain]
        lines = lines[:100]
        inst_ids = list(range(len(lines)))

        if options.cut is not None:
            lines = [lines[options.cut]]
            inst_ids = [options.cut]

        if options.encode:
            if not xgb:
                xgb = XGBooster(options, from_model=options.files[0])
                # encode it and save the encoding to another file
                # xgb.encode(test_on=options.explain)
                xgb.encode()

        for inst_id, s in zip(inst_ids, lines):
            #if options.cut is not None:
            #    if inst_id != options.cut:
            #        continue
            options.explain = s
            # read a sample from options.explain
            if options.explain:
                options.explain = [float(v.strip()) for v in options.explain.split(',')]

            if options.explain:
                print('\ninst: {}'.format(inst_id))
                if not xgb:
                    if options.uselime or options.useanchor or options.useshap:
                        xgb = XGBooster(options, from_model=options.files[0])
                    else:
                        # abduction-based approach requires an encoding
                        xgb = XGBooster(options, from_encoding=options.files[0])

                # checking LIME or SHAP should use all features
                if not options.limefeats:
                    options.limefeats = len(data.names) - 1

                # explain using anchor or the abduction-based approach
                expl = xgb.explain(options.explain, inst_id)

                if (options.uselime or options.useanchor or options.useshap) and options.validate:
                    xgb.validate(options.explain, expl)

        if options.compile:
            rules = xgb.compile()

        if options.process:
            xgb.process()

    exit()
