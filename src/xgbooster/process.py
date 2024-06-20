#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## process.py
##
##

#
#==============================================================================
from __future__ import print_function
import collections
from functools import reduce
import numpy as np
import os
from pysat.solvers import Solver
import resource
from six.moves import range
import sys


#
#==============================================================================
class DataProcessor(object):
    """
        SAT-based translator of a dataset from the original language of the
        dataset to the language of the boosted tree.
    """

    def __init__(self, formula, intvs, imaps, ivars, feats, nof_classes,
            options, xgb):
        """
            Constructor.
        """

        self.feats = feats
        self.intvs = intvs
        self.imaps = imaps
        self.ivars = ivars
        self.nofcl = nof_classes
        self.optns = options

        # in case we want to compute explanations wrt. internal BT literals
        self.lvars = xgb.mxe.lvars

        # saving XGBooster
        self.xgb = xgb

        self.verbose = self.optns.verb

        # SAT-based predictor
        self.oracle = Solver(name=self.optns.solver)
        for clid in range(nof_classes):
            self.oracle.append_formula(formula[clid].formula)

        # mapping from variable to feature id
        self.vid2fid = {}
        for feat in self.lvars:
            for v, ub in zip(self.lvars[feat], self.intvs[feat]):
                self.vid2fid[v] = (feat, ub)

        # variable to original feature index in the sample
        self.v2feat = {}
        for var in self.vid2fid:
            feat, ub = self.vid2fid[var]
            self.v2feat[var] = int(feat.split('_')[0][1:])

        self.names = {}
        self.preamble = []
        self.fvars = []
        for feat, intvs in self.intvs.items():
            # determining the right interval and the corresponding variable
            for i, (ub, fvar) in enumerate(zip(self.intvs[feat][:-1], self.lvars[feat][:-1])):
                name = self.xgb.feature_names[self.v2feat[fvar]]
                self.names[+fvar] = '{0} < {1}'.format(name, ub)
                self.names[-fvar] = '{0} >= {1}'.format(name, ub)

                self.preamble.append('{0}<{1}'.format(name, ub))
                self.fvars.append(fvar)

            # # if the feature is not binary, we need to record the last interval too
            # if len(self.intvs[feat]) > 2:
            #     self.preamble.append('{0} >= {1}'.format(name, ub))
            #     self.fvars.append(-fvar)

        self.preamble = ','.join(self.preamble)

    def __del__(self):
        """
            Destructor.
        """

        self.delete()

    def delete(self):
        """
            Actual destructor.
        """

        # deleting the SAT-based predictor
        if self.oracle:
            self.oracle.delete()
            self.oracle = None

    def execute(self, sample):
        """
            Extract a valid instance.
        """

        # translating sample into assumption literals
        hypos = self.xgb.mxe.get_literals(sample)

        self.oracle.solve(assumptions=hypos)
        assert self.oracle.get_status() == True, 'The previous call returned UNSAT!'
        model = self.oracle.get_model()

        newline = []

        for var in self.fvars:
            newline.append('1' if model[abs(var) - 1] == var else '0')

        return ','.join(newline)

    def process(self):
        """
            Do the compilation.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # going over both datasets (if any)
        for i, filename in enumerate(self.optns.process):
            with open(filename, 'r') as fp:
                samples = fp.readlines()
                samples.pop(0)

            with open('{0}_proc.csv'.format(os.path.splitext(filename)[0]), 'w') as fp:
                # dumping the header line
                print('{0},class'.format(self.preamble), file=fp)

                # traversing the samples
                for sample in samples:
                    samp = sample.strip().split(',')
                    orig = [float(v.strip()) for v in samp[:-1]]
                    proc = self.execute(np.array(orig))
                    print('{0},{1}'.format(proc, samp[-1]), file=fp)

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        if self.verbose:
            print('  rtime: {0:.2f}'.format(self.time))
