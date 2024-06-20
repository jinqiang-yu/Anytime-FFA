#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## explain.py
##
##

#
#==============================================================================
from __future__ import print_function
import collections
from functools import reduce
import numpy as np
import os
from .mxreason import MXReasoner, ClassEnc
from pysat.examples.hitman import Hitman
from pysat.formula import IDPool
from pysat.solvers import Solver as SATSolver
#from pysmt.shortcuts import Solver
#from pysmt.shortcuts import And, BOOL, Implies, Not, Or, Symbol
#from pysmt.shortcuts import Equals, GT, Int, INT, Real, REAL
import resource
from six.moves import range
import sys
import json
import torch
import matplotlib.pyplot as plt
import random
import lzma
import pickle
import six
from pysat.card import CardEnc
import statistics

#
#==============================================================================
class SMTExplainer(object):
    """
        An SMT-inspired minimal explanation extractor for XGBoost models.
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
        self.idmgr = IDPool()

        # saving XGBooster
        self.xgb = xgb

        self.verbose = self.optns.verb
        self.oracle = Solver(name=options.solver)

        self.inps = []  # input (feature value) variables
        for f in self.xgb.extended_feature_names_as_array_strings:
            if '_' not in f:
                self.inps.append(Symbol(f, typename=REAL))
            else:
                self.inps.append(Symbol(f, typename=BOOL))

        self.outs = []  # output (class  score) variables
        for c in range(self.nofcl):
            self.outs.append(Symbol('class{0}_score'.format(c), typename=REAL))

        # theory
        self.oracle.add_assertion(formula)

        # current selector
        self.selv = None

    def prepare(self, sample):
        """
            Prepare the oracle for computing an explanation.
        """

        if self.selv:
            # disable the previous assumption if any
            self.oracle.add_assertion(Not(self.selv))

        # creating a fresh selector for a new sample
        sname = ','.join([str(v).strip() for v in sample])

        # the samples should not repeat; otherwise, they will be
        # inconsistent with the previously introduced selectors
        assert sname not in self.idmgr.obj2id, 'this sample has been considered before (sample {0})'.format(self.idmgr.id(sname))
        self.selv = Symbol('sample{0}_selv'.format(self.idmgr.id(sname)), typename=BOOL)

        self.rhypos = []  # relaxed hypotheses

        # transformed sample
        self.sample = list(self.xgb.transform(sample)[0])

        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids

        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, self.sample), 1):
            feat = inp.symbol_name().split('_')[0]
            selv = Symbol('selv_{0}'.format(feat))
            val = float(val)

            self.rhypos.append(selv)
            if selv not in self.sel2fid:
                self.sel2fid[selv] = int(feat[1:])
                self.sel2vid[selv] = [i - 1]
            else:
                self.sel2vid[selv].append(i - 1)

        # adding relaxed hypotheses to the oracle
        if not self.intvs:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                if '_' not in inp.symbol_name():
                    hypo = Implies(self.selv, Implies(sel, Equals(inp, Real(float(val)))))
                else:
                    hypo = Implies(self.selv, Implies(sel, inp if val else Not(inp)))

                self.oracle.add_assertion(hypo)
        else:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                inp = inp.symbol_name()
                # determining the right interval and the corresponding variable
                for ub, fvar in zip(self.intvs[inp], self.ivars[inp]):
                    if ub == '+' or val < ub:
                        hypo = Implies(self.selv, Implies(sel, fvar))
                        break

                self.oracle.add_assertion(hypo)

        # in case of categorical data, there are selector duplicates
        # and we need to remove them
        self.rhypos = sorted(set(self.rhypos), key=lambda x: int(x.symbol_name()[6:]))

        # propagating the true observation
        if self.oracle.solve([self.selv] + self.rhypos):
            model = self.oracle.get_model()
        else:
            assert 0, 'Formula is unsatisfiable under given assumptions'

        # choosing the maximum
        outvals = [float(model.get_py_value(o)) for o in self.outs]
        maxoval = max(zip(outvals, range(len(outvals))))

        # correct class id (corresponds to the maximum computed)
        self.out_id = maxoval[1]
        self.output = self.xgb.target_name[self.out_id]

        # forcing a misclassification, i.e. a wrong observation
        disj = []
        for i in range(len(self.outs)):
            if i != self.out_id:
                disj.append(GT(self.outs[i], self.outs[self.out_id]))
        self.oracle.add_assertion(Implies(self.selv, Or(disj)))

        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.xgb.feature_names, inpvals):
                if f not in str(v):
                    self.preamble.append('{0} = {1}'.format(f, v))
                else:
                    self.preamble.append(str(v))

            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def explain(self, sample, smallest, expl_ext=None, prefer_ext=False):
        """
            Hypotheses minimization.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        # saving external explanation to be minimized further
        if expl_ext == None or prefer_ext:
            self.to_consider = [True for h in self.rhypos]
        else:
            eexpl = set(expl_ext)
            self.to_consider = [True if i in eexpl else False for i, h in enumerate(self.rhypos)]

        # if satisfiable, then the observation is not implied by the hypotheses
        if self.oracle.solve([self.selv] + [h for h, c in zip(self.rhypos, self.to_consider) if c]):
            print('  no implication!')
            print(self.oracle.get_model())
            sys.exit(1)

        if not smallest:
            self.compute_minimal(prefer_ext=prefer_ext)
        else:
            self.compute_smallest()

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        expl = sorted([self.sel2fid[h] for h in self.rhypos])

        if self.verbose:
            self.preamble = [self.preamble[i] for i in expl]
            print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.xgb.target_name[self.out_id]))
            print('  # hypos left:', len(self.rhypos))
            print('  time: {0:.2f}'.format(self.time))

        return expl

    def compute_minimal(self, prefer_ext=False):
        """
            Compute any subset-minimal explanation.
        """

        i = 0

        if not prefer_ext:
            # here, we want to reduce external explanation

            # filtering out unnecessary features if external explanation is given
            self.rhypos = [h for h, c in zip(self.rhypos, self.to_consider) if c]
        else:
            # here, we want to compute an explanation that is preferred
            # to be similar to the given external one
            # for that, we try to postpone removing features that are
            # in the external explanation provided

            rhypos  = [h for h, c in zip(self.rhypos, self.to_consider) if not c]
            rhypos += [h for h, c in zip(self.rhypos, self.to_consider) if c]
            self.rhypos = rhypos

        # simple deletion-based linear search
        while i < len(self.rhypos):
            to_test = self.rhypos[:i] + self.rhypos[(i + 1):]

            if self.oracle.solve([self.selv] + to_test):
                i += 1
            else:
                self.rhypos = to_test

    def compute_smallest(self):
        """
            Compute a cardinality-minimal explanation.
        """

        # result
        rhypos = []

        with Hitman(bootstrap_with=[[i for i in range(len(self.rhypos)) if self.to_consider[i]]]) as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(self.rhypos):
                if self.to_consider[i] == False:
                    continue

                if self.oracle.solve([self.selv] + self.rhypos[:i] + self.rhypos[(i + 1):]):
                    hitman.hit([i])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 1:
                    print('iter:', iters)
                    print('cand:', hset)

                if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset]):
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(range(len(self.rhypos))).difference(set(hset)))

                    model = self.oracle.get_model()
                    for h in removed:
                        i = self.sel2fid[self.rhypos[h]]
                        if '_' not in self.inps[i].symbol_name():
                            # feature variable and its expected value
                            var, exp = self.inps[i], self.sample[i]

                            # true value
                            true_val = float(model.get_py_value(var))

                            if not exp - 0.001 <= true_val <= exp + 0.001:
                                unsatisfied.append(h)
                            else:
                                hset.append(h)
                        else:
                            for vid in self.sel2vid[self.rhypos[h]]:
                                var, exp = self.inps[vid], int(self.sample[vid])

                                # true value
                                true_val = int(model.get_py_value(var))

                                if exp != true_val:
                                    unsatisfied.append(h)
                                    break
                            else:
                                hset.append(h)

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset] + [self.rhypos[h]]):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.verbose > 1:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    self.rhypos = [self.rhypos[i] for i in hset]
                    break


#
#==============================================================================
class MXExplainer(object):
    """
        An SMT-inspired minimal explanation extractor for XGBoost models.
    """

    def __init__(self, formula, intvs, imaps, ivars, feats, nof_classes,
            options, xgb):
        """
            Constructor.
        """
        random.seed(1234)

        self.feats = feats
        self.intvs = intvs
        self.imaps = imaps
        self.ivars = ivars
        self.nofcl = nof_classes
        self.optns = options
        self.idmgr = IDPool()
        self.fcats = []
        # saving XGBooster
        self.xgb = xgb

        self.verbose = self.optns.verb

        # MaxSAT-based oracles
        self.oracles = {}
        if self.optns.encode == 'mxa':
            ortype = 'alien'
        elif self.optns.encode == 'mxe':
            ortype = 'ext'
        else:
            ortype = 'int'

        #if self.optns.isimg and self.optns.xtype not in ('abductive', 'abd') \
        #    and (not (self.optns.smallest or self.optns.use_mhs)):
        #    self.formula = formula


        if self.optns.xtype not in ('abductive', 'abd') \
                and (not (self.optns.smallest or self.optns.use_mhs)):
            self.formula = formula
        else:
            self.formula = None
        #print('formula:', formula)

        if self.optns.knowledge:
            self.prepare_knowledge()
            #if lines:
            #    atmost1 = []
            #    # enforce exactly one of the feature values to be chosen
            #    # (for categorical features)
            #    categories = collections.defaultdict(lambda: [])
            #    expected = collections.defaultdict(lambda: 0)
            #    for f in self.xgb.extended_feature_names_as_array_strings:
            #        if '_' in f:
            #            if f in self.ivars:
            #                categories[f.split('_')[0]].append(self.ivars[f][1])
            #            expected[f.split('_')[0]] += 1
            #    top = abs(max([var for vars in self.ivars.values() for var in vars],
            #              key=abs))
            #    idpool = IDPool(start_from=top+1)
            #    print(self.ivars)
            #    for c, feats in six.iteritems(categories):
            #        if len(feats) > 1:
            #            if len(feats) == expected[c]:
            #                atmost1.extend(CardEnc.equals(feats,
            #                                              vpool=idpool, encoding=self.optns.cardenc))
            #            else:
            #                atmost1.extend(CardEnc.atmost(feats,
            #                                              vpool=idpool, encoding=self.optns.cardenc))

            #    for line in lines:
            #        sample = np.array([float(v.strip()) for v in line.split(',')])
            #        # translating sample into assumption literals
            #        self.hypos, self.hypo2fid = self.xgb.mxe.get_literals(sample)
            #        self.rm_inconsist_bg(atmost1=atmost1, curtop=idpool.top)

            if self.formula:
                self.bg_oracle = SATSolver(name=self.optns.solver,
                                           bootstrap_with=self.knowledge)
        else:
            self.knowledge = []

        for clid in range(nof_classes):
            self.oracles[clid] = MXReasoner(formula, clid,
                    solver=self.optns.solver,
                    oracle=ortype,
                    am1=self.optns.am1, exhaust=self.optns.exhaust,
                    minz=self.optns.minz, trim=self.optns.trim,
                    knowledge=self.knowledge)

        # a reference to the current oracle
        self.oracle = None

        # SAT-based predictor
        self.poracle = SATSolver(name='g3')
        for clid in range(nof_classes):
            self.poracle.append_formula(formula[clid].formula)

        # determining which features should go hand in hand
        categories = collections.defaultdict(lambda: [])
        for f in self.xgb.extended_feature_names_as_array_strings:
            # print(f, self.ivars)
            if f in self.ivars:
                if '_' in f or len(self.ivars[f]) == 2:
                    categories[f.split('_')[0]].append(self.xgb.mxe.vpos[self.ivars[f][0]])
                else:
                    for v in self.ivars[f]:
                        # this has to be checked and updated
                        categories[f].append(self.xgb.mxe.vpos[abs(v)])

        # these are the result indices of features going together
        self.fcats = [[min(ftups), max(ftups)] for ftups in categories.values()]
        self.fcats_copy = self.fcats[:]

        # all used feature categories
        self.allcats = list(range(len(self.fcats)))

        # variable to original feature index in the sample
        self.v2feat = {}
        for var in self.xgb.mxe.vid2fid:
            feat, ub = self.xgb.mxe.vid2fid[var]
            self.v2feat[var] = int(feat.split('_')[0][1:])

        # number of oracle calls involved
        self.calls = 0

    def __del__(self):
        """
            Destructor.
        """

        self.delete()

    def delete(self):
        """
            Actual destructor.
        """

        # deleting MaxSAT-based reasoners
        if self.oracles:
            for clid, oracle in self.oracles.items():
                if oracle:
                    oracle.delete()
            self.oracles = {}
        self.oracle = None

        # deleting the SAT-based predictor
        if self.poracle:
            self.poracle.delete()
            self.poracle = None

    def predict(self, sample):
        """
            Run the encoding and determine the corresponding class.
        """
        # translating sample into assumption literals
        self.hypos, self.hypo2fid = self.xgb.mxe.get_literals(sample)

        # variable to the category in use; this differs from
        # v2feat as here we may not have all the features here
        self.v2cat = {}
        for i, cat in enumerate(self.fcats):
            for v in range(cat[0], cat[1] + 1):
                self.v2cat[self.hypos[v]] = i

        # running the solver to propagate the prediction;
        # using solve() instead of propagate() to be able to extract a model
        assert self.poracle.solve(assumptions=self.hypos), 'Formula must be satisfiable!'
        model = self.poracle.get_model()
        # computing all the class scores
        scores = {}
        for clid in range(self.nofcl):
            # computing the value for the current class label
            scores[clid] = 0

            for lit, wght in self.xgb.mxe.enc[clid].leaves:
                if model[abs(lit) - 1] > 0:
                    scores[clid] += wght

        # returning the class corresponding to the max score
        return max(list(scores.items()), key=lambda t: t[1])[0]

    def prepare(self, sample):
        """
            Prepare the oracle for computing an explanation.
        """
        # first, we need to determine the prediction, according to the model
        self.out_id = self.predict(sample)

        # selecting the right oracle
        self.oracle = self.oracles[self.out_id]

        # transformed sample
        self.sample = list(self.xgb.transform(sample)[0])

        # correct class id (corresponds to the maximum computed)
        self.output = self.xgb.target_name[self.out_id]

        inpvals = self.xgb.readable_sample(sample)

        self.preamble = []
        for f, v in zip(self.xgb.feature_names, inpvals):
            if f not in str(v):
                self.preamble.append('{0} == {1}'.format(f, v))
            else:
                self.preamble.append(str(v))

        if self.verbose:
            if self.optns.isimg:
                if self.optns.use_categorical:
                    self.sample_ = []
                    for i, v in enumerate(sample):
                        if i in self.xgb.categorical_features:
                            fvs = (self.xgb.encoder[i].categories_[0])
                            try:
                                real_v = fvs[int(v)]
                            except:
                                assert len(fvs) == 1 and v > 0
                                real_v = (fvs[0] + 1) % 2
                            p = i + 1 if real_v > 0 else -(i+1)
                            self.sample_.append(p)
                        else:
                            self.sample_.append(v)
                    assert len(self.sample_) == self.optns.shape[0] * self.optns.shape[1]
                else:
                    pass
                print('  explaining:  {}'.format(self.preamble))
            else:
                print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def get_dist(self, pixel0, pixel1=None, centre=True):
        """
                Get Euclidean distance between two pixels in an image
        """
        pixel0_ = self.hypo2fid[pixel0]
        # todo consider nof channels
        x0 = (abs(pixel0_) - 1) // self.optns.shape[0]
        y0 = (abs(pixel0_) - 1) % self.optns.shape[0]
        if centre:
            x1 = (self.optns.shape[0] + 1) / 2 - 1
            y1 = (self.optns.shape[0] + 1) / 2 - 1
        elif pixel1:
            x1 = (abs(pixel1) - 1) // self.optns.shape[0]
            y1 = (abs(pixel1) - 1) % self.optns.shape[0]
        return (x0 - x1) ** 2 + (y0 - y1) ** 2

    def sort(self, input, convert=None, reverse=False):
        if self.optns.sort in ('centre', 'center', 'cent'):
            if convert is None:
                output = sorted(input,
                                key=lambda l: self.get_dist(abs(l), centre=True),
                                reverse=reverse)
            else:
                output = sorted(input,
                                key=lambda l: self.get_dist(abs(convert([l])[0]), centre=True),
                                reverse=reverse)
        elif self.optns.sort == 'random':
            output = random.sample(input, len(input))
        else:
            # sort by absolute values
            output = sorted(input, key=lambda l: abs(l),
                            reverse=reverse)
        return output

    def explain(self, sample, smallest, inst_id, expl_ext=None, prefer_ext=False):
        """
            Hypotheses minimization.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime
        self.times = []

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        if self.optns.encode != 'mxe':
            #if (self.optns.xtype in ('abductive', 'abd')) or \
            #    (smallest or self.optns.use_mhs):
            # dummy call with the full instance to detect all the necessary cores
            self.oracle.get_coex(self.hypos, full_instance=True, early_stop=True)

        # calling the actual explanation procedure
        self._explain(sample, inst_id=inst_id, smallest=smallest, xtype=self.optns.xtype,
                xnum=self.optns.xnum, unit_mcs=self.optns.unit_mcs,
                reduce_=self.optns.reduce)

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        if self.verbose:
            print('  prediciton:', self.xgb.target_name[self.out_id])
            nof_trees, depth = self.optns.files[0].split('_maxdepth_')
            nof_trees = int(nof_trees.rsplit('_', maxsplit=1)[-1])
            depth = int(depth.split('_', maxsplit=1)[0])

            for i, expl in enumerate(self.expls):
                hyps = list(reduce(lambda x, y: x + self.hypos[y[0]:y[1]+1], [self.fcats[c] for c in expl], []))
                expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))

                if self.optns.use_categorical:
                    expl_ = [self.sample_[abs(i)] for i in expl]
                else:
                    expl_ = [ii + 1 for ii in expl]

                print('  explanation:', expl_)
                print('  explanation size:', len(expl_))

                #if self.optns.isimg:
                #    if self.optns.use_categorical:
                #        expl_ = [self.sample_[abs(i)] for i in expl]
                #    else:
                #        expl_ = [ii+1 for ii in expl]

                #    print('  explanation:', expl_)
                #    print('  explanation size:', len(expl_))

                #else:
                #    preamble = [self.preamble[i] for i in expl]
                #    label = self.xgb.target_name[self.out_id]

                #    if self.optns.xtype in ('contrastive', 'con'):
                #        preamble = [l.replace('==', '!=') for l in preamble]
                #        label = 'NOT {0}'.format(label)

                #    print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(preamble), label))
                #    print('  explanation size:', len(expl))

                if len(self.times) == len(self.expls):
                    print('  expl time: {0:.2f}'.format(self.times[i]))

            if self.optns.xnum != 1 and self.optns.sort != 'random':

                for i, expl in enumerate(self.duals):
                    hyps = list(reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[c] for c in expl], []))
                    expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))

                    if self.optns.use_categorical:
                        expl_ = [self.sample_[abs(i)] for i in expl]
                    else:
                        expl_ = [ii + 1 for ii in expl]

                    print('  dual explanation:', expl_)
                    print('  dual explanation size:', len(expl_))

                    #if self.optns.isimg:
                    #    if self.optns.use_categorical:
                    #        expl_ = [self.sample_[abs(i)] for i in expl]
                    #    else:
                    #        expl_ = [ii + 1 for ii in expl]

                    #    print('  dual explanation:', expl_)
                    #    print('  dual explanation size:', len(expl_))
                    #else:
                    #    preamble = [self.preamble[i] for i in expl]
                    #    label = self.xgb.target_name[self.out_id]

                    #    if self.optns.xtype not in ('contrastive', 'con'):
                    #        preamble = [l.replace('==', '!=') for l in preamble]
                    #        label = 'NOT {0}'.format(label)

                    #    print('  dual explanation: "IF {0} THEN {1}"'.format(' AND '.join(preamble), label))
                    #    print('  dual explanation size:', len(expl))

            print('  calls:', self.calls)
            print('  rtime: {0:.2f}'.format(self.time))

        return self.expls

    def _explain(self, sample, inst_id=0, smallest=True, xtype='abd', xnum=1,
            unit_mcs=False, reduce_='none'):
        """
            Compute an explanation.
        """
        if reduce_ in ('mxp', 'map'):
            self.map_solver = SATSolver(name=self.optns.solver) #bootstrap_with=)
            self.map = []

        if self.optns.switch != 0:
            self.mhs_enumeration_switch(xnum=xnum, smallest=smallest, start_from=xtype)
        else:
            if xtype in ('abductive', 'abd'):
                # abductive explanations => MUS computation and enumeration
                if not smallest and xnum == 1:
                    self.expls = [self.extract_mus(reduce_=reduce_)]
                elif self.optns.sort == 'random' and xnum != 1:
                    self.extract_mus_enumeration(xnum=xnum, reduce_=reduce_)
                else:
                    self.mhs_mus_enumeration(xnum, smallest=smallest)
            else:  # contrastive explanations => MCS enumeration
                if smallest or self.optns.use_mhs:
                    self.mhs_mcs_enumeration(xnum, smallest, reduce_, unit_mcs)
                #elif self.optns.isimg:
                else:
                    self.cld_enumerate_mnist(sample, inst_id, xnum, unit_mcs, self.optns.use_cld)
                #else:
                #    self.cld_enumerate(sample, inst_id, xnum, unit_mcs, self.optns.use_cld)

        if 'mxp' in reduce_:
            del self.map_solver
            del self.map

    def extract_mus_enumeration(self, xnum, reduce_='lin'):
        self.expls = set()
        calls = 0
        # for i in ange(xnum):
        fail_attempts = 0
        id2xp = {}
        while True:
            self.calls = 0
            expl = self.extract_mus(reduce_=reduce_)
            calls += self.calls
            expl = frozenset(expl)
            len_expls0 = len(self.expls)
            self.expls.add(expl)
            len_expls1 = len(self.expls)
            if len_expls0 == len_expls1:
                fail_attempts += 1
                if fail_attempts >= 5:
                    print('fail enumerating {0} expls'.format(xnum))
                    break
            else:
                id2xp[len_expls1] = list(expl)
                if len_expls1 == xnum:
                    break

        self.calls = calls
        print('fail_attempts:', fail_attempts)

        self.expls = [id2xp[i] for i in sorted(id2xp.keys())]

    def extract_mus(self, reduce_='lin', start_from=None):
        """
            Compute one abductive explanation.
        """

        def _do_linear(core):
            """
                Do linear search.
            """

            def _assump_needed(a):
                if len(to_test) > 1:
                    to_test.remove(a)
                    self.calls += 1
                    # actual binary hypotheses to test
                    # print('testing', to_test, self._cats2hypos(to_test), core)
                    # print(self.v2cat)

                    # self._cats2hypos
                    # for feature indice to one hot encoding variables
                    # e.g. 1 is the 2nd feature of the instance
                    # it is possible to be expanded to more than 2 variables
                    if not self.oracle.get_coex(self._cats2hypos(to_test), early_stop=True):
                        # print('cost', self.oracle.oracles[1].cost)
                        return False
                    # print('cost', self.oracle.oracles[1].cost)
                    to_test.add(a)
                    return True
                else:
                    return True

            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))

        def _do_linear_(core):
            """
                Do linear search.
            """

            def _assump_needed(a):
                if len(to_test) > 1:
                    to_test.remove(a)
                    self.calls += 1
                    if not self.oracle.get_coex(to_test, early_stop=True):
                        # print('cost', self.oracle.oracles[1].cost)
                        return False
                    # print('cost', self.oracle.oracles[1].cost)
                    to_test.add(a)
                    return True
                else:
                    return True

            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))

        def _do_quickxplain(core):
            """
                Do QuickXplain-like search.
            """

            wset = core[:]
            filt_sz = len(wset) / 2.0
            while filt_sz >= 1:
                i = 0
                while i < len(wset):
                    to_test = wset[:i] + wset[(i + int(filt_sz)):]
                    # actual binary hypotheses to test
                    self.calls += 1
                    if to_test and not self.oracle.get_coex(self._cats2hypos(to_test), early_stop=True):
                        # assumps are not needed
                        wset = to_test
                    else:
                        # assumps are needed => check the next chunk
                        i += int(filt_sz)
                # decreasing size of the set to filter
                filt_sz /= 2.0
                if filt_sz > len(wset) / 2.0:
                    # next size is too large => make it smaller
                    filt_sz = len(wset) / 2.0
            return wset

        def _do_quickxplain_(core):
            """
                Do QuickXplain-like search.
            """

            wset = core[:]
            filt_sz = len(wset) / 2.0
            while filt_sz >= 1:
                i = 0
                while i < len(wset):
                    to_test = wset[:i] + wset[(i + int(filt_sz)):]
                    # actual binary hypotheses to test
                    self.calls += 1
                    if to_test and not self.oracle.get_coex(to_test, early_stop=True):
                        # assumps are not needed
                        wset = to_test
                    else:
                        # assumps are needed => check the next chunk
                        i += int(filt_sz)
                # decreasing size of the set to filter
                filt_sz /= 2.0
                if filt_sz > len(wset) / 2.0:
                    # next size is too large => make it smaller
                    filt_sz = len(wset) / 2.0
            return wset

        def _do_mapexplain(b, p, cand):
            # b: base; p: newly added to b or not;
            # cand: candidate,
            # all start from 1
            if len(p) != 0:
                #comp_b, b start from 1; self.allcats start from 0
                comp_b = set([c+1 for c in self.allcats]).difference(b)
                assump = b + [-c for c in comp_b]
                #print('assump:', assump)
                #print('len(assump):', len(assump))
                #print('b:', b)
                if self.map_solver.solve(assumptions=assump):
                    if self.oracle.get_coex(self._cats2hypos([c-1 for c in b])):
                        self.map.append(comp_b)
                        self.map_solver.add_clause(comp_b)
                    else:
                        return b

            if len(cand) == 1:
                return b + cand

            bootstrap_with = []
            #print('self.map:', self.map)
            if len(self.map) > 0:
                bootstrap_with.extend(self.map)
            #print('[b]:', [b])
            for bb in b:
                bootstrap_with.append([bb])
            other = set([c+1 for c in self.allcats]).difference(b).difference(cand)
            #print('[other]:', [other])
            for o in other:
                bootstrap_with.append([-o])
            #print('[cand]:', [cand])
            if cand:
                bootstrap_with.append(cand)
                bootstrap_with.append([-c for c in cand])
            #print('bootstrap_with:', bootstrap_with)
            #print()
            ssolver = SATSolver(name=self.optns.solver,
                                bootstrap_with=bootstrap_with)
            #                    bootstrap_with=self.map + [b] +
            #                    list(set([c+1 for c in self.allcats]).difference(b).difference(cand)) +
            #                    [cand] + [-c for c in cand])
            st = ssolver.solve()
            #print('st:', st)
            #print('b+cand:', b+cand)
            #print('len(b+cand):', len(b+cand))

            if not st:
                return b+cand
            # cand start from 1
            # start from 1
            m = ssolver.get_model()
            del ssolver
            # cand1 start from 1; m start from 1
            #print('cand:', cand)
            cand_set = set(cand)
            #print('m:', m)
            #print()
            cand1 = cand_set.intersection(m)
            #print('cand1:', cand1)
            #print('len(cand1):', len(cand1))
            # cand2 start from 1
            cand2 = cand_set.difference(cand1)
            #print('cand2:', cand2)
            #print('len(cand2):', len(cand2))
            # u1 start from 1
            u1 = _do_mapexplain(b+list(cand1), list(cand1), list(cand2))
            #print('u1:', u1)
            u1 = set(u1)
            #b_, cand1_, p2 start from 1
            b_, cand1_, p2 = u1.difference(cand1), u1.intersection(cand1), \
                             u1.intersection(cand2)
            u2 = _do_mapexplain(list(b_), list(p2), list(cand1_))
            #print('u2:', u2)
            return u2

        self.fcats = self.fcats_copy[:]

        # this is our MUS over-approximation
        if start_from is None:
            hypos = self.sort(self.hypos, reverse=self.optns.reverse)
            assert self.oracle.get_coex(list(reversed(hypos)),
                                        full_instance=True, early_stop=True) == None, 'No prediction'

            # getting the core
            core = self.oracle.get_reason(self.v2cat)
        else:
            core = start_from
        # sorting
        core = self.sort(core,
                         convert=self._cats2hypos,
                         reverse=self.optns.reverse)

        if self.verbose > 2:
            print('core:', core)

        self.calls = 1  # we have already made one call

        if reduce_ == 'qxp':
            expl = _do_quickxplain(core) if not self.optns.neg \
                else _do_quickxplain_(core)
        elif reduce_ in ('mxp', 'map'):
            expl = _do_mapexplain([], [], [c+1 for c in core])
            expl = sorted(map(lambda l: l-1, expl))
        else:  # by default, linear MUS extraction is used
            expl = _do_linear(core) if not self.optns.neg \
                else _do_linear_(core)
        return expl

    def extract_mcs(self, mss_cand, reduce_='lin', start_from=None):
        """
            Compute one contrastive explanation.
        """

        def _do_linear(unsatisfied, satisfied):
            """
                Do linear search.
            """
            expl = []
            for h in unsatisfied:
                self.calls += 1
                if self.oracle.get_coex(self._cats2hypos(satisfied + [h]), early_stop=True):
                    satisfied.append(h)
                else:
                    expl.append(h)
            return expl

        def _do_mapexplain(b, p, cand):
            # b: base; p: newly added to b or not;
            # cand: candidate,
            # all start from 1
            if len(p) != 0:
                #comp_b, b start from 1; self.allcats start from 0
                comp_b = list(set([c+1 for c in self.allcats]).difference(b))
                assump = [-c for c in b] + comp_b
                #print('assump:', assump)
                #print('len(assump):', len(assump))
                #print('b:', b)
                if self.map_solver.solve(assumptions=assump):
                    if self.oracle.get_coex(self._cats2hypos([c-1 for c in comp_b])):
                        return b
                    else:
                        self.map.append([-c for c in comp_b])
                        self.map_solver.add_clause([-c for c in comp_b])

            if len(cand) == 1:
                return b + cand

            bootstrap_with = []
            #print('self.map:', self.map)
            if len(self.map) > 0:
                bootstrap_with.extend(self.map)
            #print('[b]:', [b])
            for bb in b:
                bootstrap_with.append([-bb])
            other = set([c+1 for c in self.allcats]).difference(b).difference(cand)
            #print('[other]:', [other])
            for o in other:
                bootstrap_with.append([o])
            #print('[cand]:', [cand])
            if cand:
                bootstrap_with.append(cand)
                bootstrap_with.append([-c for c in cand])
            #print('bootstrap_with:', bootstrap_with)
            #print()
            ssolver = SATSolver(name=self.optns.solver,
                                bootstrap_with=bootstrap_with)
            #                    bootstrap_with=self.map + [b] +
            #                    list(set([c+1 for c in self.allcats]).difference(b).difference(cand)) +
            #                    [cand] + [-c for c in cand])
            st = ssolver.solve()
            #print('st:', st)
            #print('b+cand:', b+cand)
            #print('len(b+cand):', len(b+cand))

            if not st:
                return b+cand
            # cand start from 1
            # start from 1
            m = ssolver.get_model()
            del ssolver
            # cand1 start from 1; m start from 1
            #print('cand:', cand)
            cand_set = set(cand)
            #print('m:', m)
            #print()
            cand1 = cand_set.intersection([-c for c in m])
            #print('cand1:', cand1)
            #print('len(cand1):', len(cand1))
            # cand2 start from 1
            cand2 = cand_set.difference(cand1)
            #print('cand2:', cand2)
            #print('len(cand2):', len(cand2))
            # u1 start from 1
            u1 = _do_mapexplain(b+list(cand1), list(cand1), list(cand2))
            #print('u1:', u1)
            u1 = set(u1)
            #b_, cand1_, p2 start from 1
            b_, cand1_, p2 = u1.difference(cand1), u1.intersection(cand1), \
                             u1.intersection(cand2)
            u2 = _do_mapexplain(list(b_), list(p2), list(cand1_))
            #print('u2:', u2)
            return u2

        # this is our MCS over-approximation
        if start_from is None:
            coex = self.oracle.get_coex([], early_stop=True)
            assert coex

            # getting the un/satisifed features
            unsatisfied, satisfied = [], []
            for h in self.hypos:
                if coex[abs(h) - 1] != h:
                    unsatisfied.append(self.v2cat[h])
                else:
                    satisfied.append(self.v2cat[h])

            unsatisfied = list(set(unsatisfied))
            satisfied = list(set(satisfied))
        else:
            unsatisfied = start_from
            satisfied = mss_cand

        if self.optns.reduce in ('mxp', 'map'):
            expl = _do_mapexplain([], [], [c+1 for c in unsatisfied])
            expl = sorted(map(lambda l: l-1, expl))
        else:
            expl = _do_linear(unsatisfied, satisfied)

        return expl

    def mhs_mus_enumeration(self, xnum, smallest=False):
        """
            Enumerate subset- and cardinality-minimal explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (contrastive) explanations
        self.duals = []

        if smallest:
            htype = 'sorted'
        else:
            htype = self.optns.htype
            if htype == 'sat':
                if self.optns.hsolver not in ('mgh', 'cdl15'):
                    self.optns.hsolver = 'mgh'
        solver = self.optns.hsolver
        
        with Hitman(bootstrap_with=[self.allcats], solver=solver,
                    htype=htype) as hitman:
            # computing unit-size MCSes
            if self.optns.unit_mcs:
                for c in self.allcats:#
                    self.calls += 1
                    if self.oracle.get_coex(self._cats2hypos(self.allcats[:c] + self.allcats[(c + 1):]), early_stop=True):
                        hitman.hit([c])
                        self.duals.append([c])

                        if 'mxp' in self.optns.reduce:
                            self.map.append([c + 1])
                            self.map_solver.add_clause([c + 1])

                        if self.verbose > 2:
                            hyps = list(
                                reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[cc] for cc in [c]], []))
                            expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                            print('coex:', expl)
                            time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                   resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                            print('  coextime: {0:.2f}'.format(time))

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                hypos = self._cats2hypos(hset)
                coex = self.oracle.get_coex(hypos, early_stop=True)
                if coex:
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(self.hypos).difference(set(hypos)))

                    for h in removed:
                        if coex[abs(h) - 1] != h:
                            unsatisfied.append(self.v2cat[h])
                        else:
                            hset.append(self.v2cat[h])

                    unsatisfied = list(set(unsatisfied))
                    hset = list(set(hset))

                    # computing an MCS (expensive)
                    #for h in unsatisfied:
                    #    self.calls += 1
                    #    if self.oracle.get_coex(self._cats2hypos(hset + [h]), early_stop=True):
                    #        hset.append(h)
                    #    else:
                    #        to_hit.append(h)
                    to_hit = self.extract_mcs(hset, reduce_=self.optns.reduce, start_from=unsatisfied)
                    hitman.hit(to_hit)
                    self.duals.append(to_hit)

                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[c] for c in to_hit], []))
                        expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('coex:', expl)
                        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                        self.times.append(time)
                        print('  coextime: {0:.2f}'.format(time))
                    if 'mxp' in self.optns.reduce:
                        self.map.append([c + 1 for c in to_hit])
                        self.map_solver.add_clause(self.map[-1])
                else:
                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[c] for c in hset], []))
                        expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('expl:', expl)
                        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                        self.times.append(time)
                        print('  expltime: {0:.2f}'.format(time))
                    self.expls.append(hset)
                    if len(self.expls) != xnum:
                        hitman.block(hset)
                        if 'mxp' in self.optns.reduce:
                            self.map.append([-(c + 1) for c in hset])
                            self.map_solver.add_clause(self.map[-1])
                    else:
                        break

    def mhs_mcs_enumeration(self, xnum, smallest=False, reduce_='none', unit_mcs=False):
        """
            Enumerate subset- and cardinality-minimal contrastive explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (abductive) explanations
        self.duals = []

        if smallest:
            htype = 'sorted'
        else:
            htype = self.optns.htype
            if htype == 'sat':
                if self.optns.hsolver not in ('mgh', 'cdl15'):
                    self.optns.hsolver = 'mgh'
        solver = self.optns.hsolver

        with Hitman(bootstrap_with=[self.allcats], solver=solver,
                    htype=htype) as hitman:
            # computing unit-size MUSes
            for c in self.allcats:
                self.calls += 1

                if not self.oracle.get_coex(self._cats2hypos([c]), early_stop=True):
                    hitman.hit([c])
                    self.duals.append([c])

                    if 'mxp' in reduce_:
                        self.map.append([-(c+1)])
                        self.map_solver.add_clause([-(c+1)])

                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[cc] for cc in [c]], []))
                        expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('coex:', expl)
                        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                        self.times.append(time)
                        print('  coextime: {0:.2f}'.format(time))

                elif unit_mcs and self.oracle.get_coex(self._cats2hypos(self.allcats[:c] + self.allcats[(c + 1):]), early_stop=True):
                    # this is a unit-size MCS => block immediately
                    self.calls += 1
                    hitman.block([c])
                    self.expls.append([c])

                    if 'mxp' in reduce_:
                        self.map.append([c+1])
                        self.map_solver.add_clause([c+1])

                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[cc] for cc in [c]], []))
                        expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('expl:', expl)
                        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                        self.times.append(time)
                        print('  expltime: {0:.2f}'.format(time))

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if not self.oracle.get_coex(self._cats2hypos(set(self.allcats).difference(set(hset))), early_stop=True):
                    to_hit = self.oracle.get_reason(self.v2cat)

                    if len(to_hit) > 1 and reduce_ != 'none':
                        to_hit = self.extract_mus(reduce_=reduce_, start_from=to_hit)

                    hitman.hit(to_hit)
                    self.duals.append(to_hit)

                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[c] for c in to_hit], []))
                        expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('coex:', expl)
                        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                        print('  coextime: {0:.2f}'.format(time))

                    if 'mxp' in reduce_:
                        self.map.append([-(c+1) for c in to_hit])
                        self.map_solver.add_clause(self.map[-1])
                else:
                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[c] for c in hset], []))
                        expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('expl:', expl)
                        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                        self.times.append(time)
                        print('  expltime: {0:.2f}'.format(time))

                    self.expls.append(hset)

                    if len(self.expls) != xnum:
                        hitman.block(hset)
                        if 'mxp' in reduce_:
                            self.map.append([c+1 for c in hset])
                            self.map_solver.add_clause(self.map[-1])
                    else:
                        break

    def cld_enumerate(self, sample, inst_id, xnum, unit_mcs, use_cld):
        """
            Compute a subset-minimal contrastive explanation.
        """

        def _overapprox(model):
            #model = self.oracle.get_model()
            for i, c in enumerate(self.allcats):
                hypos = self._cats2hypos([c])
                for hypo in hypos:
                    if model[abs(hypo) - 1] != hypo:
                        #at least one of the hypos is falsified
                        self.setd.append(c)
                        break
                else:
                    #all hypos are satisfied
                    self.ss_assumps.append(c)

            #for hypo in self.hypos:
            #    if model[abs(hypo) - 1] == hypo:
            #        # soft clauses contain positive literals
            #        # so if var is true then the clause is satisfied
            #        self.ss_assumps.append(hypo)
            #    else:
            #        self.setd.append(hypo)

        def _compute():
            i = 0
            while i < len(self.setd):
                if use_cld:
                    _do_cld_check(self.setd[i:])
                    i = 0

                if self.setd:
                    # it may be empty after the clause D check

                    self.calls += 1
                    self.ss_assumps.append(self.setd[i])
                    if len(self.knowledge) > 0:
                        a = [self.svar] + self._cats2hypos(self.ss_assumps)
                    else:
                        a = [self.svar] + self._cats2hypos(self.ss_assumps) + \
                            [-h for h in self._cats2hypos(self.bb_assumps)]
                    if not self.oracle.get_coex(a):
                        self.ss_assumps.pop()
                        self.bb_assumps.append(self.setd[i])
                i += 1

        def _do_cld_check(cld_):
            self.cldid += 1
            sel = self.vpool.id('{0}_{1}'.format(self.svar, self.cldid))
            cld = self._cats2hypos(cld_) + [-sel]
            # adding clause D
            self.oracle.add_clause(cld)

            #self.ss_assumps.append(sel)

            self.setd = []

            if len(self.knowledge) > 0:
                a = [self.svar] + [sel] + \
                    self._cats2hypos(self.ss_assumps)
            else:
                a = [self.svar] + [sel] + \
                    self._cats2hypos(self.ss_assumps) + \
                [-h for h in self._cats2hypos(self.bb_assumps)]
            #self.ss_assumps.pop()  # remo
            model = self.oracle.get_coex(a)
            #self.ss_assumps.pop()  # removing clause D assumption
            if model:
                for l in cld_:
                    hypos = self._cats2hypos([l])
                    for hypo in hypos:
                        if model[abs(hypo) - 1] != hypo:
                            # at least one of the hypos is falsified
                            self.setd.append(l)
                            break
                    else:
                        # all hypos are satisfied
                        # filtering all satisfied literals
                        self.ss_assumps.append(l)
                    ## filtering all satisfied literals
                    #if model[abs(l) - 1] == l:
                    #    self.ss_assumps.append(l)
                    #else:
                    #    self.setd.append(l)
            else:
                # clause D is unsatisfiable => all literals are backbones
                self.bb_assumps.extend(cld_)
            # deactivating clause D
            self.oracle.add_clause([-sel])

        self.vpool = self.oracle.vpool
        # creating a new selector
        self.svar = self.vpool.id(tuple(sample + [inst_id]))

        # dummy call with the full instance to detect all the necessary cores
        self.oracle.get_coex([self.svar]+self.hypos, full_instance=True, early_stop=True)

        #self.oracle.add_clause([-svar, svar])
        # sets of selectors to work with
        self.cldid = 0
        self.expls = []
        self.duals = []

        # detect and block unit-size MCSes immediately
        if unit_mcs:
            for i, c in enumerate(self.allcats):
                if self.oracle.get_coex([self.svar] + self._cats2hypos(self.allcats[:i] + self.allcats[(i + 1):]),
                                        early_stop=True):
                    self.expls.append([c])
                    time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                    self.times.append(time)
                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[cc] for cc in [c]], []))
                        xp = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('expl:', xp)
                        print('  expltime: {0:.2f}'.format(time))
                    if len(self.expls) != xnum:
                        self.oracle.add_clause([-self.svar] + self._cats2hypos([self.allcats[c]]))
                    else:
                        break

        self.calls += 1
        model = self.oracle.get_coex([self.svar])
        while model:
            self.ss_assumps, self.bb_assumps, self.setd = [], [], []
            _overapprox(model)
            _compute()

            expl = [l for l in self.bb_assumps]
            self.expls.append(expl)  # here is a new CXp
            time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                   resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
            self.times.append(time)
            if self.verbose > 2:
                hyps = list(
                    reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[cc] for cc in expl], []))
                xp = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                print('expl:', xp)
                print('  expltime: {0:.2f}'.format(time))

            if len(self.expls) == xnum:
                break

            self.oracle.add_clause([-self.svar] + self._cats2hypos(expl))
            self.calls += 1
            time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                   resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
            self.times.append(time)

            model = self.oracle.get_coex([self.svar])

        self.calls += self.cldid

        # deleting all the MCSes computed for the instance
        self.oracle.add_clause([-self.svar])

    def cld_enumerate_mnist(self, sample, inst_id, xnum, unit_mcs, use_cld):
        """
            Compute a subset-minimal contrastive explanation.
        """

        def _overapprox(model):
            #model = self.oracle.get_model()
            for i, c in enumerate(self.allcats):
                hypos = self._cats2hypos([c])
                for hypo in hypos:
                    if model[abs(hypo) - 1] != hypo:
                        #at least one of the hypos is falsified
                        self.setd.append(c)
                        break
                else:
                    #all hypos are satisfied
                    self.ss_assumps.append(c)

            #for hypo in self.hypos:
            #    if model[abs(hypo) - 1] == hypo:
            #        # soft clauses contain positive literals
            #        # so if var is true then the clause is satisfied
            #        self.ss_assumps.append(hypo)
            #    else:
            #        self.setd.append(hypo)

        def _compute():
            i = 0
            while i < len(self.setd):
                #if use_cld:
                #    _do_cld_check(self.setd[i:])
                #    i = 0

                if self.setd:
                    # it may be empty after the clause D check
                    self.calls += 1
                    self.ss_assumps.append(self.setd[i])

                    ## filter out inconsistent hypos with knowledge
                    #a = self._cats2hypos(self.ss_assumps) + \
                    #    [-h for h in self._cats2hypos(self.bb_assumps)]
                    #if len(self.knowledge) > 0:
                    #    res = self.bg_oracle.solve(a)

                    #if len(self.knowledge) == 0 or res:
                    #    res = self.oracle.get_coex(a)

                    #if not res:
                    #    self.ss_assumps.pop()
                    #    self.bb_assumps.append(self.setd[i])

                    # NORMAL filter out inconsistent hypos
                    if len(self.knowledge) > 0:
                        a = self._cats2hypos(self.ss_assumps)
                    else:
                        a = self._cats2hypos(self.ss_assumps) + \
                            [-h for h in self._cats2hypos(self.bb_assumps)]
                    if not self.oracle.get_coex(a):
                        self.ss_assumps.pop()
                        self.bb_assumps.append(self.setd[i])

                    #if self.optns.isimg:
                    #    a = self._cats2hypos(self.ss_assumps) + \
                    #        [-h for h in self._cats2hypos(self.bb_assumps)]
                    #    if len(self.knowledge) > 0:
                    #        res = self.bg_oracle.solve(a)

                    #    if len(self.knowledge) == 0 or res:
                    #        res = self.oracle.get_coex(a)

                    #    if not res:
                    #        self.ss_assumps.pop()
                    #        self.bb_assumps.append(self.setd[i])
                    #else:
                    #    if len(self.knowledge) > 0:
                    #        a = self._cats2hypos(self.ss_assumps)
                    #    else:
                    #        a = self._cats2hypos(self.ss_assumps) + \
                    #            [-h for h in self._cats2hypos(self.bb_assumps)]
                    #    if not self.oracle.get_coex(a):
                    #        self.ss_assumps.pop()
                    #        self.bb_assumps.append(self.setd[i])
                i += 1

        #def _do_cld_check(cld_):
        #    self.cldid += 1
        #    sel = self.vpool.id('{0}_{1}'.format(self.svar, self.cldid))
        #    cld = self._cats2hypos(cld_) + [-sel]
        #    # adding clause D
        #    self.oracle.add_clause(cld)

        #    #self.ss_assumps.append(sel)

        #    self.setd = []

        #    if len(self.knowledge) > 0:
        #        a = [self.svar] + [sel] + \
        #            self._cats2hypos(self.ss_assumps)
        #    else:
        #        a = [self.svar] + [sel] + \
        #            self._cats2hypos(self.ss_assumps) + \
        #        [-h for h in self._cats2hypos(self.bb_assumps)]
        #    #self.ss_assumps.pop()  # remo
        #    print('self.vpool.top:', self.vpool.top)
        #    model = self.oracle.get_coex(a)
        #    #self.ss_assumps.pop()  # removing clause D assumption
        #    if model:
        #        for l in cld_:
        #            hypos = self._cats2hypos([l])
        #            for hypo in hypos:
        #                if model[abs(hypo) - 1] != hypo:
        #                    # at least one of the hypos is falsified
        #                    self.setd.append(l)
        #                    break
        #            else:
        #                # all hypos are satisfied
        #                # filtering all satisfied literals
        #                self.ss_assumps.append(l)
        #            ## filtering all satisfied literals
        #            #if model[abs(l) - 1] == l:
        #            #    self.ss_assumps.append(l)
        #            #else:
        #            #    self.setd.append(l)
        #    else:
        #        # clause D is unsatisfiable => all literals are backbones
        #        self.bb_assumps.extend(cld_)
        #    # deactivating clause D
        #    self.oracle.add_clause([-sel])

        #self.vpool = self.oracle.vpool
        # creating a new selector
        #self.svar = self.vpool.id(tuple(sample + [inst_id]))

        # dummy call with the full instance to detect all the necessary cores
        self.oracle.get_coex(self.hypos, full_instance=True, early_stop=True)

        #self.oracle.add_clause([-svar, svar])
        # sets of selectors to work with
        self.cldid = 0
        self.expls = []
        self.duals = []

        self.blk_oracle = MXReasoner(self.formula, self.out_id,
                                      solver=self.optns.solver,
                                      #oracle='ext',
                                      oracle=self.oracle.ortype,
                                      am1=self.optns.am1, exhaust=self.optns.exhaust,
                                      minz=self.optns.minz, trim=self.optns.trim,
                                      knowledge=self.knowledge)

        # dummy call with the full instance to detect all the necessary cores
        self.oracle.get_coex(self.hypos, full_instance=True, early_stop=True)

        # detect and block unit-size MCSes immediately
        if unit_mcs:
            for i, c in enumerate(self.allcats):
                if self.oracle.get_coex(self._cats2hypos(self.allcats[:i] + self.allcats[(i + 1):]),
                                        early_stop=True):
                    self.expls.append([c])
                    time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                    self.times.append(time)
                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[cc] for cc in [c]], []))
                        xp = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('expl:', xp)
                        print('  expltime: {0:.2f}'.format(time))
                    if len(self.expls) != xnum:
                        self.blk_oracle.add_clause(self._cats2hypos([self.allcats[c]]))
                    else:
                        break
        self.calls += 1
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime
        model = self.blk_oracle.get_coex([], full_instance=True, early_stop=True)
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
        print('  checktime: {0:.2f}'.format(time))
        while model:
            time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                   resource.getrusage(resource.RUSAGE_SELF).ru_utime
            self.ss_assumps, self.bb_assumps, self.setd = [], [], []
            _overapprox(model)
            _compute()

            expl = [l for l in self.bb_assumps]
            self.expls.append(expl)  # here is a new CXp
            self.blk_oracle.add_clause(self._cats2hypos(expl))

            if self.verbose > 2:
                hyps = list(
                    reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[cc] for cc in expl], []))
                xp = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                print('expl:', xp)

                time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
                print('  sexpltime: {0:.2f}'.format(time))

                time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                self.times.append(time)
                print('  expltime: {0:.2f}'.format(time))

            if len(self.expls) == xnum:
                break

            self.calls += 1
            time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                   resource.getrusage(resource.RUSAGE_SELF).ru_utime
            model = self.blk_oracle.get_coex([], early_stop=True)
            time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                   resource.getrusage(resource.RUSAGE_SELF).ru_utime - time
            print('  checktime: {0:.2f}'.format(time))

        self.calls += self.cldid

        # deleting all the MCSes computed for the instance
        #self.oracle.add_clause([-self.svar])

        self.blk_oracle.delete()
        self.blk_oracle = None

    def mhs_enumeration_switch(self, xnum, smallest=False, start_from='abd'):
        """
            Enumerate subset- and cardinality-minimal explanations
            with simple switch.
        """
        # result
        axps, cxps, uni_axps, uni_cxps = [], [], [], []
        self.uni_cxps = uni_cxps

        if smallest:
            htype = 'sorted'
        else:
            htype = self.optns.htype
            if htype == 'sat':
                if self.optns.hsolver not in ('mgh', 'cdl15'):
                    self.optns.hsolver = 'mgh'
        solver = self.optns.hsolver
        self.num_switch = 0

        # TODO if sort, we need to use 2 separate hitmen
        #hitman = Hitman(bootstrap_with=[self.allcats], solver=solver, htype=htype)
        hitmans = {xtype: Hitman(bootstrap_with=[self.allcats], solver=solver,
                                 htype=htype)
                   for xtype in range(2)}
        #self.hitman_ = Hitman(bootstrap_with=[self.allcats], solver=solver, htype=htype)
        self.allxps = []

        if self.optns.stype == 'simple':
            if self.optns.sliding == 1:
                def compare(dual_expls, target_expls, bias=0, xtype='axp'):
                    return (len(dual_expls) + bias) >= len(target_expls)
            else:
                def compare(dual_expls, target_expls, bias=0, xtype='axp'):
                    expls = self.allxps[-self.optns.sliding: ]
                    if xtype != 'axp':
                        expls = list(map(lambda l: (l + 1) % 2, expls))
                    len_target = sum(expls)
                    #if xtype == 'axp':
                    #    len_target = len(list(filter(lambda l: l > 0, expls)))
                    #else:
                    #    len_target = len(list(filter(lambda l: l < 0, expls)))
                    len_dual = len(expls) - len_target
                    return (len_dual + bias) >= len_target
        else:
            def compare_large_gap(dual_expls, target_expls, bias=0, xtype='axp'):
                #TODO when multiswitching or target is AXp
                if xtype == 'axp':
                    return True
                else:
                    # target is CXps, dual is AXps
                    duals = dual_expls[-self.optns.sliding:]
                    targets = target_expls[-self.optns.sliding:]
                    if len(self.uni_cxps) >= self.optns.sliding/2 and len(duals) == self.optns.sliding:
                        duals_size = [len(l) for l in duals]
                        avg_targets_size = 1
                        ratio = statistics.mean(duals_size) / avg_targets_size
                        return ratio < self.optns.gap

                    if len(duals) == self.optns.sliding and len(targets) == self.optns.sliding:
                        duals_size = [len(l) for l in duals]
                        targets_size = [len(l) for l in targets]
                        ratio = statistics.mean(duals_size) / statistics.mean(targets_size)
                        return ratio < self.optns.gap
                    return True

            def compare_stable(dual_expls, target_expls, bias=0, xtype='axp'):
                #TODO when multiswitching or target is AXp
                if xtype == 'axp':
                    return True
                else:
                    # target is CXps, dual is AXps
                    if len(target_expls) >= self.optns.sliding:
                        targets = target_expls[-self.optns.sliding:]
                        cur_tg_size = len(targets[-1])
                        for tg in targets[:-1]:
                            if abs(len(tg) - cur_tg_size) > self.optns.diff:
                                return True
                        return False
                    return True
            # from CXp to AXp
            if self.optns.stype == 'lgap':
                def compare(dual_expls, target_expls, bias=0, xtype='axp'):
                    if self.num_switch == 0:
                        return compare_large_gap(dual_expls, target_expls, bias, xtype)
                    return True
            elif self.optns.stype == 'stable':
                def compare(dual_expls, target_expls, bias=0, xtype='axp'):
                    if self.num_switch == 0:
                        return compare_stable(dual_expls, target_expls, bias, xtype)
                    return True
            elif self.optns.stype == 'ls':
                def compare(dual_expls, target_expls, bias=0, xtype='axp'):
                    if self.num_switch == 0:
                        return compare_large_gap(dual_expls, target_expls, bias, xtype) and \
                                compare_stable(dual_expls, target_expls, bias, xtype)
                    return True
            else:
                print('TODO: invalide stype')
                exit(1)
        def axp_enumeration(xnum, hitman, hitman_cxp, axp_done=False, cxp_done=False,
                            iters=0, num_switch=0):
            # computing unit-size MCSes
            if num_switch <= 0 and self.optns.unit_mcs:
                for c in self.allcats:
                    self.calls += 1
                    if self.oracle.get_coex(self._cats2hypos(self.allcats[:c] + self.allcats[(c + 1):]),
                                            early_stop=True):
                        hitman.hit([c])
                        hitman_cxp.block([c])
                        uni_cxps.append([c])
                        self.allxps.append(0)
                        #self.allxps.append(-1)

                        if self.verbose > 2:
                            hyps = list(
                                reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[cc] for cc in [c]], []))
                            expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                            print('cxp:', expl)
                            time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                   resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                            print('  cxptime: {0:.4f}'.format(time))

            while compare(cxps, axps, bias=0, xtype='axp'):
                hset = hitman.get()
                iters += 1

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    axp_done, cxp_done = True, True
                    break

                self.calls += 1
                hypos = self._cats2hypos(hset)
                coex = self.oracle.get_coex(hypos, early_stop=True)
                if coex:
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(self.hypos).difference(set(hypos)))

                    for h in removed:
                        if coex[abs(h) - 1] != h:
                            unsatisfied.append(self.v2cat[h])
                        else:
                            hset.append(self.v2cat[h])

                    unsatisfied = list(set(unsatisfied))
                    hset = list(set(hset))

                    # computing an MCS (expensive), i.e. CXp
                    for h in unsatisfied:
                        self.calls += 1
                        if self.oracle.get_coex(self._cats2hypos(hset + [h]), early_stop=True):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[c] for c in to_hit], []))
                        expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('cxp:', expl)
                        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                        self.times.append(time)
                        print('  cxptime: {0:.4f}'.format(time))
                    hitman.hit(to_hit)
                    hitman_cxp.block(to_hit)
                    cxps.append(to_hit)
                    self.allxps.append(0)
                    #self.allxps.append(-len(to_hit))
                else:
                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[c] for c in hset], []))
                        expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('axp:', expl)
                        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                        self.times.append(time)
                        print('  axptime: {0:.2f}'.format(time))
                    axps.append(hset)
                    self.allxps.append(1)
                    #self.allxps.append(len(hset))

                    if len(axps) != xnum:
                        hitman.block(hset)
                        hitman_cxp.hit(hset)
                    else:
                        axp_done = True
                        break

            return axp_done, cxp_done, iters

        def cxp_enumeration(xnum, hitman, hitman_axp, axp_done=False, cxp_done=False,
                            iters=0, num_switch=0, reduce_='lin'):
            if num_switch == 0:
                # computing unit-size MUSes
                for c in self.allcats:
                    self.calls += 1

                    if not self.oracle.get_coex(self._cats2hypos([c]), early_stop=True):
                        hitman.hit([c])
                        hitman_axp.block([c])
                        uni_axps.append([c])
                        self.allxps.append(1)
                        #self.allxps.append(1)
                    elif self.optns.unit_mcs and self.oracle.get_coex(self._cats2hypos(self.allcats[:c] + self.allcats[(c + 1):]),
                                                           early_stop=True):
                        # this is a unit-size MCS => block immediately
                        self.calls += 1
                        hitman.block([c])
                        hitman_axp.hit([c])
                        uni_cxps.append([c])
                        self.allxps.append(0)
                        #self.allxps.append(-1)

                        if self.verbose > 2:
                            hyps = list(
                                reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[cc] for cc in [c]], []))
                            expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                            print('cxp:', expl)
                            time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                   resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                            self.times.append(time)
                            print('  cxptime: {0:.2f}'.format(time))

            while compare(axps, cxps, 0, xtype='cxp'):
                hset = hitman.get()
                iters += 1

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    axp_done, cxp_done = True, True
                    break

                self.calls += 1
                if not self.oracle.get_coex(self._cats2hypos(set(self.allcats).difference(set(hset))), early_stop=True):
                    to_hit = self.oracle.get_reason(self.v2cat)

                    if len(to_hit) > 1 and reduce_ != 'none':
                        to_hit = self.extract_mus(reduce_=reduce_, start_from=to_hit)
                    axps.append(to_hit)
                    self.allxps.append(1)
                    #self.allxps.append(len(to_hit))

                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[c] for c in to_hit], []))
                        expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('axp:', expl)
                        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                        print('  axptime: {0:.2f}'.format(time))

                    hitman.hit(to_hit)
                    hitman_axp.block(to_hit)
                else:
                    if self.verbose > 2:
                        hyps = list(
                            reduce(lambda x, y: x + self.hypos[y[0]:y[1] + 1], [self.fcats[c] for c in hset], []))
                        expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                        print('cxp:', expl)
                        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
                        self.times.append(time)
                        print('  cxptime: {0:.2f}'.format(time))

                    cxps.append(hset)
                    self.allxps.append(0)
                    #self.allxps.append(-len(hset))

                    if len(cxps) != xnum:
                        hitman.block(hset)
                        hitman_axp.hit(hset)
                    else:
                        cxp_done = True
                        break
            return axp_done, cxp_done, iters

        cur_xtype = 1 if self.optns.xtype in ('abductive', 'abd') else 0
        axp_done, cxp_done, iters = False, False, 0
        target_done = False

        while not target_done:
            if cur_xtype == 1:
                axp_done, cxp_done, iters = axp_enumeration(xnum, hitman=hitmans[cur_xtype],
                                                            hitman_cxp=hitmans[0],
                                axp_done=axp_done, cxp_done=cxp_done, iters=iters,
                                num_switch=self.num_switch)
            else:
                axp_done, cxp_done, iters = cxp_enumeration(xnum, hitman=hitmans[cur_xtype],
                                                            hitman_axp=hitmans[1],
                                axp_done=axp_done, cxp_done=cxp_done, iters=iters,
                                num_switch=self.num_switch, reduce_=self.optns.reduce)
            #hitman.switch_phase()
            target_done = axp_done if self.optns.xtype in ('abductive', 'abd') else cxp_done
            cur_xtype = (cur_xtype+1)%2
            self.num_switch += 1
            print('switch:', self.num_switch)

        if self.optns.xtype in ('abductive', 'abd'):
            # result
            self.expls = uni_axps + axps
            # Dual explanations
            self.duals = uni_cxps + cxps
        else:
            # result
            self.expls = uni_cxps + cxps
            # Dual explanations
            self.duals = uni_axps + axps

        del self.allxps
        del self.uni_cxps

    def _cats2hypos(self, scats):
        """
            Translate selected categories into propositional hypotheses.
        """
        return list(reduce(lambda x, y: x + self.hypos[y[0] : y[1] + 1],
            [self.fcats[c] for c in scats], []))

    def prepare_knowledge(self):
        #print('self.ivars:')
        #print(self.ivars)
        #print('self.intvs:')
        #print(self.intvs)
        #print('self.xgb.feature_names:')
        #print(self.xgb.feature_names)
        #print('self.xgb.categorical_features:')
        #print(self.xgb.categorical_features)
        #print('self.xgb.categorical_names:')
        #print(self.xgb.categorical_names)

        #vpool = IDPool(start_from=self.formulas[clid].nv + 1)
        #self.vpools = {clid: IDPool(start_from=self.oracle.formulas[clid].nv+1)
        #               for clid in self.oracle.formulas if clid != self.out_id}
        #IDPool(start_from=2) OR IDPool(start_from=max(nv)+1)

        #self.c2knowledge = {clid : [] for clid in self.oracle.formulas if clid != self.out_id}
        self.knowledge = []
        """
        prepare knowledge
        """

        def iscat(feature_, f2cat):
            if feature_ not in f2cat:
                if feature_.count('<') == 2:
                    feature = feature_.replace('<=', '<').split('<')
                    value = '[{},{})'.format(feature[0].strip(), feature[-1].strip())
                    feature = feature[1].strip()
                elif feature_.count('>') == 2:
                    #todo
                    #print('todo: not supported:', feature_)
                    #feature = feature_.replace('>=', '>').split('>')[1].strip()
                    exit(1)
                elif feature_.count('<') == 1:
                    feature = feature_.replace('<=', '<').split('<')
                    value = '(-int,{})'.format(feature[-1].strip())
                    feature = feature[0].strip()
                elif feature_.count('>') == 1:
                    feature = feature_.replace('>=', '>').split('>')
                    value = '[{},int)'.format(feature[-1])
                    feature = feature[0].strip()
                else:
                    feature = feature_
                    value = None
                try:
                    fid = self.xgb.feature_names.index(feature)
                except:
                    fid = self.xgb.feature_names.index(feature.split(maxsplit=1)[0].strip("'").strip('"').strip())
                is_cat = fid in self.xgb.categorical_names
                f2cat[feature_] = [fid, is_cat, feature, value]
            else:
                fid, is_cat, feature, value = f2cat[feature_]
            #print('feature:', feature)
            #print('feature index:', fid)
            #print('is_cat:', is_cat)
            return fid, is_cat, feature, value

        def get_vars(feature, fid, is_cat, value, sign=True, fv2id={}):
            if tuple([feature, fid, is_cat, value, sign]) not in fv2id:
                s = 1 if sign else -1
                if is_cat:
                    cat_names = self.xgb.categorical_names[fid]
                    try:
                        try:
                            vid = cat_names.index(value)
                        except:
                            vid = cat_names.index(value + '.0')
                    except:
                        cat_names = list(map(lambda l: l.replace(' ', ''), cat_names))
                        vid = cat_names.index(value)
                else:
                    try:
                        thresholds = self.intvs['f{0}'.format(fid)]
                        #print('thresholds:', thresholds)
                    except:
                        fv2id[tuple([feature, fid, is_cat, value, sign])] = [None]
                        return [None]

                    if ',int' in value:
                        vid = len(thresholds) - 1
                    else:
                        try:
                            vid = thresholds.index(float(value.rsplit(',', maxsplit=1)[-1].strip(')]')))
                        except:
                            for vid, thr in enumerate(thresholds[:-1]):
                                if float(value) < thr:
                                    break
                            else:
                                #print('feature:', feature)
                                #print('value:', value)
                                vid = len(thresholds) - 1
                                #print('thresholds:', thresholds)
                                #print('vid:', vid)

                    #s = s if int(value) == 1 else -s
                    #if (vid == len(thresholds) - 2) and (not s):
                    #    vid += 1
                    #    s = -s

                var = None
                vars = []
                if is_cat:
                    if len(self.xgb.categorical_names[fid]) > 2:
                        try:
                            var = self.ivars['f{0}_{1}'.format(fid, vid)][-1] * s
                        except:
                            # NOT in BTs
                            #for clid in self.oracle.formulas:
                            #    if clid != self.out_id:
                            #        var = -self.vpools[clid].id('f{0}_{1}'.format(fid, vid)) * s
                            #        vars.append(var)
                            vars = [None]
                    else:
                        try:
                            var = -self.ivars['f{0}_0'.format(fid)][vid] * s
                        except:
                            # NOT in BTs zoo
                            #for clid in self.oracle.formulas:
                            #    if clid != self.out_id:
                            #        if vid == 0:
                            #            var = self.vpools[clid].id('f{0}_0'.format(fid)) * s
                            #        else:
                            #            var = -self.vpools[clid].id('f{0}_0'.format(fid)) * s
                            #        vars.append(var)
                            vars = [None]
                else:
                    #print("self.ivars['f{0}'.format(fid)]:", self.ivars['f{0}'.format(fid)])
                    var = self.ivars['f{0}'.format(fid)][vid] * s

                if len(vars) == 0:
                    vars.append(var)
                fv2id[tuple([feature, fid, is_cat, value, sign])] = vars
            else:
                vars = fv2id[tuple([feature, fid, is_cat, value, sign])]

            return vars

        #with open('../../datasets/pneumoniamnist/csv/complete_data.csv.pkl', 'rb') as f:
        #    info = pickle.load(f)
        #print('info:')
        #print(info)

        if self.optns.knowledge.endswith('.xz'):
            with lzma.open(self.optns.knowledge, 'r') as f:
                knowledge = json.load(f)
        else:
            with open(self.optns.knowledge, 'r') as f:
                knowledge = json.load(f)

        f2cat = {}
        fv2id = {}
        for lname in knowledge:
            fid, is_cat, new_lname, new_lvalue = iscat(lname, f2cat)

            for lvalue in knowledge[lname]:
                if lvalue.lower() == 'true':#, '1.0', '1'):
                    label_value = True
                elif lvalue.lower() == 'false':#, '0.0', '0'):
                    label_value = False
                else:
                    label_value = str(lvalue)

                #print('lname:', lname)
                #print('label_value:', label_value)
                #print('new_lname:', new_lname)
                if lname != new_lname:
                    sign = lvalue.lower() in ('true', '1.0', '1')
                    label_value = new_lvalue
                else:
                    sign = True
                #print('label_value:', label_value)
                #print('sign:', sign)

                label_vars = get_vars(lname, fid, is_cat, label_value, sign=sign, fv2id=fv2id)
                #print('label_vars:', label_vars)
                #print()
                if label_vars[0] is None:
                    continue
                #print('vars:', vars)
                #print()
                # going through all rules with label lname_lvalue
                for imp in knowledge[lname][lvalue]:
                    imp_vars = []
                    if len(imp) > 0:
                        for finfo in imp:
                            feature = finfo['feature']
                            value = finfo['value']
                            #print('feature:', feature)
                            #print('value:', value)
                            if value.lower() == 'true':#in ('true', '1.0', '1'):
                                value = True
                            elif value.lower() == 'false':#, '0.0', '0'):
                                value = False
                            else:
                                value = str(value)
                            sign = finfo['sign']
                            #print('sign:', sign)
                            fid_, is_cat_, new_fname, new_fvalue = iscat(feature, f2cat)
                            #print('new_fname:', new_fname)
                            if feature != new_fname:
                                sign = sign if finfo['value'].lower() in ('true', '1.0', '1') else not sign
                                value = new_fvalue

                            vars = get_vars(feature, fid_, is_cat_, value, sign=sign, fv2id=fv2id)
                            #print('vars:', vars)
                            #print()
                            imp_vars.append(vars)
                            if vars[0] is None:
                                break

                        if imp_vars[-1][0] is None:
                            continue

                    #clauses = {clid: [] for clid in self.c2knowledge}
                    clause = []
                    for vars in imp_vars:
                        clause.append(-vars[0])
                        #for i, clid in enumerate(self.c2knowledge):
                        #    if len(vars) == 1:
                        #        clauses[clid].append(-vars[0])
                        #    else:
                        #        clauses[clid].append(-vars[i])

                    #for i, clid in enumerate(self.c2knowledge):
                    #    if len(label_vars) == 1:
                    #        clauses[clid].append(label_vars[0])
                    #    else:
                    #        clauses[clid].append(label_vars[i])
                    clause.append(label_vars[0])
                    #for clid in self.c2knowledge:
                    #    self.c2knowledge[clid].append(clauses[clid])
                    self.knowledge.append(clause)

        #aa = {abs(l) for c in self.knowledge for l in c}
        #print(max(aa))
        #exit()

    #def rm_inconsist_bg(self, atmost1, curtop):
    #    encoded_knowledge = []
    #    t2cid = {}

    #    top = curtop
    #    for i, cl in enumerate(self.knowledge):
    #        top += 1
    #        encoded_knowledge.append(cl + [-top])
    #        t2cid[top] = i

    #    oracle = SATSolver(name=self.optns.solver, bootstrap_with=encoded_knowledge + atmost1)

    #    # for h in self._cats2hypos(self.allcats):
    #    for h in self.hypos:
    #        oracle.add_clause([h])

    #    assump = list(t2cid.keys())
    #    st, prop = oracle.propagate(assumptions=assump)
    #    notuse = []
    #    print('top:', top)
    #    while not st:
    #        print('prop:', prop)
    #        unsat_ids = assump.index(prop[-1]) + 1 if len(prop) > 0 else 0
    #        notuse.append(assump[unsat_ids])

    #        try:
    #            assump = assump[unsat_ids + 1:]
    #            st, prop = oracle.propagate(assumptions=assump)
    #        except:
    #            st = True

    #    use = set(t2cid.keys()).difference(set(notuse))
    #    self.knowledge = [self.knowledge[t2cid[t]] for t in use]

    def filter_knowledge(self):
        """

        # Propagation

        """

        encoded_knowledge = []
        t2cid = {}

        #for clid in self.c2knowledge:
        #    top = self.vpools[clid].top
        #    for i, cl in enumerate(self.c2knowledge[clid]):
        #        top += 1
        #        encoded_knowledge.append(cl + [-top])
        #        t2cid[top] = i
        #    break

        top = max([abs(var) for ivars in self.ivars.values() for var in ivars])
        for i, cl in enumerate(self.knowledge):
            top += 1
            encoded_knowledge.append(cl + [-top])
            t2cid[top] = i

        oracle = SATSolver(name=self.optns.solver, bootstrap_with=encoded_knowledge)

        #for h in self._cats2hypos(self.allcats):
        for h in self.hypos:
            oracle.add_clause([h])

        assump = list(t2cid.keys())
        st, prop = oracle.propagate(assumptions=assump)
        notuse = []

        while not st:
            unsat_ids = assump.index(prop[-1]) + 1 if len(prop) > 0 else 0
            notuse.append(assump[unsat_ids])

            try:
                assump = assump[unsat_ids + 1:]
                st, prop = oracle.propagate(assumptions=assump)
            except:
                st = True

        use = set(t2cid.keys()).difference(set(notuse))

        for i, t in enumerate(sorted(use)):
            cid = t2cid[t]
            for clid in self.oracle.oracles:
                if clid == self.out_id:
                    continue
                cl = self.knowledge[cid]
                if self.oracle.ortype == 'int':
                    self.oracle.oracles[clid].add_clause(cl)
                else:
                    self.oracle.formulas[clid].append(cl)

#
#==============================================================================
class MXIExplainer(object):
    """
        A MaxSAT-based explainer based on the model's internal literals.
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
        self.idmgr = IDPool()
        self.fcats = []

        # in case we want to compute explanations wrt. internal BT literals
        self.lvars = xgb.mxe.lvars

        # saving XGBooster
        self.xgb = xgb

        self.verbose = self.optns.verb

        # MaxSAT-based oracles
        self.oracles = {}
        if self.optns.encode == 'mxa':
            ortype = 'alien'
        elif self.optns.encode == 'mxe':
            ortype = 'ext'
        else:
            ortype = 'int'
        for clid in range(nof_classes):
            self.oracles[clid] = MXReasoner(formula, clid,
                    solver=self.optns.solver,
                    oracle=ortype,
                    am1=self.optns.am1, exhaust=self.optns.exhaust,
                    minz=self.optns.minz, trim=self.optns.trim)

        # a reference to the current oracle
        self.oracle = None

        # SAT-based predictor
        self.poracle = SATSolver(name='g3')
        for clid in range(nof_classes):
            self.poracle.append_formula(formula[clid].formula)

        # interval connections oracle
        self.coracle = SATSolver(name='g3')
        # for feat in self.lvars:
        #     if len(self.lvars[feat]) > 2:
        #         for i, lit in enumerate(self.lvars[feat][:-2]):
        #             self.coracle.add_clause([lit, -self.lvars[feat][i + 1]])

        # determining which features should go hand in hand
        categories = collections.defaultdict(lambda: [])
        for f in self.xgb.extended_feature_names_as_array_strings:
            # print(f, self.ivars)
            if f in self.ivars:
                if '_' in f or len(self.ivars[f]) == 2:
                    categories[f.split('_')[0]].append(self.xgb.mxe.vpos[self.ivars[f][0]])
                else:
                    for v in self.ivars[f]:
                        # this has to be checked and updated
                        categories[f].append(self.xgb.mxe.vpos[abs(v)])

        # these are the result indices of features going together
        self.fcats = [[min(ftups), max(ftups)] for ftups in categories.values()]
        self.fcats_copy = self.fcats[:]

        # all used feature categories
        self.allcats = list(range(len(self.fcats)))

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

        # actual names for the variables used
        # self.names = {}
        # print('tt0', self.xgb.extended_feature_names_as_array_strings)
        # print('tt1', self.xgb.feature_names)
        # print('tt2', self.vid2fid)
        # print('map', self.v2feat)
        # for feat, name in zip(self.xgb.extended_feature_names_as_array_strings, self.xgb.feature_names):
        #     if feat in self.intvs:
        #         # determining the right interval and the corresponding variable
        #         for i, (ub, fvar) in enumerate(zip(self.intvs[feat][:-1], self.lvars[feat][:-1])):
        #             self.names[+fvar] = '{0} < {1}'.format(name, ub)
        #             self.names[-fvar] = '{0} >= {1}'.format(name, ub)

        #         # # here we are using the last known ub
        #         # self.names[self.lvars[feat][-1]] = '{0} >= {1}'.format(name, ub)
        # print('nnn1', self.names)

        self.names = {}
        for feat, intvs in self.intvs.items():
            # determining the right interval and the corresponding variable
            for i, (ub, fvar) in enumerate(zip(self.intvs[feat][:-1], self.lvars[feat][:-1])):
                name = self.xgb.feature_names[self.v2feat[fvar]]
                cfeat = self.v2feat[fvar] in self.xgb.categorical_names
                if cfeat:
                    all_feature_values = self.xgb.categorical_names[self.v2feat[fvar]]
                    if len(all_feature_values) > 2:
                        fvid = int(feat.split('_', maxsplit=1)[-1])
                        if name not in str(all_feature_values[fvid]):
                            self.names[+fvar] = '{0} != {1}'.format(name, all_feature_values[fvid])
                            self.names[-fvar] = '{0} == {1}'.format(name, all_feature_values[fvid])
                        else:
                            self.names[+fvar] = 'NOT {0}'.format(all_feature_values[fvid])
                            self.names[-fvar] = str(all_feature_values[fvid])
                    else:
                        if name not in str(all_feature_values[0]) or name == str(all_feature_values[0]):
                           self.names[+fvar] = '{0} == {1}'.format(name, all_feature_values[1])
                        else:
                            self.names[+fvar] = str(all_feature_values[1])

                        if name not in str(all_feature_values[1]) or name == str(all_feature_values[1]):
                            self.names[-fvar] = '{0} == {1}'.format(name, all_feature_values[0])
                        else:
                            self.names[-fvar] = str(all_feature_values[0])
                else:
                    self.names[+fvar] = '{0} < {1}'.format(name, ub)
                    self.names[-fvar] = '{0} >= {1}'.format(name, ub)

        # number of oracle calls involved
        self.calls = 0

    def __del__(self):
        """
            Destructor.
        """

        self.delete()

    def delete(self):
        """
            Actual destructor.
        """

        # deleting MaxSAT-based reasoners
        if self.oracles:
            for clid, oracle in self.oracles.items():
                if oracle:
                    oracle.delete()
            self.oracles = {}
        self.oracle = None

        # deleting the SAT-based predictor
        if self.poracle:
            self.poracle.delete()
            self.poracle = None

        # deleting the interval connections oracle
        if self.coracle:
            self.coracle.delete()
            self.coracle = None

    def predict(self, sample):
        """
            Run the encoding and determine the corresponding class.
        """

        # translating sample into assumption literals
        self.hypos = self.xgb.mxe.get_literals(sample)

        #print('hypos', self.hypos)

        # running the solver to propagate the prediction;
        # using solve() instead of propagate() to be able to extract a model
        assert self.poracle.solve(assumptions=self.hypos), 'Formula must be satisfiable!'
        model = self.poracle.get_model()

        # computing all the class scores
        scores = {}
        for clid in range(self.nofcl):
            # computing the value for the current class label
            scores[clid] = 0

            for lit, wght in self.xgb.mxe.enc[clid].leaves:
                if model[abs(lit) - 1] > 0:
                    scores[clid] += wght

        x = 1 if self.optns.xtype == 'abd' else -1

        # here is the full list of hypotheses over the language of the model
        self.hfull = []
        self.conns = []


        if self.optns.ilits:
            for feat in self.lvars:
                if len(self.lvars[feat]) > 2:
                    # first the negative part
                    stack = []
                    for i, lit in enumerate(self.lvars[feat][:-1]):
                        if model[abs(lit) - 1] == lit:
                            break
                        stack.append(-lit)
                    else:
                        i += 1

                    # adding negative connections
                    for j in range(len(stack) - 1):
                        self.coracle.add_clause([x * stack[j], -x * stack[j + 1]])
                        self.conns.append([x * stack[j], -x * stack[j + 1]])

                    # adding negative literals to hypotheses
                    self.hfull += [stack.pop() for v in range(len(stack))]

                    # second, the positive part
                    for j in range(i, len(self.lvars[feat]) - 1):
                        self.hfull.append(self.lvars[feat][j])

                    # collecting positive connections
                    for j in range(i, len(self.lvars[feat]) - 2):
                        self.coracle.add_clause([-x * self.lvars[feat][j], x * self.lvars[feat][j + 1]])
                        self.conns.append([-x * self.lvars[feat][j], x * self.lvars[feat][j + 1]])

                else:
                    # there is a single Boolean variable used for this feature
                    self.hfull.append(model[abs(self.lvars[feat][0]) - 1])

        # feature literal order
        self.order = {l: i for i, l in enumerate(self.hfull)}

        self.hfull = sorted(set(self.hfull))
        # print('intvs', self.intvs)
        # print('ivars', self.ivars)
        # print('lvars', self.lvars)
        # print('hypos', self.hypos)
        # print('hfull', self.hfull)
        # print('names', self.names)
        # print('conns', self.conns)

        # variable to the category in use; this differs from
        # v2feat as here we may not have all the features here
        self.v2cat = {}
        for i, v in enumerate(self.hfull):
            self.v2cat[v] = i

        # returning the class corresponding to the max score
        return max(list(scores.items()), key=lambda t: t[1])[0]

    def prepare(self, sample):
        """
            Prepare the oracle for computing an explanation.
        """
        # first, we need to determine the prediction, according to the model
        self.out_id = self.predict(sample)

        # selecting the right oracle
        self.oracle = self.oracles[self.out_id]

        # if self.selv:
        #     # disable the previous assumption if any
        #     self.oracle.add_assertion(Not(self.selv))

        # # creating a fresh selector for a new sample
        # sname = ','.join([str(v).strip() for v in sample])

        # # the samples should not repeat; otherwise, they will be
        # # inconsistent with the previously introduced selectors
        # assert sname not in self.idmgr.obj2id, 'this sample has been considered before (sample {0})'.format(self.idmgr.id(sname))
        # self.selv = Symbol('sample{0}_selv'.format(self.idmgr.id(sname)), typename=BOOL)

        # transformed sample
        self.sample = list(self.xgb.transform(sample)[0])

        # correct class id (corresponds to the maximum computed)
        self.output = self.xgb.target_name[self.out_id]

        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.xgb.feature_names, inpvals):
                if f not in str(v):
                    self.preamble.append('{0} == {1}'.format(f, v))
                else:
                    self.preamble.append(str(v))

            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def explain(self, sample, smallest, expl_ext=None, prefer_ext=False):
        """
            Hypotheses minimization.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # adapt the solver to deal with the current sample
        self.prepare(sample)
        if self.optns.encode != 'mxe':
            # dummy call with the full instance to detect all the necessary cores
            self.oracle.get_coex(self.hfull, full_instance=True, early_stop=True)

        #if self.optns.knowledge:
        #    self.prepare_knowledge()
        #    self.filter_knowledge()

        # calling the actual explanation procedure
        self._explain(sample, smallest=smallest, xtype=self.optns.xtype,
                xnum=self.optns.xnum, unit_mcs=self.optns.unit_mcs,
                reduce_=self.optns.reduce)

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        if self.verbose:
            #print('expls', self.expls)
            for expl in self.expls:
                preamble = [self.names[i] for i in expl]
                label = self.xgb.target_name[self.out_id]

                if self.optns.xtype in ('contrastive', 'con'):
                    preamble = [l.replace('>=', '==') for l in preamble]
                    preamble = [l.replace('<', '>=') for l in preamble]
                    preamble = [l.replace('==', '<') for l in preamble]
                    label = 'NOT {0}'.format(label)

                print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(preamble), label))
                print('  # hypos left:', len(expl))

            print('  calls:', self.calls)
            print('  rtime: {0:.2f}'.format(self.time))

        return self.expls

    def _explain(self, sample, smallest=True, xtype='abd', xnum=1,
            unit_mcs=False, reduce_='none'):
        """
            Compute an explanation.
        """

        if xtype in ('abductive', 'abd'):
            # abductive explanations => MUS computation and enumeration
            if not smallest and xnum == 1:
                self.expls = [self.extract_mus(reduce_=reduce_)]
            else:
                self.mhs_mus_enumeration(xnum, smallest=smallest)
        else:  # contrastive explanations => MCS enumeration
            self.mhs_mcs_enumeration(xnum, smallest, reduce_)

    def extract_mus(self, reduce_='lin', start_from=None):
        """
            Compute one abductive explanation.
        """

        def _do_linear(core):
            """
                Do linear search.
            """

            def _assump_needed(a):
                if len(to_test) > 1:
                    to_test.remove(a)
                    self.calls += 1
                    # actual binary hypotheses to test
                    # print('testing', to_test, self._cats2hypos(to_test), core)
                    # print(self.v2cat)
                    if not self.oracle.get_coex(to_test, early_stop=True):
                        # print('cost', self.oracle.oracles[1].cost)
                        return False
                    # print('cost', self.oracle.oracles[1].cost)
                    to_test.add(a)
                    return True
                else:
                    return True

            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))

        def _do_quickxplain(core):
            """
                Do QuickXplain-like search.
            """

            wset = core[:]
            filt_sz = len(wset) / 2.0
            while filt_sz >= 1:
                i = 0
                while i < len(wset):
                    to_test = wset[:i] + wset[(i + int(filt_sz)):]
                    # actual binary hypotheses to test
                    self.calls += 1
                    if to_test and not self.oracle.get_coex(to_test, early_stop=True):
                        # assumps are not needed
                        wset = to_test
                    else:
                        # assumps are needed => check the next chunk
                        i += int(filt_sz)
                # decreasing size of the set to filter
                filt_sz /= 2.0
                if filt_sz > len(wset) / 2.0:
                    # next size is too large => make it smaller
                    filt_sz = len(wset) / 2.0
            return wset

        self.fcats = self.fcats_copy[:]

        # this is our MUS over-approximation
        if start_from is None:
            assert self.oracle.get_coex(self.hfull, full_instance=True, early_stop=True) == None, 'No prediction'

            # getting the core
            core = sorted(self.oracle.get_reason(), key=lambda l: self.order[l])
        else:
            core = start_from

        if self.verbose > 2:
            print('core:', core)

        self.calls = 1  # we have already made one call

        if reduce_ == 'qxp':
            expl = _do_quickxplain(core)
        else:  # by default, linear MUS extraction is used
            expl = _do_linear(core)

        return expl

    def mhs_mus_enumeration(self, xnum, smallest=False):
        """
            Enumerate subset- and cardinality-minimal explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (contrastive) explanations
        self.duals = []

        with Hitman(bootstrap_with=[self.hfull], htype='sorted' if smallest else 'lbx') as hitman:
            # adding negated literals to the mapping
            for l in self.hfull:
                hitman.idpool.obj2id[-l] = -hitman.idpool.obj2id[l]

            # adding all the interval connections
            for c in self.conns:
                cc = list(map(lambda lit: hitman.idpool.id(lit), c))
                hitman.oracle.add_clause(cc)

            # computing unit-size MCSes
            if self.optns.unit_mcs:
                for i in range(len(self.hfull)):
                    self.calls += 1
                    if self.oracle.get_coex(self.hfull[:i] + self.hfull[(i + 1):], early_stop=True):
                        hitman.hit([self.hfull[i]])
                        self.duals.append([self.hfull[i]])

            if self.verbose > 2:
                print('dual:', self.duals)

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                # applying candidate reduction based on interval dependencies
                if hset:
                    hset = self.reduce_xp(hset, axp=True)

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                coex = self.oracle.get_coex(hset, early_stop=True)
                if coex:
                    # print('coeo:', coex)
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(self.hfull).difference(set(hset)))

                    for h in removed:
                        if coex[abs(h) - 1] != h:
                            unsatisfied.append(h)
                        else:
                            hset.append(h)

                    # unsatisfied = sorted(list(set(unsatisfied)), key=lambda v: self.order[v], reverse=True)
                    unsatisfied = sorted(list(set(unsatisfied)), key=lambda v: self.order[v])
                    hset = list(set(hset))

                    # print('chck:', unsatisfied)
                    # computing an MCS (expensive)
                    while unsatisfied:
                        self.calls += 1
                        lit = unsatisfied.pop()

                        st, props = self.coracle.propagate(assumptions=[lit])
                        # props = []
                        assert st, 'Connections solver propagated to False!'

                        props = list(set(props).intersection(set(unsatisfied)))
                        # print('props', lit, props)
                        if self.oracle.get_coex(hset + [lit] + props, early_stop=True):
                            hset.append(lit)
                        else:
                            to_hit.append(lit)

                            # dropping all the related literals at once
                            # unsatisfied = sorted(set(unsatisfied).difference(set(props)), key=lambda v: self.order[v], reverse=True)
                            unsatisfied = sorted(set(unsatisfied).difference(set(props)), key=lambda v: self.order[v])

                    if self.verbose > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)

                    #self.duals.append([to_hit])
                    self.duals.append(to_hit)
                else:
                    if self.verbose > 2:
                        print('expl:', hset)

                    self.expls.append(hset)

                    if len(self.expls) != xnum:
                        hitman.block(hset)
                    else:
                        break

    def mhs_mcs_enumeration(self, xnum, smallest=False, reduce_='none', unit_mcs=False):
        """
            Enumerate subset- and cardinality-minimal contrastive explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (abductive) explanations
        self.duals = []

        with Hitman(bootstrap_with=[self.hfull], htype='sorted' if smallest else 'lbx') as hitman:
            # adding negated literals to the mapping
            for l in self.hfull:
                hitman.idpool.obj2id[-l] = -hitman.idpool.obj2id[l]

            # adding all the interval connections
            for c in self.conns:
                cc = list(map(lambda lit: hitman.idpool.id(lit), c))
                hitman.oracle.add_clause(cc)

            # computing unit-size MUSes
            for i in range(len(self.hfull)):
                self.calls += 1

                if not self.oracle.get_coex([self.hfull[i]], early_stop=True):
                    hitman.hit([self.hfull[i]])
                    self.duals.append([self.hfull[i]])
                elif self.optns.unit_mcs and self.oracle.get_coex(self.hfull[:i] + self.hfull[(i + 1):], early_stop=True):
                    # this is a unit-size MCS => block immediately
                    self.calls += 1
                    hitman.block([self.hfull[i]])
                    self.expls.append([self.hfull[i]])

            # allhypos = self.hfull[:]
            # while allhypos:
            #     lit = allhypos.pop()

            #     self.calls += 1
            #     if not self.oracle.get_coex([lit], early_stop=True):
            #         hitman.hit([lit])
            #         self.duals.append([lit])
            #     elif self.optns.unit_mcs:
            #         self.calls += 1

            #         st, props = self.coracle.propagate(assumptions=[lit])
            #         assert st, 'Connections solver propagated to False!'

            #         # props = list(set(props).intersection(set(allhypos)))
            #         props = []
            #         print('props', lit, props)

            #         assumps = sorted(set(self.hfull).difference(set([lit] + props)), key=lambda v: self.order[v])
            #         if self.oracle.get_coex(assumps, early_stop=True):
            #             hitman.block([lit])
            #             self.expls.append([lit])

            #             # dropping all the related literals at once
            #             allhypos = sorted(set(allhypos).difference(set(props)), key=lambda v: self.order[v])

            if self.verbose > 2:
                print('dual:', self.duals)
                print('expl:', self.expls)

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                # st, hset_aug = self.coracle.propagate(assumptions=hset)
                hset_aug = hset[:]
                # assert st, 'Connections solver propagated to False!'

                # if self.verbose > 2:
                #     print('haug:', sorted(set(self.hfull).intersection(set(hset_aug)), key=lambda v: self.order[v]))

                self.calls += 1
                # hypos = sorted(set(self.hfull).difference(set(hset)), key=lambda v: self.order[v])
                if not self.oracle.get_coex(set(self.hfull).difference(set(hset_aug)), early_stop=True):
                    # to_hit = sorted(self.oracle.get_reason(), key=lambda v: self.order[v])
                    to_hit = sorted(set(self.hfull).difference(set(hset)), key=lambda v: self.order[v])

                    if len(to_hit) > 1 and reduce_ != 'none':
                        to_hit = self.extract_mus(reduce_=reduce_, start_from=to_hit)

                    self.duals.append(to_hit)

                    if self.verbose > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    hset = self.reduce_xp(hset, axp=False)
                    if self.verbose > 2:
                        print('expl:', hset)

                    self.expls.append(hset)

                    if len(self.expls) != xnum:
                        hitman.block(hset)
                    else:
                        break

    def reduce_xp(self, expl, axp=True):
        """
            Get rid of redundant literals in an explanation. This is based
            on the trivial dependencies between feature intervals.
        """

        expl = sorted(expl, key=lambda v: self.order[v], reverse=not axp)

        i = 0
        while i < len(expl):
            lit = expl[i]
            st, props = self.coracle.propagate(assumptions=[lit])

            expl = expl[:i + 1] + sorted(list(set(expl[i + 1:]).difference(set(props))),
                    key=lambda v: self.order[v], reverse=not axp)

            i += 1
            i += 1

        return expl

    def prepare_knowledge(self):
        #print('self.lvars:')
        #print(self.lvars)
        #print('self.intvs:')
        #print(self.intvs)
        #print('self.xgb.feature_names:')
        #print(self.xgb.feature_names)
        #print('self.xgb.categorical_features:')
        #print(self.xgb.categorical_features)
        #print('self.xgb.categorical_names:')
        #print(self.xgb.categorical_names)

        #vpool = IDPool(start_from=self.formulas[clid].nv + 1)
        #self.vpools = {clid: IDPool(start_from=self.oracle.formulas[clid].nv+1)
        #               for clid in self.oracle.formulas if clid != self.out_id}
        #IDPool(start_from=2) OR IDPool(start_from=max(nv)+1)

        #self.c2knowledge = {clid : [] for clid in self.oracle.formulas if clid != self.out_id}
        self.knowledge = []

        """
        prepare knowledge
        """

        def iscat(feature):
            try:
                fid = self.xgb.feature_names.index(feature)
            except:
                fid = self.xgb.feature_names.index(feature.split(maxsplit=1)[0].strip("'").strip('"').strip())
            is_cat = fid in self.xgb.categorical_names
            # print('feature:', feature)
            # print('feature index:', fid)
            # print('is_cat:', is_cat)

            return fid, is_cat

        def get_vars(feature, fid, is_cat, value, sign=True):
            s = 1 if sign else -1
            if is_cat:
                cat_names = self.xgb.categorical_names[fid]
                try:
                    vid = cat_names.index(value)
                except:
                    cat_names = list(map(lambda l: l.replace(' ', ''), cat_names))
                    vid = cat_names.index(value)
                # print('value_id:', vid)
            else:
                try:
                    thresholds = self.intvs['f{0}'.format(fid)]
                except:
                    return [None]
                if '>=' in feature:
                    vid = len(thresholds) - 1
                else:
                    vid = thresholds.index(float(feature.rsplit(maxsplit=1)[-1]))
                s = s if int(value) == 1 else -s
                if vid == len(thresholds) - 2:
                    vid += 1
                    s = -s

            var = None
            vars = []
            if is_cat:
                if len(self.xgb.categorical_names[fid]) > 2:
                    try:
                        var = self.lvars['f{0}_{1}'.format(fid, vid)][-1] * s
                    except:
                        # NOT in BTs
                        #for clid in self.oracle.formulas:
                        #    if clid != self.out_id:
                        #        var = -self.vpools[clid].id('f{0}_{1}'.format(fid, vid)) * s
                        #        vars.append(var)
                        vars = [None]
                else:
                    try:
                        var = -self.lvars['f{0}_0'.format(fid)][vid] * s
                    except:
                        # NOT in BTs zoo
                        #for clid in self.oracle.formulas:
                        #    if clid != self.out_id:
                        #        if vid == 0:
                        #            var = self.vpools[clid].id('f{0}_0'.format(fid)) * s
                        #        else:
                        #            var = -self.vpools[clid].id('f{0}_0'.format(fid)) * s
                        #        vars.append(var)
                        vars = [None]
            else:
                var = self.lvars['f{0}'.format(fid)][vid] * s

            if len(vars) == 0:
                vars.append(var)

            return vars

        with open(self.optns.knowledge, 'r') as f:
            knowledge = json.load(f)


        for lname in knowledge:
            fid, is_cat = iscat(lname)

            for lvalue in knowledge[lname]:
                if lvalue.lower() == 'true':
                    label_value = True
                elif lvalue.lower() == 'false':
                    label_value = False
                else:
                    label_value = str(lvalue)
                #print('lname:', lname)
                #print('label_value:', label_value)

                labeL_vars = get_vars(lname, fid, is_cat, label_value, sign=True)
                if labeL_vars[0] is None:
                    continue
                #print('vars:', vars)
                #print()

                # going through all rules with label lname_lvalue
                for imp in knowledge[lname][lvalue]:
                    imp_vars = []
                    for finfo in imp:
                        feature = finfo['feature']
                        value = finfo['value']
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        else:
                            value = str(value)
                        sign = finfo['sign']

                        fid_, is_cat_ = iscat(feature)

                        #print('feature:', feature)
                        #print('value:', value)
                        #print('sign:', sign)
                        vars = get_vars(feature, fid_, is_cat_, value, sign=sign)
                        #print('vars:', vars)
                        #print()
                        imp_vars.append(vars)
                        if vars[0] is None:
                            break

                    if imp_vars[-1][0] is None:
                        continue

                    clause = [] #clauses = {clid: [] for clid in self.c2knowledge}
                    for vars in imp_vars:
                        clause.append(-vars[0])
                        #for i, clid in enumerate(self.c2knowledge):
                            #if len(vars) == 1:
                            #    clauses[clid].append(-vars[0])
                            #else:
                            #    clauses[clid].append(-vars[i])

                    clause.append(labeL_vars[0])
                    #for i, clid in enumerate(self.c2knowledge):
                    #    if len(labeL_vars) == 1:
                    #        clauses[clid].append(labeL_vars[0])
                    #    else:
                    #        clauses[clid].append(labeL_vars[i])

                    #for clid in self.c2knowledge:
                    #    self.c2knowledge[clid].append(clauses[clid])
                    self.knowledge.append(clause)


    def filter_knowledge(self):
        """

        # Propagation

        """

        encoded_knowledge = []
        t2cid = {}
        top = max([abs(var) for lvars in self.lvars.values() for var in lvars])
        for i, cl in enumerate(self.knowledge):
            top += 1
            encoded_knowledge.append(cl + [-top])
            t2cid[top] = i

        oracle = SATSolver(name=self.optns.solver, bootstrap_with=encoded_knowledge)

        for h in self.hfull:
            oracle.add_clause([h])

        assump = list(t2cid.keys())
        st, prop = oracle.propagate(assumptions=assump)
        notuse = []
        while not st:
            unsat_ids = assump.index(prop[-1]) + 1 if len(prop) > 0 else 0
            notuse.append(assump[unsat_ids])

            try:
                assump = assump[unsat_ids + 1:]
                st, prop = oracle.propagate(assumptions=assump)
            except:
                st = True

        use = set(t2cid.keys()).difference(set(notuse))

        for t in use:
            cid = t2cid[t]
            for clid in self.oracle.oracles:
                if clid == self.out_id:
                    continue
                cl = self.knowledge[cid]
                if self.oracle.ortype == 'int':
                    self.oracle.oracles[clid].add_clause(cl)
                else:
                    self.oracle.formulas[clid].append(cl)
