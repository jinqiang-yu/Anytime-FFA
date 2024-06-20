#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## compile.py
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
from pysat.card import CardEnc
from pysat.examples.hitman import Hitman
from pysat.formula import IDPool, WCNF, CNF
from pysat.solvers import Solver as SATSolver
from pysat.examples.rc2 import RC2, RC2Stratified
import resource
from six.moves import range
import sys
import json
import math
import decimal
import random
import signal

# checking whether gurobi is available
gurobi_present = True
try:
    import gurobipy as gurobi
except ImportError:
    gurobi_present = False

#
#==============================================================================
class MXCompiler(object):
    """
        A MaxSAT-based compiler of XGBoost models into decision sets.
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
        #for clid in range(nof_classes):
        #    self.oracles[clid] = MXReasoner(formula, clid,
        #            solver=self.optns.solver,
        #            oracle=ortype,
        #            am1=self.optns.am1, exhaust=self.optns.exhaust,
        #            minz=self.optns.minz, trim=self.optns.trim)
        self.ortype = ortype
        self.formula = formula

        # a reference to the current oracle
        self.oracle = None

        # SAT-based predictor
        if not self.optns.clocal:
            self.instgen = SATSolver(name=self.optns.solver)
            for clid in range(nof_classes):
                self.instgen.append_formula(formula[clid].formula)
        else:
            self.instgen = None

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

        random.seed(self.optns.seed)

        if self.optns.lam:
            self.lambda_ = None


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
        if self.instgen:
            self.instgen.delete()
            self.instgen = None

    def predict(self):
        """
            Extract a valid instance.
        """

        assert self.instgen.get_status() == True, 'The previous call returned UNSAT!'
        model = self.instgen.get_model()

        # computing all the class scores
        scores = {}
        for clid in range(self.nofcl):
            # computing the value for the current class label
            scores[clid] = 0

            for lit, wght in self.xgb.mxe.enc[clid].leaves:
                if model[abs(lit) - 1] > 0:
                    scores[clid] += wght

        # here is the full list of hypotheses over the language of the model
        self.hfull = []
        self.conns = []

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
                    self.conns.append([stack[j], -stack[j + 1]])

                # adding negative literals to hypotheses
                self.hfull += [stack.pop() for v in range(len(stack))]

                # second, the positive part
                for j in range(i, len(self.lvars[feat]) - 1):
                    self.hfull.append(self.lvars[feat][j])

                # collecting positive connections
                for j in range(i, len(self.lvars[feat]) - 2):
                    self.conns.append([-self.lvars[feat][j], self.lvars[feat][j + 1]])

            else:
                # there is a single Boolean variable used for this feature
                self.hfull.append(model[abs(self.lvars[feat][0]) - 1])

        # feature literal order
        self.order = {l: i for i, l in enumerate(self.hfull)}

        self.hfull = sorted(set(self.hfull))
        # print('lvars', self.lvars)
        # print('hfull', self.hfull)
        # print('names', self.names)
        # print('conns', self.conns)
        # print('order', self.order)

        # variable to the category in use; this differs from
        # v2feat as here we may not have all the features here
        self.v2cat = {}
        for i, v in enumerate(self.hfull):
            self.v2cat[v] = i

        # returning the class corresponding to the max score
        self.out_id = max(list(scores.items()), key=lambda t: t[1])[0]

        # selecting the right oracle
        #self.oracle = self.oracles[self.out_id]

        self.oracle = MXReasoner(self.formula, self.out_id,
                                 solver=self.optns.solver,
                                 oracle=self.ortype,
                                 am1=self.optns.am1, exhaust=self.optns.exhaust,
                                 minz=self.optns.minz, trim=self.optns.trim)

        # correct class id (corresponds to the maximum computed)
        self.output = self.xgb.target_name[self.out_id]

        # if self.verbose:
        #     inpvals = self.xgb.readable_sample(sample)

        #     self.preamble = []
        #     for f, v in zip(self.xgb.feature_names, inpvals):
        #         if f not in str(v):
        #             self.preamble.append('{0} == {1}'.format(f, v))
        #         else:
        #             self.preamble.append(str(v))

        #     print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

        return self.hfull, self.out_id

    def predict_(self):
        self.out_id = random.choice(list(self.train_info['pred2samples'].keys()))
        samp_id = random.choice(self.train_info['pred2samples'][self.out_id])
        self.conns = self.train_info['connses'][samp_id]
        self.order = self.train_info['orders'][samp_id]
        self.hfull = sorted(self.train_info['hfulls'][samp_id][0])

        self.v2cat = self.train_info['v2cats'][samp_id]
        self.oracle = MXReasoner(self.formula, self.out_id,
                                 solver=self.optns.solver,
                                 oracle=self.ortype,
                                 am1=self.optns.am1, exhaust=self.optns.exhaust,
                                 minz=self.optns.minz, trim=self.optns.trim)

        # correct class id (corresponds to the maximum computed)
        self.output = self.xgb.target_name[self.out_id]

        return self.hfull, self.out_id

    def compile_(self):
        # this is how we will store the rules
        self.rules = {clid: [] for clid in range(self.nofcl)}

        self.train_info = None

        if self.optns.clocal or self.optns.reduce_lit:
            self.train_info = self.prepare_train()

        if self.optns.knowledge:
            self.prepare_knowledge()

        all_rules = set()

        # main loop keeps iterative while there are instances unexplained
        #while self.instgen.solve() == True:
        while self.uncover() == True:

            ctime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime

            # extracting a new instance uncovered by any of the explanations;
            # this way, none of the explanations can be repeated
            # can we still guarantee equivalence?!
            if not self.optns.clocal:
                inst, clid = self.predict()
            else:
                inst, clid = self.predict_()

            if self.verbose > 2:
                print('premises: {0}'.format(' AND '.join([self.names[l] for l in inst])))
                print('class:', self.output)

            if self.optns.use_duals:
                # these are all dual explanations (counterexamples) we know of so far
                duals = self.get_duals(inst, clid, self.rules)
            else:
                duals = []

            # enumerating explanations for the instance
            new_rules, isreduce = self.explain(self.optns.smallest, clid, duals)
            # new_rules = self.explain(self.optns.smallest, [])

            # updating all the explainers
            self.update_rules(new_rules, isreduce, clid)
            #for expl in new_rules:
            #    self.instgen.add_clause([-l for l in expl])

            ctime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - ctime
            self.ctime += ctime

            if self.optns.reduce_lit and self.optns.reduce_lit.lower().startswith('a'):
                new_rules_ = []
                ltime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                        resource.getrusage(resource.RUSAGE_SELF).ru_utime
                for expl in new_rules:
                    rule = self.reduce_lit(expl, clid, self.optns.reduce_lit_appr)[0]
                    if len(rule) > 0:
                        plen = len(all_rules)
                        all_rules.add(tuple(sorted(rule)))
                        clen = len(all_rules)
                        if plen != clen:
                           new_rules_.append(rule)
                    else:
                        break
                new_rules = new_rules_
                ltime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                        resource.getrusage(resource.RUSAGE_SELF).ru_utime - ltime
                self.ltime += ltime
            # updating the dictionary of all rules
            self.rules[clid] += new_rules

    def compile(self):
        """
            Do the compilation.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        self.ctime = 0
        self.ltime = 0
        self.rtime = 0

        def signal_handler():
            raise Exception('\nCompilation time out')

        if self.optns.timeout:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(self.optns.timeout - 60)

        try:
            self.compile_()
        except:
            print('Compilation time out!')

        if self.optns.reduce_rule:
            self.rtime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                         resource.getrusage(resource.RUSAGE_SELF).ru_utime
            self.rules = self.reduce_rule(self.rules)

            self.rtime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                         resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.rtime

        if self.verbose:
            print('\ncompiled representation:')

            for clid, expls in self.rules.items():
                if self.optns.dsformat:
                    for expl in expls:
                        preamble = []
                        for i in expl:
                            if ' >= ' in self.names[i]:
                                preamble.append('\'{0}: 0\''.format(self.names[i]).replace(' >= ', '<'))
                            else:
                                preamble.append('\'{0}: 1\''.format(self.names[i].replace(' < ', '<')))
                        label = f'\'class: {self.xgb.target_name[clid]}\''
                        print('c2 cover: {0} => {1}'.format(', '.join(preamble), label))
                else:
                    for expl in expls:
                        preamble = [self.names[i] for i in expl]
                        label = self.xgb.target_name[clid]
                        print('  "IF {0} THEN {1}"'.format(' AND '.join(preamble), label))

            print('  nof rules:', sum([len(expls) for expls in self.rules.values()]))

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        if self.verbose:
            #print('  calls:', self.calls)
            print('  compile calls:', self.calls)
            print('  compile rtime: {0:.2f}'.format(self.ctime))
            if self.optns.reduce_lit:
                print('  lit reduction rtime: {0:.2f}'.format(self.ltime))
            if self.optns.reduce_rule:
                print('  rule reduction rtime: {0:.2f}'.format(self.rtime))
            print('  rtime: {0:.2f}'.format(self.time))

        return self.rules

    def get_duals(self, inst, clid, rules):
        """
            Extract the dual explanations using the literals of the instance.
        """

        duals = []

        # here is the list of literals of the instance
        # in the form of "feature" -> "list of literals"
        flits = collections.defaultdict(lambda: [])
        for l in inst:
            flits[self.v2feat[abs(l)]].append(l)

        # extracing all duals
        for lb in range(self.nofcl):
            if lb == clid:
                continue

            for expl in rules[lb]:
                dual = []
                for l in expl:
                    dual += flits[self.v2feat[abs(l)]]

                # there may be duplicate literals, hence filtering them out
                dual = sorted(set(dual))

                # add this set to hit only if it is not the complete instance
                if len(dual) < len(inst):
                    duals.append(dual)

        return duals

    def explain(self, smallest, clid, duals=[]):
        """
            Hypotheses minimization.
        """

        if self.optns.encode != 'mxe':
            # dummy call with the full instance to detect all the necessary cores
            self.oracle.get_coex(self.hfull, full_instance=True, early_stop=True)
            self.calls += 1

        if self.optns.knowledge:
            self.filter_knowledge()

        # calling the actual explanation procedure
        if not smallest and self.optns.xnum == 1:
            # extracting a single MUS
            expls = [self.extract_mus(reduce_=self.optns.reduce)]
        else:
            # MHS-based exhaustive explanation enumeration
            expls = self.mhs_mus_enumeration(smallest=smallest, duals=duals)

        if self.optns.reduce_lit and self.optns.reduce_lit.lower().startswith('b'):
            expls_ = []
            isreduce = []
            for expl in expls:
                xp, reduce = self.reduce_lit(expl, clid, self.optns.reduce_lit_appr)
                expls_.append(xp)
                isreduce.append(reduce)
            expls = expls_
            #expls = [self.reduce_lit(expl, clid, self.optns.reduce_lit_appr) for expl in expls]
        else:
            isreduce = [False for expl in expls]

        if self.verbose > 1:
            for expl in expls:
                preamble = [self.names[i] for i in expl]
                label = self.xgb.target_name[self.out_id]
                print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(preamble), label))
                print('  # hypos left:', len(expl))

        return expls, isreduce

    def mhs_mus_enumeration(self, smallest=False, duals=[]):
        """
            Enumerate subset- and cardinality-minimal explanations.
        """

        # result
        expls = []

        # we are boostrapping our hitting set enumerator with this set
        if not duals:
            duals = [self.hfull]

        # initialising connections oracle
        connor = SATSolver(name=self.optns.solver)

        with Hitman(bootstrap_with=duals, htype='sorted' if smallest else 'lbx') as hitman:
            # adding negated literals to the mapping
            for l in self.hfull:
                hitman.idpool.obj2id[-l] = -hitman.idpool.obj2id[l]

            # adding all the interval connections
            for c in self.conns:
                cc = list(map(lambda lit: hitman.idpool.id(lit), c))
                hitman.oracle.add_clause(cc)
                connor.add_clause(c)

            # computing unit-size MCSes
            if self.optns.unit_mcs:
                for i in range(len(self.hfull)):
                    self.calls += 1
                    if self.oracle.get_coex(self.hfull[:i] + self.hfull[(i + 1):], early_stop=True):
                        hitman.hit([self.hfull[i]])

            # if self.verbose > 2:
            #     print('dual:', self.duals)

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                # applying candidate reduction based on interval dependencies
                if hset:
                    hset = self.reduce_xp(hset, connor, axp=True)

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

                        st, props = connor.propagate(assumptions=[lit])
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
                else:
                    if self.verbose > 2:
                        print('expl:', hset)

                    expls.append(hset)

                    if len(expls) != self.optns.xnum:
                        hitman.block(hset)
                    else:
                        break

        connor.delete()
        return expls

    def reduce_xp(self, expl, connor, axp=True):
        """
            Get rid of redundant literals in an explanation. This is based
            on the trivial dependencies between feature intervals.
        """

        expl = sorted(expl, key=lambda v: self.order[v], reverse=not axp)

        i = 0
        while i < len(expl):
            lit = expl[i]
            st, props = connor.propagate(assumptions=[lit])

            expl = expl[:i + 1] + sorted(list(set(expl[i + 1:]).difference(set(props))),
                    key=lambda v: self.order[v], reverse=not axp)

            i += 1

        return expl

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
                    to_test.pop(0)
                    #to_test.remove(a)
                    self.calls += 1
                    # actual binary hypotheses to test
                    # print('testing', to_test, self._cats2hypos(to_test), core)
                    # print(self.v2cat)
                    if not self.oracle.get_coex(to_test, early_stop=True):
                        # print('cost', self.oracle.oracles[1].cost)
                        return False
                    # print('cost', self.oracle.oracles[1].cost)
                    to_test.append(a)
                    #to_test.add(a)
                    return True
                else:
                    return True


            to_test = core[:]
            #to_test = set(core)

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
            self.calls += 1

            # getting the core
            # core = sorted(self.oracle.get_reason(), key=lambda l: self.order[l])
            if self.optns.fsort:
                core = sorted(self.hfull, key=lambda l: self.pred2feats[self.out_id][l])
            else:
                core = sorted(self.hfull, key=lambda l: self.order[l])
        else:
            core = start_from

        if self.verbose > 2:
            print('core:', core)

        if reduce_ == 'qxp':
            expl = _do_quickxplain(core)
        else:  # by default, linear MUS extraction is used
            expl = _do_linear(core)

        expl = sorted(expl, key=lambda l: self.order[l])

        return expl

    def prepare_knowledge(self):
        #print('self.lvars:')
        #print(self.lvars)
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

        # vpool = IDPool(start_from=self.formulas[clid].nv + 1)
        #self.vpools = {clid: IDPool(start_from=self.oracle.formulas[clid].nv + 1)
        #               for clid in self.oracle.formulas if clid != self.out_id}
        # IDPool(start_from=2) OR IDPool(start_from=max(nv)+1)

        #self.c2knowledge = {clid: [] for clid in self.oracle.formulas if clid != self.out_id}
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
                    try:
                        cat_names = list(map(lambda l: l.replace(' ', ''), cat_names))
                        vid = cat_names.index(value)
                    except:
                        return [None]
                # print('value_id:', vid)
            else:
                try:
                    thresholds = self.intvs['f{0}'.format(fid)]
                except:
                    return [None]
                try:
                    vid = thresholds.index(float(feature.rsplit(maxsplit=1)[-1]))
                    if '>=' in feature:
                        vid = len(thresholds) - 1
                except:
                    return [None]
                #if '>=' in feature:
                #    vid = len(thresholds) - 1
                #else:
                #    vid = thresholds.index(float(feature.rsplit(maxsplit=1)[-1]))
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
                        # for clid in self.oracle.formulas:
                        #    if clid != self.out_id:
                        #        var = -self.vpools[clid].id('f{0}_{1}'.format(fid, vid)) * s
                        #        vars.append(var)
                        vars = [None]
                else:
                    try:
                        var = -self.lvars['f{0}_0'.format(fid)][vid] * s
                    except:
                        # for clid in self.oracle.formulas:
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
                # print('lname:', lname)
                # print('label_value:', label_value)

                labeL_vars = get_vars(lname, fid, is_cat, label_value, sign=True)
                if labeL_vars[0] is None:
                    continue
                # print('vars:', vars)
                # print()

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

                        # print('feature:', feature)
                        # print('value:', value)
                        # print('sign:', sign)
                        vars = get_vars(feature, fid_, is_cat_, value, sign=sign)
                        # print('vars:', vars)
                        # print()
                        imp_vars.append(vars)
                        if vars[0] is None:
                            break

                    if imp_vars[-1][0] is None:
                        continue

                    clause = [] #{clid: [] for clid in self.c2knowledge}
                    for vars in imp_vars:
                        #for i, clid in enumerate(self.c2knowledge):
                        #    if len(vars) == 1:
                        clause.append(-vars[0])
                        #    else:
                        #        clauses[clid].append(-vars[i])

                    #for i, clid in enumerate(self.c2knowledge):
                        #if len(labeL_vars) == 1:
                    clause.append(labeL_vars[0])
                        #else:
                        #    clauses[clid].append(labeL_vars[i])

                    #for clid in self.c2knowledge:
                    self.knowledge.append(clause)

    def filter_knowledge(self):
        """

        # Propagation

        """

        encoded_knowledge = []
        t2cid = {}
        #for clid in self.c2knowledge:
        #top = self.vpools[clid].top
        top = max([abs(var) for lvars in self.lvars.values() for var in lvars])
        for i, cl in enumerate(self.knowledge):
            top += 1
            encoded_knowledge.append(cl + [-top])
            t2cid[top] = i
        #print(encoded_knowledge)
        #print(t2cid)
        #exit()
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

    def prepare_train(self):
        samples = self.xgb.X_train
        Y_train = set(self.xgb.Y_train)
        #self.lvar_max = max([lvar for lvars in self.lvars.values() for lvar in lvars], key=lambda l: abs(l))
        y2samples = {int(y): [] for y in Y_train}
        pred2samples = {int(y): [] for y in Y_train}
        pred2samples_ = {int(y): [] for y in Y_train}
        hfulls = []
        connses = []
        orders = []
        v2cats = []
        out_ids = []
        if self.optns.fsort:
            self.pred2feats = collections.defaultdict(lambda : collections.defaultdict(lambda : 0))

        # SAT-based predictor
        instgen = SATSolver(name=self.optns.solver)
        for clid in range(self.nofcl):
            instgen.append_formula(self.formula[clid].formula)
        for samp_id, sample in enumerate(samples):
            hypos = self.xgb.mxe.get_literals(sample)
            assert instgen.solve(assumptions=hypos) == True
            model = instgen.get_model()

            # computing all the class scores
            scores = {}
            for clid in range(self.nofcl):
                # computing the value for the current class label
                scores[clid] = 0

                for lit, wght in self.xgb.mxe.enc[clid].leaves:
                    if model[abs(lit) - 1] > 0:
                        scores[clid] += wght

            # here is the full list of hypotheses over the language of the model
            hfull = []
            conns = []

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
                        conns.append([stack[j], -stack[j + 1]])

                    # adding negative literals to hypotheses
                    hfull += [stack.pop() for v in range(len(stack))]

                    # second, the positive part
                    for j in range(i, len(self.lvars[feat]) - 1):
                        hfull.append(self.lvars[feat][j])

                    # collecting positive connections
                    for j in range(i, len(self.lvars[feat]) - 2):
                        conns.append([-self.lvars[feat][j], self.lvars[feat][j + 1]])

                else:
                    # there is a single Boolean variable used for this feature
                    hfull.append(model[abs(self.lvars[feat][0]) - 1])

            # feature literal order
            order = {l: i for i, l in enumerate(hfull)}

            hfull = sorted(set(hfull))

            # variable to the category in use; this differs from
            # v2feat as here we may not have all the features here
            v2cat = {}
            for i, v in enumerate(hfull):
                v2cat[v] = i

            # returning the class corresponding to the max score
            out_id = max(list(scores.items()), key=lambda t: t[1])[0]

            y = int(self.xgb.Y_train[samp_id])
            #hfulls[y].append(hfull + [self.lvar_max + y + 1])
            wght_ = self.xgb.wghts[tuple(list(sample) + [self.xgb.Y_train[samp_id]])]

            y2samples[y].append(samp_id)
            pred2samples[out_id].append(samp_id)
            pred2samples_[out_id].append(samp_id)
            hfulls.append([set(hfull), wght_])
            connses.append(conns)
            orders.append(order)
            v2cats.append(v2cat)
            out_ids.append(out_id)

            if self.optns.fsort:
                for h in hfull:
                    self.pred2feats[out_id][h] += 1

        train_info = {'hfulls': hfulls}

        if self.optns.reduce_lit:
            train_info['y2samples'] = y2samples

        if self.optns.clocal:
            train_info['pred2samples'] = pred2samples
            train_info['connses'] = connses
            train_info['orders'] = orders
            train_info['v2cats'] = v2cats

        if self.optns.reduce_rule:
            train_info['pred2samples_'] = pred2samples_

        return train_info

    def reduce_lit(self, expl, clid, reduce_='lin'):

        def nof_inconsist(samples, expl):
            inconsist = 0
            for sample, wght in samples:
                for h in expl:
                    if h not in sample:
                        break
                else:
                    inconsist += wght
            return inconsist

        def _do_linear(samples, core, max_inconsist):
            """
                Do linear search.
            """

            def _assump_needed(samples, a, max_inconsist):
                if len(to_test) > 1:
                    to_test.remove(a)
                    #self.calls += 1
                    if nof_inconsist(samples, to_test) <= max_inconsist:
                        return False
                    to_test.add(a)
                    return True
                else:
                    return True

            to_test = set(core)
            return list(filter(lambda a: _assump_needed(samples, a, max_inconsist), core))

        def _do_quickxplain(samples, core, max_inconsist):
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
                    if to_test and (nof_inconsist(samples, to_test) <= max_inconsist):
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

        def _do_maxsat(expl, clid):
            enc = WCNF()
            expl_len = len(expl)

            m = len(expl)+1
            y2m = {}
            wght_ = 0
            #max_wght = max(self.xgb.wghts.values())

            y_samps_ids = [[y, samp_ids] for y, samp_ids in self.train_info['y2samples'].items()
                           if y != clid]

            if len(y_samps_ids) == 0:
                return []

            # define misclassification and add soft clauses to minimise misclassification
            for y, samp_ids in y_samps_ids:
                y2m[y] = []
                for samp_id in samp_ids:
                    sample, wght = self.train_info['hfulls'][samp_id]
                    disj = []
                    for i, h in enumerate(expl, start=1):
                        if h not in sample:
                            disj.append(i)

                    if len(disj) < expl_len:
                        if self.optns.lam:
                            enc.append([-m], weight=wght)
                        else:
                            enc.append([-m], weight=wght * (expl_len + 1))
                        wght_ += wght
                        if len(disj) == 0:
                            # this sample must be misclassified
                            enc.append([m])
                        else:
                            # this sample is possible to be miclassified
                            # -m <-> disj0 \/ disj1 ..
                            enc.append([m] + [lit for lit in disj])
                            enc.extend([[-lit, -m] for lit in disj])

                        y2m[y].append([m, samp_id])
                        m += 1

            if wght_ == 0:
                return [expl[-1]]
            m -= 1

            if self.optns.lam:
                if self.lambda_ is None:
                    if self.optns.approx:
                        self.lambda_ = int(math.ceil(sum(self.xgb.wghts.values()) * float(self.optns.lam)))
                    else:
                        self.lambda_ = sum(self.xgb.wghts.values()) * decimal.Decimal(self.optns.lam)
                #else:
                #if self.optns.approx:
                #    self.lambda_ = int(math.ceil(wght_ * float(self.optns.lam)))
                #else:
                #    self.lambda_ = wght_ * decimal.Decimal(self.optns.lam)
            else:
                self.lambda_ = 0

            #self.train_info['y2samples'][y] == [hfull, wght]

            # add soft clauses to reduce literals used
            disj = []
            for i in range(1, expl_len + 1):
                disj.append(i)
                if self.lambda_ > 0:
                    enc.append([-i], weight=self.lambda_)
                else:
                    enc.append([-i], weight=1)

            # at least one literals is used
            #if self.optns.reduce_lit.startswith('b'):
            enc.append(disj)

            stratification = len(enc.wght) != sum(enc.wght)

            Solver = RC2Stratified if stratification else RC2
            with Solver(enc, solver=self.optns.solver,
                adapt=self.optns.am1, exhaust=self.optns.exhaust,
                minz=self.optns.minz, trim=self.optns.trim) as rc2:
                model = rc2.compute()
            if self.lambda_ > 0:
                for y in y2m:
                    missamp_ids = set()
                    for m, samp_id in y2m[y]:
                        if model[m-1] > 0: # samp_id is misclassified
                            missamp_ids.add(samp_id)

                    if len(missamp_ids) > 0:
                        if len(self.train_info['y2samples'][y]) == len(missamp_ids):
                            del self.train_info['y2samples'][y]
                        else:
                            kept = set(self.train_info['y2samples'][y]).difference(missamp_ids)
                            self.train_info['y2samples'][y] = sorted(kept)

            return [expl[i] for i in range(expl_len) if model[i] > 0]

        if reduce_ == 'maxsat':
            expl_ = _do_maxsat(expl, clid)
        else:
            samples = [self.train_info['hfulls'][samp_id]
                       for y, samp_ids in self.train_info['y2samples'].items()
                       if y != clid for samp_id in samp_ids]
            max_inconsist = nof_inconsist(samples, expl)
            if reduce_ == 'qxp':
                expl_ = _do_quickxplain(samples, expl, max_inconsist)
            else:  # by default, linear extraction is used
                expl_ = _do_linear(samples, expl, max_inconsist)

        return [expl_, len(expl_) < len(expl)]

    def uncover(self):
        if not self.optns.clocal:
            return self.instgen.solve()
        else:
            return len(self.train_info['pred2samples']) > 0

    def update_rules(self, new_rules, isreduce, clid):
        # updating all the explainers
        if not self.optns.clocal:
            for expl in new_rules:
                self.instgen.add_clause([-l for l in expl])
        else:
            ## update uncovered training instances
            #for expl, reduce in zip(new_rules, isreduce):
            #    clids = list(self.train_info['pred2samples'].keys()) if reduce else [clid]
            #    idpool = IDPool
            #    enc = WCNF()
            #    expl_cl = [idpool.id(h) for h in expl]
            #    samp_vars = []
            #    samp_var2hfull = {}
            #    for clid_ in clids:
            #        for samp_id in self.train_info['pred2samples'][clid_]:
            #            samp_var = idpool.id(samp_id)
            #            samp_vars.append(samp_var)
            #            samp_var2hfull[samp_var] = (clid_, samp_id)

            #            sample = self.train_info['hfulls'][samp_id][0]
            #            samp_cl = [idpool.id(s) for s in sample]
            #            right_hand = expl_cl + samp_cl

            #            enc.extend([[-samp_var, rh] for rh in right_hand])
            #            enc.append([samp_var] + [-rh for rh in right_hand])
            #    enc.extend([[-samp_var] for samp_var in samp_vars])
            #    oracle = SATSolver(name=self.optns.solver, bootstrap_with=enc)
            #    oracle.solve()

            # update uncovered training instances
            for expl, reduce in zip(new_rules, isreduce):
                clids = list(self.train_info['pred2samples'].keys()) if reduce else [clid]
                for clid_ in clids:
                    uncover_train = []
                    for samp_id in self.train_info['pred2samples'][clid_]:
                        sample = self.train_info['hfulls'][samp_id][0]
                        for h in expl:
                            if h not in sample:
                                uncover_train.append(samp_id)
                                break

                    if len(uncover_train) == 0:
                        del self.train_info['pred2samples'][clid_]

                    elif len(uncover_train) < len(self.train_info['pred2samples'][clid_]):
                        self.train_info['pred2samples'][clid_] = uncover_train

                if len(self.train_info['pred2samples']) == 0:
                    break

            # update feature frequencies
            if self.optns.fqupdate and len(self.train_info['pred2samples']) > 0:
                self.pred2feats = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
                hfulls = self.train_info['hfulls']
                for out_id in self.train_info['pred2samples']:
                    for samp_id in self.train_info['pred2samples'][out_id]:
                        hfull = hfulls[samp_id][0]
                        for h in hfull:
                            self.pred2feats[out_id][h] += 1

    def reduce_rule(self, rules):
        """
                    Cover samples for all labels using Gurobi.
                """

        assert gurobi_present, 'Gurobi is unavailable'

        if self.optns.verb:
            print('c2 (using gurobi)')

        covers = {clid: [] for clid in range(self.nofcl)}

        for clid in rules:
            if len(rules[clid]) == 0:
                continue
            # a hack to disable license file logging
            stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            # initializing the solver
            hitman = gurobi.Model()

            # restoring sys.stdout
            sys.stdout = stdout

            # turning logger off
            hitman.Params.OutputFlag = 0
            hitman.Params.LogToConsole = 0

            # variables
            vnames = []
            rvars = []
            expls = list(map(lambda l: set(l), rules[clid]))
            print('expls:')
            for i, rule in enumerate(expls):
                print(rule)
                vnames.append('r_{0}'.format(i + 1))
                if self.optns.weighted:
                    rvars.append(hitman.addVar(vtype=gurobi.GRB.BINARY,
                                                name=vnames[i], obj=len(rule) + 1))
                else:
                    rvars.append(hitman.addVar(vtype=gurobi.GRB.BINARY,
                                                name=vnames[i], obj=1))
            print()
            # hard constraints
            samples = set(self.train_info['pred2samples_'][clid])
            print('samples:', sorted(samples))
            print()
            if clid in self.train_info['pred2samples']:
                samples = samples.difference(set(self.train_info['pred2samples'][clid]))
                print('fired:', set(self.train_info['pred2samples'][clid]))
                print()

            print('samples:', sorted(samples))
            print()
            for samp_id in samples:
                to_hit = []
                hfull = self.train_info['hfulls'][samp_id][0]
                for rid, rule in enumerate(expls):
                    if rule.issubset(hfull):
                        to_hit.append(rvars[rid])

                if len(to_hit) == 0:
                    print('samp_id not fired')
                    print()
                hitman.addConstr(lhs=gurobi.quicksum(1 * v for v in to_hit),
                                 sense=gurobi.GRB.GREATER_EQUAL,
                                 rhs=1,
                                 name='sid{0}'.format(samp_id))

            # and the cover is...
            hitman.optimize()
            expls = rules[clid]
            for rid, rule in enumerate(rvars):
                if int(rule.X) > 0:
                    covers[clid].append(expls[rid])

            #self.cost += int(hitman.objVal)

            #if self.optns.weighted:
            #    self.cost -= len(self.covers[clid])

            # cleaning up
            for samp_id in samples:
                c = hitman.getConstrByName('sid{0}'.format(samp_id))
                hitman.remove(c)

            hitman.remove(hitman.getVars())

        return covers

