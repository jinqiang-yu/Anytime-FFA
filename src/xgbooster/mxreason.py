#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## mxreason.py
##
##

# imported modules:
#==============================================================================
from __future__ import print_function
import collections
import copy
import decimal
from functools import reduce
from .erc2 import ERC2
import math
from pysat.examples.rc2 import RC2Stratified
from pysat.formula import CNF, WCNF, IDPool
import subprocess
import sys
import tempfile


# a named tuple for class encodings
#==============================================================================
ClassEnc = collections.namedtuple('ClassEnc', ['formula', 'leaves', 'trees'])


#
#==============================================================================
class MXReasoner:
    """
        MaxSAT-based explanation oracle. It can be called to decide whether a
        given set of feature values forbids any potential misclassifications,
        or there is a counterexample showing that the set of feature values is
        not an explanation for the prediction.
    """

    def __init__(self, encoding, target, solver='g3', oracle='int',
            am1=False, exhaust=False, minz=False, trim=0, knowledge=[]):
        """
            Magic initialiser.
        """

        self.oracles = {}   # MaxSAT solvers
        self.target = None  # index of the target class
        self.reason = None  # reason / unsatisfiable core
        self.values = collections.defaultdict(lambda: [])  # values for all the classes
        self.scores = {}  # class scores
        self.scbump = 0   # score bump if negative scores are present
        self.formulas = {}
        self.ortype = oracle

        # MaxSAT-oracle options
        self.am1 = am1
        self.exhaust = exhaust
        self.minz = minz
        self.trim = trim
        self.solver = solver  # keeping for alien solvers

        self.vpool = None
        # doing actual initialisation
        self.init(encoding, target, solver, knowledge)

        self.vpool = IDPool(start_from=self.vpool + 1)

    def __del__(self):
        """
            Magic destructor.
        """

        self.delete()

    def __enter__(self):
        """
            'with' constructor.
        """

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            'with' destructor.
        """

        self.delete()

    def init(self, encoding, target, solver, knowledge):
        """
            Actual constructor.
        """

        # computing score bump in case there are leaves with negative weights
        # minw = min([min(map(lambda v: v[1], encoding[clid].leaves)) for clid in encoding])
        # self.scbump = 1 - minw if minw <= 0 else 0

        # saving the target
        self.target = target
        # copying class values
        for clid in encoding:
            for lit, wght in encoding[clid].leaves:
                self.values[clid].append(tuple([lit, wght]))

        # creating the formulas and oracles
        for clid in encoding:
            if clid == self.target:
                continue

            # adding hard clauses
            self.formulas[clid] = WCNF()
            for cl in encoding[clid].formula:
                self.formulas[clid].append(cl)

            # adding background knowledge
            for cl in knowledge:
                self.formulas[clid].append(cl)

            if len(encoding) > 2:
                for cl in encoding[self.target].formula:
                    self.formulas[clid].append(cl)

            # adding soft clauses and recording all the leaf values
            self.init_soft(encoding, clid)

            if self.ortype == 'int':
                # a new MaxSAT solver
                # self.oracles[clid] = ERC2(self.formulas[clid], solver=solver, adapt=True,
                #         exhaust=True, minz=True, verbose=3)
                self.oracles[clid] = ERC2(self.formulas[clid], solver=solver,
                        adapt=self.am1, blo='cluster', exhaust=self.exhaust,
                        minz=self.minz, verbose=0)
                # self.oracles[clid] = ERC2(formula, exhaust=True, minz=True)
                # self.oracles[clid] = ERC2(formula)

                # ideally, we should treat each tree separately
                # but for now let's do it the simple way assuming
                # that there is a single base case (with no feature-values given)
                # model = self.oracles[clid].compute()
                # assert self.get_winner(model, clid) != self.target, 'No misclassification possible!'

    def init_soft(self, encoding, clid):
        """
            Processing the leaves and creating the set of soft clauses.
        """

        # new vpool for the leaves, and total cost
        vpool = IDPool(start_from=self.formulas[clid].nv + 1)

        # all leaves to be used in the formula, am1 constraints and cost
        wghts, atmosts, cost = collections.defaultdict(lambda: 0), [], 0

        for label in (clid, self.target):
            if label != self.target:
                coeff = 1
            else:  # this is the target class
                if len(encoding) > 2:
                    coeff = -1
                else:
                    # we don't encoding the target class if there are
                    # only two classes - it duplicates the other class
                    continue

            # here we are going to automatically detect am1 constraints
            for tree in encoding[label].trees:
                am1 = []
                for i in range(tree[0], tree[1]):
                    lit, wght = encoding[label].leaves[i]

                    # all leaves of each tree comprise an AtMost1 constraint
                    am1.append(lit)

                    # updating literal's final weight
                    wghts[lit] += coeff * wght

                atmosts.append(am1)

        # filtering out those with zero-weights
        wghts = dict(filter(lambda p: p[1] != 0, wghts.items()))

        # processing the opposite literals, if any
        i, lits = 0, sorted(wghts.keys(), key=lambda l: 2 * abs(l) + (0 if l > 0 else 1))

        while i < len(lits) - 1:
            if lits[i] == -lits[i + 1]:
                l1, l2 = lits[i], lits[i + 1]
                minw = min(wghts[l1], wghts[l2], key=lambda w: abs(w))

                # updating the weights
                wghts[l1] -= minw
                wghts[l2] -= minw

                # updating the cost if there is a conflict between l and -l
                if wghts[l1] * wghts[l2] > 0:
                    cost += abs(minw)

                i += 2
            else:
                i += 1

        # flipping literals with negative weights
        lits = list(wghts.keys())
        for l in lits:
            if wghts[l] < 0:
                cost += -wghts[l]
                wghts[-l] = -wghts[l]
                del wghts[l]

        # maximum value of the objective function
        self.formulas[clid].vmax = sum(wghts.values())

        # processing all AtMost1 constraints
        atmosts = set([tuple([l for l in am1 if l in wghts and wghts[l] != 0]) for am1 in atmosts])
        for am1 in sorted(atmosts, key=lambda am1: len(am1), reverse=True):
            if len(am1) < 2:
                continue
            cost += self.process_am1(self.formulas[clid], am1, wghts, vpool)

        # here is the start cost
        self.formulas[clid].cost = cost

        # adding remaining leaves with non-zero weights as soft clauses
        for lit, wght in wghts.items():
            if wght != 0:
                self.formulas[clid].append([ lit], weight=wght)

        #a = 0
        #for cl in self.formulas[clid].hard:
        #    m = abs(max(cl, key=lambda l: abs(l)))
        #    a = max(a, m)

        #for cl in self.formulas[clid].soft:
        #    m = abs(max(cl, key=lambda l: abs(l)))
        #    a = max(a, m)
        #print('nof var:', a)
        #print('nof hard:', len(self.formulas[clid].hard))
        #print('nof soft:', len(self.formulas[clid].soft))
        #print()
        # self.formulas[clid].to_file('formula{0}.wcnf'.format(clid))
        # exit(1)
        self.vpool = max(self.vpool, vpool.top) if self.vpool is not None \
            else vpool.top

    def add_clause(self, clause):
        if self.ortype == 'int':
            for clid in self.oracles:
                if clid == self.target:
                    continue
                self.oracles[clid].add_clause(clause)
        else:
            for clid in self.formulas:
                if clid == self.target:
                    continue
                self.formulas[clid].append(clause)

    def process_am1(self, formula, am1, wghts, vpool):
        """
            Detect AM1 constraints between the leaves of one tree and add the
            corresponding soft clauses to the formula.
        """

        cost = 0

        # filtering out zero-weight literals
        am1 = [l for l in am1 if wghts[l] != 0]

        # processing the literals until there is only one literal left
        while len(am1) > 1:
            minw = min(map(lambda l: wghts[l], am1))
            cost += minw * (len(am1) - 1)

            lset = frozenset(am1)
            if lset not in vpool.obj2id:
                selv = vpool.id(lset)

                # adding a new hard clause
                formula.append(am1 + [-selv])
            else:
                selv = vpool.id(lset)

            # adding a new soft clause
            formula.append([selv], weight=minw)
            # print(am1, minw)

            # filtering out non-zero weight literals
            i = 0
            while i < len(am1):
                wghts[am1[i]] -= minw

                if wghts[am1[i]] == 0:
                    am1[i] = am1[len(am1) - 1]
                    am1.pop()
                else:
                    i += 1

        return cost

    def delete(self):
        """
            Actual destructor.
        """

        if self.oracles:
            for oracle in self.oracles.values():
                if oracle:
                    oracle.delete()

            self.oracles = {}
            self.target = None
            self.values = None
            self.reason = None
            self.scores = {}
            self.formulas = {}

    def get_coex(self, feats, full_instance=False, early_stop=False):
        """

            A call to the oracle to obtain a counterexample to a given set of
            feature values (may be a complete instance or a subset of its
            feature values). If such a counterexample exists, it is returned.
            Otherwise, the method returns None.

            Note that if None is returned, the given set of feature values is
            an abductive explanation for the prediction (not necessarily a
            minimal one).
        """

        # resetting the scores
        self.scores = {clid: 0 for clid in self.oracles}

        # updating the reason
        self.reason = set()

        if self.ortype == 'int':
            # using internal MaxSAT solver incrementally
            for clid in self.oracles:
                if clid == self.target:
                    continue

                model = self.oracles[clid].compute(feats, full_instance, early_stop)
                assert model or (early_stop and self.oracles[clid].cost > self.oracles[clid].slack), \
                        'Something is wrong, there is no MaxSAT model'

                # if misclassification, return the model
                # note that this model is not guaranteed
                # to represent the predicted class!
                if model and self.get_winner(model, clid) != self.target:
                    return model

                reason = self.oracles[clid].get_reason()

                if reason is not None:
                    # otherwise, proceed to another clid
                    self.reason = self.reason.union(set(reason))

                # print('done')

            if not self.reason:
                self.reason = None

            # if no counterexample exists, return None
        else:
            # here we start an external MaxSAT solver every time
            for clid in self.formulas:
                if clid == self.target:
                    continue

                if self.ortype == 'ext':  # external RC2
                    with RC2Stratified(self.formulas[clid], solver='g3',
                            adapt=self.am1, blo='div', exhaust=self.exhaust,
                            incr=False, minz=self.minz, nohard=False,
                            trim=self.trim, verbose=0) as rc2:
                        # rc2.hard = False

                        # adding more hard clauses on top
                        for lit in feats:
                            rc2.add_clause([lit])

                        model = rc2.compute()
                else:  # expecting 'alien' here
                    # dumping the formula into a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wcnf') as fp:
                        sz = len(self.formulas[clid].hard)
                        self.formulas[clid].hard += [[l] for l in feats]
                        self.formulas[clid].to_file(fp.name)
                        self.formulas[clid].hard = self.formulas[clid].hard[:sz]
                        fp.flush()

                        outp = subprocess.check_output(self.solver.split() + [fp.name], shell=False)
                        outp = outp.decode(encoding='ascii').split('\n')

                    # going backwards in the log and extracting the model
                    for line in range(len(outp) - 1, -1, -1):
                        line = outp[line]
                        if line.startswith('v '):
                            model = [int(l) for l in line[2:].split()]

                assert model, 'Something is wrong, there is no MaxSAT model'

                # if misclassification, return the model
                # note that this model is not guaranteed
                # to represent the predicted class!
                if self.get_winner(model, clid) != self.target:
                    return model

            # otherwise, proceed to another clid
            self.reason = set(feats)

    def get_winner(self, model, clid):
        """
            Check the values for each class and extract the prediction.
        """

        for label in (self.target, clid):
            # computing the value for the current class label
            self.scores[label] = 0

            for lit, wght in self.values[label]:
                if model[abs(lit) - 1] > 0:
                    self.scores[label] += wght

        if self.scores[clid] >= self.scores[self.target]:
            return clid

        return self.target

    def get_scores(self):
        """
            Get all the actual scores for the classes computed with the previous call.
        """

        # this makes sense only for complete instances
        assert all([score != 0 for score in self.scores.values()])

        return [self.scores[clid] for clid in range(len(self.scores))]

    def get_reason(self, v2fmap=None):
        """
            Reports the last reason (analogous to unsatisfiable core in SAT).
            If the extra parameter is present, it acts as a mapping from
            variables to original categorical features, to be used a the
            reason.
        """

        assert self.reason, 'There no reason to return!'

        if v2fmap:
            return sorted(set(v2fmap[v] for v in self.reason))
        else:
            return self.reason

def load_classifier(filename):
    """
        Load a formula from a file name.
    """

    formula = CNF(from_file=filename)

    # empty intervals for the standard encoding
    intvs, imaps, ivars = {}, {}, {}

    leaves = collections.defaultdict(lambda: [])
    clids = []

    for line in formula.comments:
        if line.startswith('c i ') and 'none' not in line:
            f, arr = line[4:].strip().split(': ', 1)
            f = f.replace('-', '_')
            intvs[f], imaps[f], ivars[f] = [], {}, []

            for i, pair in enumerate(arr.split(', ')):
                ub, symb = pair.split(' <-> ')
                ub = ub.strip('"')
                symb = symb.strip('"')

                if ub[0] != '+':
                    ub = float(ub)

                intvs[f].append(ub)
                ivars[f].append(symb)
                imaps[f][ub] = i

        elif line.startswith('c features:'):
            feats = line[11:].strip().split(', ')
        elif line.startswith('c classes:'):
            nofcl = int(line[10:].strip())
        elif line.startswith('c clid starts:'):
            clid, starts = line[15:].strip().split()
            clids.append((int(clid), int(starts)))
        elif line.startswith('c leaf:'):
            clid, lvar, wght = line[8:].strip().split()
            leaves[int(clid)].append((int(lvar), decimal.Decimal(wght)))

    enc = {}
    for i in range(len(clids)):
        clid, starts = clids[i]
        ends = len(formula.clauses) if i == len(clids) - 1 else clids[i + 1][1]

        enc[clid] = ClassEnc(
                formula=CNF(from_clauses=formula.clauses[starts:ends]),
                leaves=leaves[clid]
                )

    return enc


#
#==============================================================================
if __name__ == '__main__':
    # # hard coded example
    # formula = WCNF(from_file='zoo-simple.wcnf')

    # # translating the example to
    # # encoding, sums, and target
    # enc = {
    #         0: ClassEnc(CNF(from_clauses=formula.hard), leaves=[(5,  decimal.Decimal('1.2579')), (6,  decimal.Decimal('1.6230'))]),
    #         1: ClassEnc(CNF(from_clauses=formula.hard), leaves=[(7,  decimal.Decimal('1.2568')), (8,  decimal.Decimal('1.5968'))]),
    #         2: ClassEnc(CNF(from_clauses=formula.hard), leaves=[(9,  decimal.Decimal('1.2565')), (10, decimal.Decimal('1.5061'))]),
    #         3: ClassEnc(CNF(from_clauses=formula.hard), leaves=[(11, decimal.Decimal('1.2563')), (12, decimal.Decimal('1.4957'))])
    #         }
    # target = 0

    enc = load_classifier(sys.argv[1])
    target = int(sys.argv[2])
    inst = [int(v.strip()) for v in sys.argv[3].split(',')]

    with MXReasoner(enc, target) as x:
        # inst = [1, 2, 3, 4]
        print('testing prediction')
        assert x.get_coex(inst) is None, 'Wrong prediction is enforced by the model'
        print('init reason:', x.get_reason())

        # explanation procedure (linear search)
        i = 0
        while i < len(inst):
            print('testing', i)
            to_test = inst[:i] + inst[(i + 1):]
            print('testing', to_test)
            if x.get_coex(to_test):
                print('needed (there is a coex)')
                i += 1
            else:
                print('not needed')
                inst = to_test
                print('reason:', x.get_reason())

        print('expl:', inst)
