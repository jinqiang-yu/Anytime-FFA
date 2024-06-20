#usr/bin/env python
#-*- coding:utf-8 -*-
##
## options.py
##
##

#
#==============================================================================
from __future__ import print_function
import getopt
import math
import os
from pysat.card import EncType
import sys

#
#==============================================================================
encmap = {
    "pw": EncType.pairwise,
    "seqc": EncType.seqcounter,
    "cardn": EncType.cardnetwrk,
    "sortn": EncType.sortnetwrk,
    "tot": EncType.totalizer,
    "mtot": EncType.mtotalizer,
    "kmtot": EncType.kmtotalizer,
    "native": EncType.native
}


#
#==============================================================================
class Options(object):
    """
        Class for representing command-line options.
    """

    def __init__(self, command=None):
        """
            Constructor.
        """

        # actions
        self.train = False
        self.relax = 0
        self.encode = 'none'
        self.explain = ''
        self.compile = False
        self.process = False
        self.useanchor = False
        self.uselime = False
        self.useshap = False
        self.limefeats = 5
        self.validate = False
        self.use_categorical = False
        self.preprocess_categorical = False
        self.preprocess_categorical_files = ""

        # training options
        self.accmin = 0.95
        self.n_estimators = 100
        self.num_boost_round = 10
        self.maxdepth = 3
        self.testsplit = 0.2
        self.seed = 7

        # maxsat options
        self.minz = False
        self.am1 = False
        self.exhaust = False
        self.trim = 0

        # other options
        self.files = None
        self.cardenc = 'seqc'
        self.output = 'temp'
        self.mapfile = None
        self.reduce = 'none'
        self.separator = ','
        self.smallest = False
        self.solver = 'z3'
        self.unit_mcs = False
        self.verb = 0
        self.xnum = 1
        self.xtype = 'abd'
        self.ilits = False
        self.use_duals = False
        self.dsformat = False
        self.knowledge = None

        self.reduce_lit = None
        self.reduce_lit_appr = 'maxsat' #'lin'
        self.reduce_rule = False
        self.lam = None
        self.approx = 0

        self.clocal = False
        self.fsort = False
        self.fqupdate = False
        self.weighted = False
        self.timeout = None

        # exemplar
        self.exxp = False
        self.explains = ''
        self.xid = None
        self.neg = False

        # image
        self.visual = None
        self.explain_ = 'formal'
        self.sort = 'centre'
        self.reverse = False


        self.use_mhs = False
        self.use_cld = False
        self.cut = None

        self.htype = 'lbx'
        self.hsolver = 'g3'

        self.switch = 0 # 0, 1, ..., all
        self.sliding = 1
        self.stype = 'simple' #'lgap', 'stable', 'ls'
        self.gap = 0 # 2, 1.75, 1.5, 1.25
        self.diff = 1 # 1, 2, 3

        if command:
            self.parse(command)
            if self.fqupdate:
                self.fsort = True
            if self.fsort:
                self.clocal = True

    def parse(self, command):
        """
            Parser.
        """

        self.command = command

        try:
            opts, args = getopt.getopt(command[1:],
                                    '1a:C:cd:De:EfF:g:GhIik:lL:m:Mn:N:o:pP:qr:R:s:tT:uvVwWx:X:z',
                                    ['am1', 'accmin=', 'encode=', 'cardenc=',
                                     'exhaust',  'ds-format', 'help', 'internal',
                                     'knowledge=', 'map-file=', 'use-anchor=', 'lime-feats=',
                                     'use-lime=', 'use-shap=', 'use-categorical=',
                                     'use-duals', 'preprocess-categorical',
                                     'pfiles=', 'maxdepth=', 'minimum',
                                     'nbestims=', 'output=', 'process=',
                                     'reduce=', 'rounds=', 'relax=', 'seed=',
                                     'sep=', 'solver=', 'testsplit=', 'train',
                                     'trim=', 'unit-mcs', 'validate', 'verbose',
                                     'xnum=', 'xtype=', 'explain=', 'minz',
                                     'reduce-lit=', 'reduce-lit-appr=', 'lam=', 'lambda=', 'approx=',
                                     'clocal', 'fsort', 'reduce-rule', 'fqupdate', 'weighted',
                                     'timeout=',
                                     # exemplar
                                     'exxp', 'explains=', 'xid=', 'neg',
                                     'visual=', 'explain_=', 'sort=', 'reverse',
                                     'use-mhs', 'use-cld', 'cut=', 'switch=',
                                     'htype=', 'hsolver=', 'sliding=', 'stype=',
                                     'gap=', 'diff='])
        except getopt.GetoptError as err:
            sys.stderr.write(str(err).capitalize())
            self.usage()
            sys.exit(1)

        for opt, arg in opts:
            if opt in ('-1', '--am1'):
                self.am1 = True
            elif opt in ('-a', '--accmin'):
                self.accmin = float(arg)
            elif opt == '--approx':
                self.approx = int(arg)
            elif opt in ('-c', '--use-categorical'):
                self.use_categorical = True
            elif opt in ('-C', '--cardenc'):
                self.cardenc = str(arg)
            elif opt == '--cut':
                self.cut = int(arg)
            elif opt == '--clocal':
                self.clocal = True
            elif opt in ('-d', '--maxdepth'):
                self.maxdepth = int(arg)
            elif opt == '--diff':
                self.diff = int(arg)
            elif opt in ('-D', '--use-duals'):
                self.use_duals = True
            elif opt in ('-e', '--encode'):
                self.encode = str(arg)
            elif opt in ('-E', '--exhaust'):
                self.exhaust = True
            elif opt == '--explain_':
                self.explain_ = str(arg)
            elif opt in ('-f', '--ds-format'):
                self.dsformat = True
            elif opt in ('-F', '--explains'):
                self.explains = str(arg)
            elif opt == '--fsort':
                self.fsort = True
            elif opt == '--fqupdate':
                self.fqupdate = True
            elif opt in ('-g', '--xid'):
                self.xid = int(arg)
            elif opt in ('-G', '--neg'):
                self.neg = True
            elif opt in ('-h', '--help'):
                self.usage()
                sys.exit(0)
            elif opt == '--htype':
                self.htype = str(arg)
            elif opt == '--hsolver':
                self.hsolver = str(arg)
            elif opt in ('-i', '--internal'):
                self.ilits = True
            elif opt in ('-I', '--interpret'):
                self.compile = True
            elif opt in ('-k', '--knowledge'):
                self.knowledge = str(arg)
            elif opt in ('-l', '--use-lime'):
                self.uselime = True
            elif opt in ('-L', '--lime-feats'):
                self.limefeats = 0 if arg == 'all' else int(arg)
            elif opt in ('--lam', '--lambda'):
                self.lam = float(arg)
            elif opt in ('-m', '--map-file'):
                self.mapfile = str(arg)
            elif opt in ('-M', '--minimum'):
                self.smallest = True
            elif opt in ('-n', '--nbestims'):
                self.n_estimators = int(arg)
            elif opt in ('-N', '--xnum'):
                self.xnum = str(arg)
                self.xnum = -1 if self.xnum == 'all' else int(self.xnum)
            elif opt in ('-o', '--output'):
                self.output = str(arg)
            elif opt in ('-p', '--preprocess-categorical'):
                self.preprocess_categorical = True
            elif opt in ('-P', '--process'):
                files = str(arg)
                self.process = files.split(',')
            elif opt in ('--pfiles'):
                self.preprocess_categorical_files = str(arg) #train_file, test_file(or empty, resulting file
            elif opt in ('-q', '--use-anchor'):
                self.useanchor = True
            elif opt in ('-r', '--rounds'):
                self.num_boost_round = int(arg)
            elif opt in ('-R', '--reduce'):
                self.reduce = str(arg)
            elif opt == '--reduce-lit':
                self.reduce_lit = str(arg)
            elif opt == '--reduce-rule':
                self.reduce_rule = True
            elif opt == '--reduce-lit-appr':
                self.reduce_lit_appr = str(arg)
            elif opt == '--relax':
                self.relax = int(arg)
            elif opt == '--reverse':
                self.reverse = True
            elif opt == '--sort':
                self.sort = str(arg)
            elif opt == '--seed':
                self.seed = int(arg)
            elif opt == '--sep':
                self.separator = str(arg)
            elif opt in ('-s', '--solver'):
                self.solver = str(arg)
            elif opt == '--switch':
                self.switch = str(arg)
                self.switch = -1 if self.switch == 'all' else int(self.switch)
            elif opt == '--sliding':
                self.sliding = int(arg)
            elif opt == '--stype':
                self.stype = str(arg)
            elif opt == '--gap':
                self.gap = float(arg)
            elif opt == '--testsplit':
                self.testsplit = float(arg)
            elif opt in ('-t', '--train'):
                self.train = True
            elif opt in ('-T', '--trim'):
                self.trim = int(arg)
            elif opt == '--timeout':
                self.timeout = int(arg)
            elif opt in ('-u', '--unit-mcs'):
                self.unit_mcs = True
            elif opt in ('--use-mhs'):
                self.use_mhs = True
            elif opt in ('--use-cld'):
                self.use_cld = True
            elif opt in ('-V', '--validate'):
                self.validate = True
            elif opt in ('-v', '--verbose'):
                self.verb += 1
            elif opt == '--visual':
                self.visual = str(arg)
            elif opt in ('-W', '--weighted'):
                self.weighted = True
            elif opt in ('-x', '--explain'):
                self.explain = str(arg)
            elif opt in ('-X', '--xtype'):
                self.xtype = str(arg)
            elif opt in ('-z', '--minz'):
                self.minz = True
            # exemplar
            elif opt == '--exxp':
                self.exxp = True
            else:
                assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

        if self.encode == 'none':
            self.encode = None
        elif self.encode in ('mx', 'mxe', 'maxsat', 'mxint', 'mxa') and self.solver in ('cvc4', 'mathsat', 'yices', 'z3'):
            # setting the default solver for the mxreasoning-based oracle
            self.solver = 'm22'

        # assigning the encoding for cardinality constraints
        self.cardenc = encmap[self.cardenc]

        self.files = args

    def usage(self):
        """
            Print usage message.
        """

        print('Usage: ' + os.path.basename(self.command[0]) + ' [options] input-file')
        print('Options:')
        print('        -1, --am1                  Adapt AM1 constraints when running RC2')
        print('        -a, --accmin=<float>       Minimal accuracy')
        print('                                   Available values: [0.0, 1.0] (default = 0.95)')
        print('        --approx=<int>             Approximate Lambda value')
        print('                                   Available values: [0 .. INT_MAX] (default = 0)')
        print('        -c, --use-categorical      Treat categorical features as categorical (with categorical features info if available)')
        print('        -C, --cardenc=<string>     Cardinality encoding to use')
        print('        --clocal                   Local compilation')
        print('                                   Available values: cardn, kmtot, mtot, sortn, seqc, tot (default = seqc)')
        print('        -d, --maxdepth=<int>       Maximal depth of a tree')
        print('                                   Available values: [1, INT_MAX] (default = 3)')
        print('        -D, --use-duals            Collect and use dual explanations when compiling')
        print('        -e, --encode=<string>      Encode a previously trained model')
        print('                                   Available values: maxsat, smt, smtbool, none (default = none)')
        print('        -E, --exhaust              Apply core exhaustion when running RC2')
        print('        -h, --help                 Show this message')
        print('        -f, --ds-format            Show the compiled representation (see option \'P\') in the MDS format')
        print('        --fsort                    Sorting features based on frequencies in the compilation process')
        print('        --fqupdate                 Dynamically update feature frequencies')
        print('        -i, --internal             Compute explanations wrt. internal literals of the model')
        print('        -I, --interpret            Compile an interpretable representation of the model as a decision set')
        print('        -k, --knowledge=<string>   A path to a json file storing background knowledge')
        print('        --lam=<float>              The regularized penalty')
        print('                                   Available values: none, (0, FLOAT_MAX] (default = none)')
        #print('        -l, --use-lime             Use LIME to compute an explanation')
        #print('        -L, --lime-feats           Instruct LIME to compute an explanation of this size')
        #print('                                   Available values: [1, INT_MAX], all (default = 5)')
        print('        -m, --map-file=<string>    Path to a file containing a mapping to original feature values. (default: none)')
        print('        -M, --minimum              Compute a smallest size explanation (instead of a subset-minimal one)')
        print('        -n, --nbestims=<int>       Number of trees per class')
        print('                                   Available values: [1, INT_MAX] (default = 100)')
        print('        -N, --xnum=<int>           Number of explanations to compute')
        print('                                   Available values: [1, INT_MAX], all (default = 1)')
        print('        -o, --output=<string>      Directory where output files will be stored (default: \'temp\')')
        print('        -p,                        Preprocess categorical data')
        print('        -P, --process=<string>     Preprocess a given dataset using internal BT feature-literals')
        print('        --pfiles                   Filenames to use when preprocessing')
        #print('        -q, --use-anchor           Use Anchor to compute an explanation')
        print('        -r, --rounds=<int>         Number of training rounds')
        print('                                   Available values: [1, INT_MAX] (default = 10)')
        print('        --reduce-lit=<string>      Literal reduction')
        print('                                   Available values: none, after (default = none)')
        print('        --reduce-lit-appr=<string> The way to reduct literals')
        print('                                   Available values: lin, maxsat (default = max)')
        print('        --reduce-rule              Rule reduction')
        print('        -R, --reduce=<string>      Extract an MUS from each unsatisfiable core')
        print('                                   Available values: lin, none, qxp (default = none)')
        print('        --relax=<int>              Relax the model by reducing number of weight decimal points')
        print('                                   Available values: [0, INT_MAX] (default = 0)')
        print('        --seed=<int>               Seed for random splitting')
        print('                                   Available values: [1, INT_MAX] (default = 7)')
        print('        --sep=<string>             Field separator used in input file (default = \',\')')
        print('        -s, --solver=<string>      An SMT reasoner to use')
        print('                                   Available values (smt): cvc4, mathsat, yices, z3 (default = z3)')
        print('                                   Available values (sat): g3, g4, m22, mgh, all-others-from-pysat (default = m22)')
        print('        -t, --train                Train a model of a given dataset')
        print('        -T, --trim=<int>           Trim unsatisfiable cores at most this number of times when running RC2')
        print('                                   Available values: [0, INT_MAX] (default = 0)')
        print('        --testsplit=<float>        Training and test sets split')
        print('                                   Available values: [0.0, 1.0] (default = 0.2)')
        print('        -u, --unit-mcs             Detect and block unit-size MCSes')
        print('        --use-mhs                  Use hitting set based enumeration')
        print('        --use-cld                  Use CLD heuristic')
        print('        -v, --verbose              Increase verbosity level')
        print('        -V, --validate             Validate explanation (show that it is too optimistic)')
        print('        --weighted                 Minimize the total number of literals in rule reduction')
        print('        -x, --explain=<string>     Explain a decision for a given comma-separated sample (default: none)')
        print('        -X, --xtype=<string>       Type of explanation to compute: abductive or contrastive')
        print('                                   Available values: abd, con (default = abd)')
        print('        -z, --minz                 Apply heuristic core minimization when running RC2')
