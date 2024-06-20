#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
##

# imported modules:
# ==============================================================================
from __future__ import print_function
import collections
import resource
import sys
import os
import glob
import statistics
import json
import matplotlib.pyplot as plt
try:
    from .stats import axp_stats, axp_stats2, measure_dist, \
        compare_lists, normalise, kl
except:
    from stats import axp_stats, axp_stats2, measure_dist, \
        compare_lists, normalise, kl

import re
import lzma

#
# =============================================================================
def parse_log(log, explain, classes='all', visual='../visual',
                    batch=100, compress=False):

    if 'mnist' in log:
        if 'switch' not in log:
            parse_mnist_mhs_log(log, explain, classes=classes, visual=visual, batch=batch, compress=compress)
        else:
            parse_mnist_switch_log(log, explain, classes=classes, visual=visual, batch=batch, compress=compress)
    else:
        if 'switch' not in log:
            parse_text_mhs_log(log, explain, classes=classes, visual=visual, batch=batch, compress=compress)
        else:
            parse_text_switch_log(log, explain, classes=classes, visual=visual, batch=batch, compress=compress)


def parse_mnist_mhs_log(log, explain, classes='all', visual='../visual', batch=100, compress=False):
    bg = True if '_bg' in log else False
    smallest = True if '_min' in log else False
    xtype = 'abd' if '_con' not in log else 'con'

    xnum = 1
    if 'xnum' in log:
        xnum = log.split('xnum_')[-1].rsplit('.', maxsplit=1)[0].split('_', maxsplit=1)[0].strip()
        if xnum != 'all':
            xnum = int(xnum)

    conf = log.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0].replace('_', '-').replace('.log', '')

    label = conf
    info = {"preamble": {"program": label, "prog_args": label, "prog_alias": label, "benchmark": None},
            "stats": {}}

    if log.endswith('.xz'):
        with lzma.open(log, 'rb') as f:
            lines = f.readlines()
            lines = list(map(lambda l: l.decode('utf8'), lines))
    else:
        with open(log, 'r') as f:
            lines = f.readlines()

    for i, line in enumerate(lines):
        if 'inst:' in line:
            lines = lines[i:]
            break
    else:
        assert False, 'something wrong'
        exit(1)

    insts = []
    for i, line in enumerate(lines):
        if 'inst:' in line:
            insts.append(i)
    insts.append(len(lines))

    rtimes = []
    # expls in progress
    explss_ = []
    # expls runtimes in progress
    expltimess = []
    # coexes in progress
    coexess = []
    # coexes runtimes in progress
    coextimess = []
    # final expls
    explss = []
    # final expls runtimes
    expl_timess = []
    # final dual expls
    dual_explss = []

    inst_feats = []

    for i in range(len(insts) - 1):
        explss_.append([])
        expltimess.append([])
        coexess.append([])
        coextimess.append([])
        explss.append([])
        expl_timess.append([])
        dual_explss.append([])
        rtimes.append(False)
        # inst_f = lines[i+1].split('"IF ', maxsplit=1)[0].rsplit(' THEN', maxsplit=1)[0].split(' AND ')
        # inst_feats.append(inst_f)
        for ii in range(insts[i], insts[i + 1]):
            line = lines[ii]
            if 'explaining:' in line:
                inst_feats.append(line.split(':', maxsplit=1)[-1].strip().strip('[]').split(','))
                inst_feats[-1] = list(map(lambda l: float(l.strip().strip("'").rsplit('==')[-1].strip()),
                                          inst_feats[-1]))
            elif 'expl:' in line:
                # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                explss_[-1].append(expl)
            elif '  expltime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                expltimess[-1].append(rtime)
            elif 'coex:' in line:
                try:
                    expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                    expl = list(map(lambda l: int(l), expl))
                    coexess[-1].append(expl)
                except:
                    continue
            elif '  coextime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                coextimess[-1].append(rtime)
            #elif '  explanation:' in line:
            #    # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
            #    expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
            #    expl = list(map(lambda l: int(l), expl))
            #    explss[-1].append(expl)
            #elif '  expl time:' in line:
            #    rtime = float(line.split(':', maxsplit=1)[-1])
            #    expl_timess[-1].append(rtime)
            #elif '  dual explanation:' in line:
            #    # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
            #    expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
            #    expl = list(map(lambda l: int(l), expl))
            #    dual_explss[-1].append(expl)
            elif '  rtime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                rtimes[-1] = rtime
                break
        else:
            continue

    def real_expl(inst_f, expl):
        if 'origin' not in log:
            r_expl = []
            for l in expl:
                assert l >= 0
                r_expl.append(inst_f[l])
        else:
            r_expl = [abs(l)+1 for l in expl]
        return r_expl

    # if len(rtimes) == 0:
    #    rtimes.append(3600 * 24)

    # for i in range(len(rtimes)):
    for i in range(len(insts) - 1):
        expls_ = explss_[i]
        expltimes = expltimess[i]
        coexes = coexess[i]
        coextimes = coextimess[i]
        #try:
        #    expls = explss[i]
        #except:
        #    expls = []
        # todo
        #if len(expls) == 0:
        expls = expls_

        expl_times = expl_timess[i]
        #try:
        #    dual_expls = dual_explss[i]
        #except:
        #    dual_expls = []
        #todo
        #if len(dual_expls) == 0:
        dual_expls = coexes
        status2 = True if rtimes[i] != False else False
        if 'inst' in lines[insts[i]]:
            inst_line = lines[insts[i]]
        elif 'inst' in lines[insts[i] + 1]:
            inst_line = lines[insts[i] + 1]
        elif 'inst' in lines[insts[i] + 2]:
            inst_line = lines[insts[i] + 2]
        elif 'inst' in lines[insts[i] + 3]:
            inst_line = lines[insts[i] + 3]

        inst = 'inst{0}'.format(inst_line.rsplit(':', maxsplit=1)[-1].strip())
        info['stats'][inst] = info['stats'].get(inst, {})
        stats = info['stats'][inst]
        inst_f = inst_feats[i]
        stats['inst'] = inst_f
        stats['status'] = True
        stats['status2'] = status2
        stats['expls-'] = [real_expl(inst_f, expl) for expl in expls_]
        stats['expltimes'] = expltimes
        stats['coexes'] = [real_expl(inst_f, expl) for expl in coexes]
        stats['coextimes'] = coextimes
        stats['expls'] = expls
        stats['expl-times'] = expl_times
        stats['dexpls'] = dual_expls
        stats['rtime'] = float(rtimes[i])

        try:
            stats['nof-expls'] = len(expls_)
        except:
            stats['nof-expls'] = len(expls)
        try:
            stats['nof-dexpls'] = max(len(coexes), len(dual_expls))
        except:
            stats['nof-dexpls'] = len(dual_expls)
        try:
            stats['avgtime'] = round(expltimes[-1] / len(expls), 4)
        except:
            stats['avgtime'] = 3600 * 24
        try:
            stats['avgdtime'] = round(coextimes[-1] / len(dual_expls), 4)
        except:
            stats['avgdtime'] = 3600 * 24

        try:
            stats['len-expl'] = min([len(x) for x in expls])
        except:
            stats['len-expl'] = min([len(x) for x in expls_])

        try:
            stats['len-dexpl'] = min([len(x) for x in dual_expls])
        except:
            stats['len-dexpl'] = min([len(x) for x in coexes])

    saved_dir = '../stats'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    new_file = saved_dir + '/' + conf + '.json'
    with open(new_file, 'w') as f:
        json.dump(info, f, indent=4)

    if compress:
        os.system('xz -9f {}'.format(new_file))

def parse_mnist_switch_log(log, explain, classes='all', visual='../visual', batch=100, compress=False):
    if 'bt_' in log:
        model = 'bt'
        td = 't25d3' if 't25d3' in log else 't50d3'  # (50, 3)
    elif 'bnn_' in log:
        model = 'bnn'
    else:
        model = ''

    if 'pneumoniamnist' in log.lower():
        dtname = 'pneumoniamnist'
    elif 'mnist' in log.lower():
        dtname = 'mnist'
    else:
        print('something wrong')
        exit(1)

    bg = True if '_bg' in log else False
    smallest = True if '_min' in log else False
    xtype = 'abd' if '_con' not in log else 'con'

    xnum = 1
    if 'xnum' in log:
        xnum = log.split('xnum_')[-1].rsplit('.', maxsplit=1)[0].split('_', maxsplit=1)[0].strip()
        if xnum != 'all':
            xnum = int(xnum)

    conf = log.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0].replace('_', '-').replace('.log', '')

    label = conf
    info = {"preamble": {"program": label, "prog_args": label, "prog_alias": label, "benchmark": None},
            "stats": {}}

    if log.endswith('.xz'):
        with lzma.open(log, 'rb') as f:
            lines = f.readlines()
            lines = list(map(lambda l: l.decode('utf8'), lines))
    else:
        with open(log, 'r') as f:
            lines = f.readlines()

    for i, line in enumerate(lines):
        if 'inst:' in line:
            lines = lines[i:]
            break
    else:
        print('something wrong')
        exit()

    insts = []
    for i, line in enumerate(lines):
        if 'inst:' in line:
            insts.append(i)
    insts.append(len(lines))

    rtimes = []
    # expls in progress
    explss_ = []
    # expls runtimes in progress
    expltimess = []
    # coexes in progress
    coexess = []
    # coexes runtimes in progress
    coextimess = []
    # final expls
    explss = []
    # final expls runtimes
    expl_timess = []
    # final dual expls
    dual_explss = []
    switches = []

    inst_feats = []

    for i in range(len(insts) - 1):
        explss_.append([])
        expltimess.append([])
        coexess.append([])
        coextimess.append([])
        explss.append([])
        expl_timess.append([])
        dual_explss.append([])
        rtimes.append(False)
        switches.append([])
        # inst_f = lines[i+1].split('"IF ', maxsplit=1)[0].rsplit(' THEN', maxsplit=1)[0].split(' AND ')
        # inst_feats.append(inst_f)
        for ii in range(insts[i], insts[i + 1]):
            line = lines[ii]
            if 'explaining:' in line:
                inst_feats.append(line.split(':', maxsplit=1)[-1].strip().strip('[]').split(','))
                inst_feats[-1] = list(map(lambda l: float(l.strip().strip("'").rsplit('==')[-1].strip()),
                                          inst_feats[-1]))
            elif 'switch' in line:
                switch = float('inf')
                if len(expltimess[-1]) > 0:
                    switch = min(switch, expltimess[-1][-1])
                if len(coextimess[-1]) > 0:
                    switch = min(switch, coextimess[-1][-1])
                switches[-1].append(switch)
            elif 'cxp:' in line:
                #print(ii)
                # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                if xtype == 'abd':
                    coexess[-1].append(expl)
                else:
                    explss_[-1].append(expl)
            elif '  cxptime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                if xtype == 'abd':
                    coextimess[-1].append(rtime)
                else:
                    expltimess[-1].append(rtime)
            elif 'axp:' in line:
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                if xtype == 'abd':
                    explss_[-1].append(expl)
                else:
                    coexess[-1].append(expl)
            elif '  axptime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                if xtype == 'abd':
                    expltimess[-1].append(rtime)
                else:
                    coextimess[-1].append(rtime)
            elif '  explanation:' in line:
                # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                explss[-1].append(expl)
            elif '  expl time:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                expl_timess[-1].append(rtime)
            elif '  dual explanation:' in line:
                # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                dual_explss[-1].append(expl)
            elif '  rtime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                rtimes[-1] = rtime
                break
        else:
            continue

    def real_expl(inst_f, expl):
        if 'origin' not in log:
            r_expl = []
            for l in expl:
                assert l >= 0
                r_expl.append(inst_f[l])
        else:
            r_expl = [abs(l)+1 for l in expl]
        return r_expl

    # if len(rtimes) == 0:
    #    rtimes.append(3600 * 24)

    # for i in range(len(rtimes)):
    for i in range(len(insts) - 1):
        expls_ = explss_[i]
        expltimes = expltimess[i]
        coexes = coexess[i]
        coextimes = coextimess[i]
        #try:
        #    expls = explss[i]
        #except:
        #    expls = []
        # todo
        #if len(expls) == 0:
        expls = expls_

        expl_times = expl_timess[i]
        #try:
        #    dual_expls = dual_explss[i]
        #except:
        #    dual_expls = []
        #todo
        #if len(dual_expls) == 0:
        dual_expls = coexes
        status2 = True if rtimes[i] != False else False
        if 'inst' in lines[insts[i]]:
            inst_line = lines[insts[i]]
        elif 'inst' in lines[insts[i] + 1]:
            inst_line = lines[insts[i] + 1]
        elif 'inst' in lines[insts[i] + 2]:
            inst_line = lines[insts[i] + 2]
        elif 'inst' in lines[insts[i] + 3]:
            inst_line = lines[insts[i] + 3]

        inst = 'inst{0}'.format(inst_line.rsplit(':', maxsplit=1)[-1].strip())
        info['stats'][inst] = info['stats'].get(inst, {})
        stats = info['stats'][inst]
        inst_f = inst_feats[i]
        stats['inst'] = inst_f
        stats['status'] = True
        stats['status2'] = status2
        stats['switch'] = switches[i]
        if len(stats['switch']) > 1:
            stats['switch'] = stats['switch'][:-1]
        stats['expls-'] = [real_expl(inst_f, expl) for expl in expls_]
        stats['expltimes'] = expltimes
        stats['coexes'] = [real_expl(inst_f, expl) for expl in coexes]
        stats['coextimes'] = coextimes
        stats['expls'] = expls
        stats['expl-times'] = expl_times
        stats['dexpls'] = dual_expls
        stats['rtime'] = float(rtimes[i])

        try:
            stats['nof-expls'] = len(expls_)
        except:
            stats['nof-expls'] = len(expls)
        try:
            stats['nof-dexpls'] = max(len(coexes), len(dual_expls))
        except:
            stats['nof-dexpls'] = len(dual_expls)
        try:
            stats['avgtime'] = round(expltimes[-1] / len(expls), 4)
        except:
            stats['avgtime'] = 3600 * 24
        try:
            stats['avgdtime'] = round(coextimes[-1] / len(dual_expls), 4)
        except:
            stats['avgdtime'] = 3600 * 24

        try:
            stats['len-expl'] = min([len(x) for x in expls])
        except:
            stats['len-expl'] = min([len(x) for x in expls_])
        try:
            stats['len-dexpl'] = min([len(x) for x in dual_expls])
        except:
            stats['len-dexpl'] = min([len(x) for x in coexes])

    saved_dir = '../stats'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    new_file = saved_dir + '/' + conf + '.json'
    with open(new_file, 'w') as f:
        json.dump(info, f, indent=4)

    if compress:
        os.system('xz -9f {}'.format(new_file))

def parse_text_mhs_log(log, explain, classes='all', visual='../visual', batch=100, compress=False):
    xnum = 1
    if 'xnum' in log:
        xnum = log.split('xnum_')[-1].rsplit('.', maxsplit=1)[0].split('_', maxsplit=1)[0].strip()
        if xnum != 'all':
            xnum = int(xnum)

    conf = log.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0].replace('_', '-').replace('.log', '')

    label = conf
    info = {"preamble": {"program": label, "prog_args": label, "prog_alias": label, "benchmark": None},
            "stats": {}}

    if log.endswith('.xz'):
        with lzma.open(log, 'rb') as f:
            lines = f.readlines()
            lines = list(map(lambda l: l.decode('utf8'), lines))
    else:
        with open(log, 'r') as f:
            lines = f.readlines()

    for i, line in enumerate(lines):
        if 'inst:' in line:
            lines = lines[i:]
            break
    else:
        assert False, 'something wrong'
        exit(1)

    insts = []
    for i, line in enumerate(lines):
        if 'inst:' in line:
            insts.append(i)
    insts.append(len(lines))

    rtimes = []
    # expls in progress
    explss_ = []
    # expls runtimes in progress
    expltimess = []
    # coexes in progress
    coexess = []
    # coexes runtimes in progress
    coextimess = []
    # final expls
    explss = []
    # final expls runtimes
    expl_timess = []
    # final dual expls
    dual_explss = []

    inst_feats = []

    for i in range(len(insts) - 1):
        explss_.append([])
        expltimess.append([])
        coexess.append([])
        coextimess.append([])
        explss.append([])
        expl_timess.append([])
        dual_explss.append([])
        rtimes.append(False)
        # inst_f = lines[i+1].split('"IF ', maxsplit=1)[0].rsplit(' THEN', maxsplit=1)[0].split(' AND ')
        # inst_feats.append(inst_f)
        for ii in range(insts[i], insts[i + 1]):
            line = lines[ii]
            if 'explaining:' in line:
                inst_feats.append(line.split(':', maxsplit=1)[-1].strip().strip('"').split(' THEN ', maxsplit=1)[0].split('IF ', maxsplit=1)[-1].split(' AND '))
            elif 'expl:' in line:
                # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                explss_[-1].append(expl)
            elif '  expltime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                expltimess[-1].append(rtime)
            elif 'coex:' in line:
                try:
                    expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                    expl = list(map(lambda l: int(l), expl))
                    coexess[-1].append(expl)
                except:
                    continue
            elif '  coextime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                coextimess[-1].append(rtime)
            #elif '  explanation:' in line:
            #    # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
            #    expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
            #    expl = list(map(lambda l: int(l), expl))
            #    explss[-1].append(expl)
            #elif '  expl time:' in line:
            #    rtime = float(line.split(':', maxsplit=1)[-1])
            #    expl_timess[-1].append(rtime)
            #elif '  dual explanation:' in line:
            #    # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
            #    expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
            #    expl = list(map(lambda l: int(l), expl))
            #    dual_explss[-1].append(expl)
            elif '  rtime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                rtimes[-1] = rtime
                break
        else:
            continue

    def real_expl(inst_f, expl):
        #if 'origin' not in log:
        #    r_expl = []
        #    for l in expl:
        #        assert l >= 0
        #        r_expl.append(inst_f[l])
        #else:
        r_expl = [abs(l)+1 for l in expl]
        return r_expl

    # if len(rtimes) == 0:
    #    rtimes.append(3600 * 24)
    print(len(insts)-1)
    # for i in range(len(rtimes)):
    for i in range(len(insts) - 1):
        expls_ = explss_[i]
        expltimes = expltimess[i]
        coexes = coexess[i]
        coextimes = coextimess[i]
        #try:
        #    expls = explss[i]
        #except:
        #    expls = []
        # todo
        #if len(expls) == 0:
        expls = expls_

        expl_times = expl_timess[i]
        #try:
        #    dual_expls = dual_explss[i]
        #except:
        #    dual_expls = []
        #todo
        #if len(dual_expls) == 0:
        dual_expls = coexes
        status2 = True if rtimes[i] != False else False
        if 'inst' in lines[insts[i]]:
            inst_line = lines[insts[i]]
        elif 'inst' in lines[insts[i] + 1]:
            inst_line = lines[insts[i] + 1]
        elif 'inst' in lines[insts[i] + 2]:
            inst_line = lines[insts[i] + 2]
        elif 'inst' in lines[insts[i] + 3]:
            inst_line = lines[insts[i] + 3]

        inst = 'inst{0}'.format(inst_line.rsplit(':', maxsplit=1)[-1].strip())
        info['stats'][inst] = info['stats'].get(inst, {})
        stats = info['stats'][inst]
        inst_f = inst_feats[i]
        stats['inst'] = inst_f
        stats['status'] = True
        stats['status2'] = status2
        stats['expls-'] = [real_expl(inst_f, expl) for expl in expls_]
        stats['expltimes'] = expltimes
        stats['coexes'] = [real_expl(inst_f, expl) for expl in coexes]
        stats['coextimes'] = coextimes
        stats['expls'] = expls
        stats['expl-times'] = expl_times
        stats['dexpls'] = dual_expls
        stats['rtime'] = float(rtimes[i])

        try:
            stats['nof-expls'] = len(expls_)
        except:
            stats['nof-expls'] = len(expls)
        try:
            stats['nof-dexpls'] = max(len(coexes), len(dual_expls))
        except:
            stats['nof-dexpls'] = len(dual_expls)
        try:
            stats['avgtime'] = round(expltimes[-1] / len(expls), 4)
        except:
            stats['avgtime'] = 3600 * 24
        try:
            stats['avgdtime'] = round(coextimes[-1] / len(dual_expls), 4)
        except:
            stats['avgdtime'] = 3600 * 24

        try:
            stats['len-expl'] = min([len(x) for x in expls])
        except:
            stats['len-expl'] = min([len(x) for x in expls_])
        try:
            stats['len-dexpl'] = min([len(x) for x in dual_expls])
        except:
            stats['len-dexpl'] = min([len(x) for x in coexes])

    saved_dir = '../stats'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    new_file = saved_dir + '/' + conf + '.json'
    with open(new_file, 'w') as f:
        json.dump(info, f, indent=4)

    if compress:
        os.system('xz -9f {}'.format(new_file))

def parse_text_switch_log(log, explain, classes='all', visual='../visual', batch=100, compress=False):
    xtype = 'abd' if '_con' not in log else 'con'

    xnum = 1
    if 'xnum' in log:
        xnum = log.split('xnum_')[-1].rsplit('.', maxsplit=1)[0].split('_', maxsplit=1)[0].strip()
        if xnum != 'all':
            xnum = int(xnum)

    conf = log.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0].replace('_', '-').replace('.log', '')

    label = conf
    info = {"preamble": {"program": label, "prog_args": label, "prog_alias": label, "benchmark": None},
            "stats": {}}

    if log.endswith('.xz'):
        with lzma.open(log, 'rb') as f:
            lines = f.readlines()
            lines = list(map(lambda l: l.decode('utf8'), lines))
    else:
        with open(log, 'r') as f:
            lines = f.readlines()

    for i, line in enumerate(lines):
        if 'inst:' in line:
            lines = lines[i:]
            break
    else:
        print('something wrong')
        exit()

    insts = []
    for i, line in enumerate(lines):
        if 'inst:' in line:
            insts.append(i)
    insts.append(len(lines))

    rtimes = []
    # expls in progress
    explss_ = []
    # expls runtimes in progress
    expltimess = []
    # coexes in progress
    coexess = []
    # coexes runtimes in progress
    coextimess = []
    # final expls
    explss = []
    # final expls runtimes
    expl_timess = []
    # final dual expls
    dual_explss = []
    switches = []

    inst_feats = []

    for i in range(len(insts) - 1):
        explss_.append([])
        expltimess.append([])
        coexess.append([])
        coextimess.append([])
        explss.append([])
        expl_timess.append([])
        dual_explss.append([])
        rtimes.append(False)
        switches.append([])
        # inst_f = lines[i+1].split('"IF ', maxsplit=1)[0].rsplit(' THEN', maxsplit=1)[0].split(' AND ')
        # inst_feats.append(inst_f)
        for ii in range(insts[i], insts[i + 1]):
            line = lines[ii]
            if 'explaining:' in line:
                inst_feats.append(line.split(':', maxsplit=1)[-1].strip().strip('"').split(' THEN ', maxsplit=1)[0].split('IF ', maxsplit=1)[-1].split(' AND '))
            elif 'switch' in line:
                switch = float('inf')
                if len(expltimess[-1]) > 0:
                    switch = min(switch, expltimess[-1][-1])

                if len(coextimess[-1]) > 0:
                    switch = min(switch, coextimess[-1][-1])
                switches[-1].append(switch)
            elif 'cxp:' in line:
                #print(ii)
                # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                if xtype == 'abd':
                    coexess[-1].append(expl)
                else:
                    explss_[-1].append(expl)
            elif '  cxptime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                if xtype == 'abd':
                    coextimess[-1].append(rtime)
                else:
                    expltimess[-1].append(rtime)
            elif 'axp:' in line:
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                if xtype == 'abd':
                    explss_[-1].append(expl)
                else:
                    coexess[-1].append(expl)
            elif '  axptime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                if xtype == 'abd':
                    expltimess[-1].append(rtime)
                else:
                    coextimess[-1].append(rtime)
            #elif '  explanation:' in line:
            #    # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
            #    expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
            #    expl = list(map(lambda l: int(l), expl))
            #    explss[-1].append(expl)
            #elif '  expl time:' in line:
            #    rtime = float(line.split(':', maxsplit=1)[-1])
            #    expl_timess[-1].append(rtime)
            #elif '  dual explanation:' in line:
            #    # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
            #    expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
            #    expl = list(map(lambda l: int(l), expl))
            #    dual_explss[-1].append(expl)
            elif '  rtime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                rtimes[-1] = rtime
                break
        else:
            continue

    def real_expl(inst_f, expl):
        #if 'origin' not in log:
        #    r_expl = []
        #    for l in expl:
        #        assert l >= 0
        #        r_expl.append(inst_f[l])
        #else:
        r_expl = [abs(l)+1 for l in expl]
        return r_expl

    # if len(rtimes) == 0:
    #    rtimes.append(3600 * 24)

    # for i in range(len(rtimes)):
    for i in range(len(insts) - 1):
        expls_ = explss_[i]
        expltimes = expltimess[i]
        coexes = coexess[i]
        coextimes = coextimess[i]
        #try:
        #    expls = explss[i]
        #except:
        #    expls = []
        # todo
        #if len(expls) == 0:
        expls = expls_
        expl_times = expl_timess[i]
        #try:
        #    dual_expls = dual_explss[i]
        #except:
        #    dual_expls = []
        #todo
        #if len(dual_expls) == 0:
        dual_expls = coexes
        status2 = True if rtimes[i] != False else False
        if 'inst' in lines[insts[i]]:
            inst_line = lines[insts[i]]
        elif 'inst' in lines[insts[i] + 1]:
            inst_line = lines[insts[i] + 1]
        elif 'inst' in lines[insts[i] + 2]:
            inst_line = lines[insts[i] + 2]
        elif 'inst' in lines[insts[i] + 3]:
            inst_line = lines[insts[i] + 3]

        inst = 'inst{0}'.format(inst_line.rsplit(':', maxsplit=1)[-1].strip())
        info['stats'][inst] = info['stats'].get(inst, {})
        stats = info['stats'][inst]
        inst_f = inst_feats[i]
        stats['inst'] = inst_f
        stats['status'] = True
        stats['status2'] = status2
        stats['switch'] = switches[i]
        if len(stats['switch']) > 1:
            stats['switch'] = stats['switch'][:-1]
        stats['expls-'] = [real_expl(inst_f, expl) for expl in expls_]
        stats['expltimes'] = expltimes
        stats['coexes'] = [real_expl(inst_f, expl) for expl in coexes]
        stats['coextimes'] = coextimes
        stats['expls'] = expls
        stats['expl-times'] = expl_times
        stats['dexpls'] = dual_expls
        stats['rtime'] = float(rtimes[i])

        try:
            stats['nof-expls'] = len(expls_)
        except:
            stats['nof-expls'] = len(expls)
        try:
            stats['nof-dexpls'] = max(len(coexes), len(dual_expls))
        except:
            stats['nof-dexpls'] = len(dual_expls)
        try:
            stats['avgtime'] = round(expltimes[-1] / len(expls), 4)
        except:
            stats['avgtime'] = 3600 * 24
        try:
            stats['avgdtime'] = round(coextimes[-1] / len(dual_expls), 4)
        except:
            stats['avgdtime'] = 3600 * 24

        try:
            stats['len-expl'] = min([len(x) for x in expls])
        except:
            stats['len-expl'] = min([len(x) for x in expls_])
        try:
            stats['len-dexpl'] = min([len(x) for x in dual_expls])
        except:
            stats['len-dexpl'] = min([len(x) for x in coexes])

    saved_dir = '../stats'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    new_file = saved_dir + '/' + conf + '.json'
    with open(new_file, 'w') as f:
        json.dump(info, f, indent=4)

    if compress:
        os.system('xz -9f {}'.format(new_file))

def inst2dt_log(files, compress=False):
    k2logs = collections.defaultdict(lambda: [])
    for file in files:
        k = file.rsplit('_', maxsplit=1)[0]
        k2logs[k].append(file)
    keys = sorted(k2logs.keys())

    for k, logs in k2logs.items():
        #print()
        #continue
        logs.sort(key=lambda l: int(l.rsplit('_', maxsplit=1)[-1].rsplit('.log', maxsplit=1)[0]))
        cmb_lines = []
        for log in logs:
            if log.endswith('.xz'):
                with lzma.open(log, 'rb') as f:
                    lines = f.readlines()
                    lines = list(map(lambda l: l.decode('utf8'), lines))
            else:
                with open(log, 'r') as f:
                    lines = f.readlines()
            for i, line in enumerate(lines):
                if 'inst:' in line:
                    lines = lines[i:]
                    break

            # rtime = list(filter(lambda l: '  rtime:' in l, lines))
            # if len(rtime) == 0:
            #    rtime = list(filter(lambda l: '  expltime:' in l, lines))[-1].replace('expltime:', 'rtime:')
            #    lines.append(rtime + '\n')
            for i, line in enumerate(lines):
                if '  rtime:' in line:
                    lines = lines[:i+1]
                    break
            else:
                #assert False, 'incompleted'
                print('incompleted:', log)
                #print(log)
                continue

            cmb_lines.extend(lines)
            cmb_lines.append('\n')
        new_file = k + '.log'
        new_file = new_file.replace('cut/', '')
        with open(new_file, 'w') as f:
            f.write(''.join(cmb_lines))

        if compress:
            os.system('xz -9f {}'.format(new_file))
            #new_xz_file = new_file + '.xz'
            #if os.path.isfile(new_xz_file):
            #    os.system('rm {}'.format(new_xz_file))

def cal_metrics_iter(file):
    if file.endswith('.json'):
        with open(file, 'r') as f:
            info = json.load(f)
    elif file.endswith('.xz'):
        with lzma.open(file, 'rb') as f:
            info = json.load(f)
    else:
        print('something wrong')
        exit(1)

    for inst in info['stats']:
        inst_info = info['stats'][inst]
        if 'abd' in file:
            axps = inst_info['expls-']
            #cxps = inst_info['coexes']
        elif 'con' in file:
            #cxps = inst_info['expls-']
            axps = inst_info['coexes']
        else:
            print('xtype wrong')
            exit(1)

        gt = {'axp': axp_stats(axps)}
              #'cxp': cxp_stats(cxps)}
        expls = {'axp': axps}
                 #'cxp': cxps}
        xtype2stats = {'axp': axp_stats}
                       #'cxp': cxp_stats}
        errors = {'axp': []}
                  #'cxp': []}
        coefs = {'axp': {metric: [] for metric in ('kendalltau', 'rbo')}}
                 #'cxp': {metric: [] for metric in ('kendalltau', 'rbo')}}

        for xtype in ['axp']:# 'cxp']:
            cur_expls = []
            for xp in expls[xtype]:
                cur_expls.append(xp)
                cnt = xtype2stats[xtype](cur_expls)
                cnt_gt = gt[xtype]
                """error"""
                metric = 'manhattan'
                errors[xtype].append(measure_dist(cnt, cnt_gt,
                                                  shape=(10, 10), metric=metric))

                """rank comparison"""
                for metric in ('kendall_tau', 'rbo'):
                    coef = compare_lists(cnt, cnt_gt, metric=metric)
                    coefs[xtype][metric.replace('_', '')].append(coef)
            xtype_ = 'expl' if ('abd' in file and xtype == 'axp') or ('con' in file and xtype == 'cxp')\
                else 'coex'

            inst_info['errors-{}'.format(xtype_)] = errors[xtype]
            inst_info['coefs-{}'.format(xtype_)] = coefs[xtype]

    if file.endswith('.json'):
        with open(file, 'w') as f:
            json.dump(info, f, indent=4)
        return file
    else:
        ori_json_file = file.rsplit('.', maxsplit=1)[0]
        #new_json_file = ori_json_file.replace('.json', '-new.json')
        with open(ori_json_file, 'w') as f:
            json.dump(info, f, indent=4)
        return ori_json_file
        #os.system('xz -9fv {}'.format(ori_json_file))

def cal_kl(file):
    if file.endswith('.json'):
        with open(file, 'r') as f:
            info = json.load(f)
    elif file.endswith('.xz'):
        with lzma.open(file, 'rb') as f:
            info = json.load(f)
    else:
        print('something wrong')
        exit(1)

    for inst in info['stats']:
        inst_info = info['stats'][inst]
        if 'abd' in file:
            axps = inst_info['expls-']
            #cxps = inst_info['coexes']
        elif 'con' in file:
            #cxps = inst_info['expls-']
            axps = inst_info['coexes']
        else:
            print('xtype wrong')
            exit(1)

        gt = {'axp': axp_stats(axps)}
        sum_ffa = sum(gt['axp'].values())
        gt['axp-dist'] = {pix: ffa/sum_ffa for pix, ffa in gt['axp'].items()}
        #gt['dist-axp'] =
              #'cxp': cxp_stats(cxps)}
        expls = {'axp': axps}
                 #'cxp': cxps}
        xtype2stats = {'axp': axp_stats}
                       #'cxp': cxp_stats}
        kls = {'axp': []}
        #errors = {'axp': []}
        #          #'cxp': []}
        #coefs = {'axp': {metric: [] for metric in ('kendalltau', 'rbo')}}
        #         #'cxp': {metric: [] for metric in ('kendalltau', 'rbo')}}

        for xtype in ['axp']:# 'cxp']:
            cur_expls = []
            for xp in expls[xtype]:
                cur_expls.append(xp)
                cnt = xtype2stats[xtype](cur_expls)
                cnt_gt = gt[xtype]
                """
                Kullback–Leibler divergence
                """
                approx_sum_ffa = sum(cnt.values())
                cur_dist = {pix: ffa / approx_sum_ffa for pix, ffa in cnt.items()}
                kls[xtype].append(kl(p=gt[xtype + '-dist'], q=cur_dist))
                #"""error"""
                #metric = 'manhattan'
                #errors[xtype].append(measure_dist(cnt, cnt_gt,
                #                                  shape=(10, 10), metric=metric))

                #"""rank comparison"""
                #for metric in ('kendall_tau', 'rbo'):
                #    coef = compare_lists(cnt, cnt_gt, metric=metric)
                #    coefs[xtype][metric.replace('_', '')].append(coef)

            xtype_ = 'expl' if ('abd' in file and xtype == 'axp') or ('con' in file and xtype == 'cxp')\
                else 'coex'

            #inst_info['errors-{}'.format(xtype_)] = errors[xtype]
            #inst_info['coefs-{}'.format(xtype_)] = coefs[xtype]
            inst_info['kls-{}'.format(xtype_)] = kls[xtype]

    if file.endswith('.json'):
        with open(file, 'w') as f:
            json.dump(info, f, indent=4)
    else:
        ori_json_file = file.rsplit('.', maxsplit=1)[0]
        #new_json_file = ori_json_file.replace('.json', '-new.json')
        with open(ori_json_file, 'w') as f:
            json.dump(info, f, indent=4)
        os.system('xz -9fv {}'.format(ori_json_file))

#
# =============================================================================
if __name__ == "__main__":
    """
    merge inst level logs to dt level
    """
    logs = sorted(glob.glob('../logs/cut/*.log*'),
                  key=lambda l: int(l.replace('cut', '').rsplit('.log', maxsplit=1)[0].rsplit('_', maxsplit=1)[1]))
    inst2dt_log(logs, compress=False)

    """
    parse logs
    """
    logs = {}
    logs[('formal', 'all')] = sorted(glob.glob('../logs/*.log*'))
    slides = set()
    for (explain, classes), logs_ in logs.items():
        for log in sorted(logs_):
            print(log)
            parse_log(log, classes=classes, explain=explain,
                            compress=False)

    """
    calculate metrics
    """
    files = sorted(glob.glob('../stats/*.json*'))

    for file in files:
        print(file)
        # ffa
        file_ = cal_metrics_iter(file)
        # kl divergence
        cal_kl(file_)
    exit()