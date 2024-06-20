import os
import sys
import numpy as np
import functools
import glob
import json
import matplotlib.pyplot as plt
import lzma
try:
    from .stats import axp_stats, axp_stats2, measure_dist, \
        compare_lists, normalise, kl
except:
    from stats import axp_stats,measure_dist, \
        compare_lists, normalise, kl

class Queue(object):
    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self.queue = []
        self.old_head = 0

    def __str__(self):
        return '{}'.format(self.queue)

    def put(self, v):
        self.queue.append(v)
        if len(self.queue) > self.maxsize and self.maxsize != 0:
            self.old_head = self.queue.pop(0)
            return self.old_head
        return None

    def get(self):
        if len(self.queue) > 0:
            return self.queue.pop(0)
        else:
            print('Queue is empty')
            return None

    def __len__(self):
        return len(self.queue)

    def __iter__(self):
        for q in self.queue:
            yield q

    @property
    def last(self):
        return self.queue[-1]


def sliding_win(times, k=25):
    # expl_win = Queue(maxsize=k)
    time_win = Queue(maxsize=k)
    res = []
    for time in times:
        time_win.put(time)
        res.append(len(time_win) / (time_win.last - time_win.old_head))
    return res


def sliding_win_iter(expls, k=25):
    # expl_win = Queue(maxsize=k)
    win = Queue(maxsize=k)
    res = []
    for expl in expls:
        win.put(expl)
        res.append(sum(win) / len(win))
    return res


def sliding_win_size(sizes, k=25):
    # expl_win = Queue(maxsize=k)
    size_win = Queue(maxsize=k)
    res = []
    for size in sizes:
        size_win.put(size)
        res.append(sum(size_win) / len(size_win))
    return res

def merge_list(time_lists, expl_lists, issize=False, accumulate=False,
               ismetric=False, derivative=False, iteration=False):
    idx_list = [0] * len(time_lists)
    def unexplore(idx_list, time_lens):
        for i, time_len in zip(idx_list, time_lens):
            if i < time_len:
                return True
        return False
    time_lens = list(map(lambda l: len(l), time_lists))
    if ismetric:
        res = []
        for r in expl_lists:
            res.append([r[0]])
    else:
        res = [[0] for i in range(len(time_lists))]

    times = []
    while unexplore(idx_list, time_lens):
        min_idx = -1
        min_time = float('inf')
        for i, time_list in enumerate(time_lists):
            if idx_list[i] < len(time_list):
                cur_time = time_list[idx_list[i]]
                if cur_time < min_time:
                    min_time = cur_time
                    min_idx = i
        assert min_idx != -1
        if issize:
            if accumulate:
                for i in range(len(time_lists)):
                    if i == min_idx:
                        res[i].append(res[i][-1]+len(expl_lists[i][idx_list[min_idx]]))
                    else:
                        res[i].append(res[i][-1])
            else:
                for i in range(len(time_lists)):
                    if i == min_idx:
                        res[i].append(len(expl_lists[i][idx_list[min_idx]]))
                    else:
                        res[i].append(res[i][-1])
        elif ismetric:
            for i in range(len(time_lists)):
                if i == min_idx:
                    res[i].append(expl_lists[i][idx_list[min_idx]])
                else:
                    res[i].append(res[i][-1])
        elif derivative:
            if iteration:
                for i in range(len(time_lists)):
                    if i == min_idx:
                        res[i].append(len(expl_lists[i][idx_list[min_idx]]))
                    elif idx_list[i] < time_lens[i]:
                        res[i].append(res[i][-1])
                    else:
                        res[i].append(res[i][-1])
            else:
                for i in range(len(time_lists)):
                    if i == min_idx:
                        res[i].append(expl_lists[i][idx_list[min_idx]])
                    elif idx_list[i] < time_lens[i]:
                        res[i].append(res[i][-1])
                    else:
                        res[i].append(0)
        elif iteration:
            for i in range(len(time_lists)):
                if i == min_idx:
                    res[i].append(1)
                else:
                    res[i].append(0)
        else:
            for i in range(len(time_lists)):
                if i == min_idx:
                    res[i].append(res[i][-1]+1)
                else:
                    res[i].append(res[i][-1])
        times.append(min_time)
        idx_list[min_idx] += 1
    res = list(map(lambda l: l[1:], res))
    return res, times


def line_chart(y, x, labels, title, saved_file=None, rotation=0,
               xlabel='Time(s)', ylabel=None, xlog=False, switches=None):
    # Change the style of plot
    plt.style.use('seaborn-darkgrid')

    # Create a color palette
    palette = plt.get_cmap('Set1')

    if switches is None:
        switches = [[]] * len(labels)

    # Plot multiple lines
    num = 0
    for label, v, switch in zip(labels, y, switches):
        num += 1
        plt.plot(x, v, marker='', color=palette(num),
                 linewidth=1, alpha=0.9, label=label)

        if len(switch) > 0:
            sp = v[len(list(filter(lambda l: l <= switch[0], x)))-1]
            plt.plot(switch[0], sp, marker="o", markersize=5, markeredgecolor="orange",
                     markerfacecolor="orange")

    # Add legend
    plt.legend(loc=2, ncol=2)

    # Add titles
    plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel(xlabel)
    if xlog:
        plt.xscale('log', base=10)
    if 'size' in title:
        plt.ylabel("Size")
        if saved_file:
            saved_file = saved_file.replace('-size', '') + '-size'
    elif ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("#Expls")

    plt.xticks(rotation=rotation)
    #print('saved_file:', saved_file)
    ## Show the graph
    #if saved_file:
    #    plt.savefig(saved_file + '.pdf', bbox_inches='tight', pad_inches=0)
    #else:
    #    plt.show()
    plt.close()
    return saved_file

def visualise(abd_info=None, con_info=None, title='', abd_info_=None, con_info_=None,
              ismetric=False, issize=False, accumulate=False, conf_='switch', metric='cnt',
              xlog=False):
    insts = set()
    for i, info in enumerate([abd_info, con_info, abd_info_, con_info_]):
        if info is not None:
            if len(insts) == 0:
                insts = set(info['stats'].keys())
            else:
                insts = insts.intersection(info['stats'].keys())

    insts = sorted(insts, key=lambda l: int(l[4:]))

    all = []
    for inst in insts:
        time_list, expl_list, labels, switches = [], [], [], []

        if abd_info is not None:
            abd_stats = abd_info['stats'][inst]
            if metric in ('error', 'kl'):
                expl_list.extend([abd_stats[metric + 's-expl']])#, abd_stats['errors-coex']])
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                expl_list.extend([abd_stats['coefs-expl'][m1]])#, abd_stats['coefs-coex'][m1]])
            else:
                expl_list.extend([abd_stats['expls-'], abd_stats['coexes']])

            switches.append([])
            if metric in ('error', 'kl') or 'coef' in metric:
                labels.extend(['axpenum-axp'])#, 'axpenum-cxp'])
                time_list.extend([abd_stats['expltimes']])
            else:
                labels.extend(['axpenum-axp', 'axpenum-cxp'])
                time_list.extend([abd_stats['expltimes'], abd_stats['coextimes']])
                switches.append([])


        if con_info is not None:
            con_stats = con_info['stats'][inst]
            if metric in ('error', 'kl'):
                #expl_list.extend([con_stats['errors-expl'], con_stats['errors-coex']])
                expl_list.extend([con_stats[metric + 's-coex']])
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                #expl_list.extend([con_stats['coefs-expl'][m1], con_stats['coefs-coex'][m1]])
                expl_list.extend([con_stats['coefs-coex'][m1]])
            else:
                expl_list.extend([con_stats['expls-'], con_stats['coexes']])

            switches.append([])
            if metric in ('error', 'kl') or 'coef' in metric:
                labels.extend(['cxpenum-axp'])
                time_list.extend([con_stats['coextimes']])
            else:
                labels.extend(['cxpenum-cxp', 'cxpenum-axp'])
                time_list.extend([con_stats['expltimes'], con_stats['coextimes']])
                switches.append([])

        if abd_info_ is not None:
            abd_stats_ = abd_info_['stats'][inst]
            if metric in ('error', 'kl'):
                expl_list.extend([abd_stats_[metric + 's-expl']])#, abd_stats_['errors-coex']])
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                expl_list.extend([abd_stats_['coefs-expl'][m1]])#, abd_stats_['coefs-coex'][m1]])
            else:
                expl_list.extend([abd_stats_['expls-'], abd_stats_['coexes']])

            switches.append(abd_stats_['switch'])
            if metric in ('error', 'kl') or 'coef' in metric:
                labels.extend(['{}-axpenum-axp'.format(conf_)])#, '{}-axpenum-cxp'.format(conf_)])
                time_list.extend([abd_stats_['expltimes']])#, abd_stats_['coextimes']])
            else:
                labels.extend(['{}-axpenum-axp'.format(conf_), '{}-axpenum-cxp'.format(conf_)])
                time_list.extend([abd_stats_['expltimes'], abd_stats_['coextimes']])
                switches.append(abd_stats_['switch'])

        if con_info_ is not None:
            con_stats_ = con_info_['stats'][inst]
            if metric in ('error', 'kl'):
                #expl_list.extend([con_stats_['errors-expl'], con_stats_['errors-coex']])
                expl_list.extend([con_stats_[metric + 's-coex']])
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                #expl_list.extend([con_stats_['coefs-expl'][m1], con_stats_['coefs-coex'][m1]])
                expl_list.extend([con_stats_['coefs-coex'][m1]])
            else:
                expl_list.extend([con_stats_['expls-'], con_stats_['coexes']])
            switches.append(con_stats_['switch'])
            if metric in ('error', 'kl') or 'coef' in metric:
                labels.extend(['{}-cxpenum-axp'.format(conf_)])
                time_list.extend([con_stats_['coextimes']])
            else:
                labels.extend(['{}-cxpenum-cxp'.format(conf_), '{}-cxpenum-axp'.format(conf_)])
                time_list.extend([con_stats_['expltimes'], con_stats_['coextimes']])
                switches.append(con_stats_['switch'])

        res, times = merge_list(time_list, expl_list, issize=issize, ismetric=ismetric,
                                accumulate=accumulate)
        if metric != 'cnt':
            saved_file = title + '-{}-{}'.format(metric, inst) + '-m'
        else:
            saved_file = title + '-{}'.format(inst) + '-m'

        # replace inf by the max value
        res = np.array(res)
        #res[res == float('inf')] = res[res != float('inf')].max()
        res[res == float('inf')] = res[res != float('inf')].max() * 2 #0.5#

        new_saved_file = line_chart(res, times, labels, title=title + '-{}'.format(inst),
                                    saved_file=saved_file, ylabel=metric, xlog=xlog,
                                    switches=switches)
        # return num_expls, times
        all.append(new_saved_file + '.pdf')

    return all

def visualise_iteration(abd_info=None, con_info=None, title='', abd_info_=None, con_info_=None,
                        issize=False, conf_='switch', ismetric=False, metric='cnt'):
    insts = set()
    for i, info in enumerate([abd_info, con_info, abd_info_, con_info_]):
        if info is not None:
            if len(insts) == 0:
                insts = set(info['stats'].keys())
            else:
                insts = insts.intersection(info['stats'].keys())

    insts = sorted(insts, key=lambda l: int(l[4:]))

    all_ = []
    for inst in insts:

        labels, nof_iters, res = [], 0, []

        if abd_info is not None:
            abd_stats = abd_info['stats'][inst]
            time_list = [abd_stats['expltimes'], abd_stats['coextimes']]
            if metric in ('error', 'kl'):
                expl_list = [abd_stats[metric + 's-expl'], abd_stats['coexes']]#, abd_stats['errors-coex']]
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                expl_list = [abd_stats['coefs-expl'][m1], abd_stats['coexes']]#:, abd_stats['coefs-coex'][m1]]
            else:
                expl_list = [abd_stats['expls-'], abd_stats['coexes']]
            res_, times = merge_list(time_list, expl_list, issize=issize, ismetric=ismetric)
            nof_iters = max(nof_iters, len(times))

            if metric in ('error', 'kl') or 'coef' in metric:
                res.extend(res_[0:1])
                labels.extend(['axpenum-axp'])
            else:
                res.extend(res_)
                labels.extend(['axpenum-axp', 'axpenum-cxp'])

        if con_info is not None:
            con_stats = con_info['stats'][inst]
            time_list = [con_stats['expltimes'], con_stats['coextimes']]
            if metric in ('error', 'kl'):
                expl_list = [con_stats['expls-'], con_stats[metric + 's-coex']]
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                expl_list = [con_stats['expls-'], con_stats['coefs-coex'][m1]]
            else:
                expl_list = [con_stats['expls-'], con_stats['coexes']]
            res_, times = merge_list(time_list, expl_list, issize=issize, ismetric=ismetric)
            nof_iters = max(nof_iters, len(times))
            if metric in ('error', 'kl') or 'coef' in metric:
                res.extend(res_[1:])
                labels.extend(['cxpenum-axp'])
            else:
                res.extend(res_)
                labels.extend(['cxpenum-cxp', 'cxpenum-axp'])

        if abd_info_ is not None:
            abd_stats_ = abd_info_['stats'][inst]
            time_list = [abd_stats_['expltimes'], abd_stats_['coextimes']]
            if metric in ('error', 'kl'):
                expl_list = [abd_stats_[metric + 's-expl'], abd_stats_['coexes']]
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                expl_list = [abd_stats_['coefs-expl'][m1], abd_stats_['coexes']]
            else:
                expl_list = [abd_stats_['expls-'], abd_stats_['coexes']]

            res_, times = merge_list(time_list, expl_list, issize=issize, ismetric=ismetric)
            nof_iters = max(nof_iters, len(times))
            if metric in ('error', 'kl') or 'coef' in metric:
                res.extend(res_[0:])
                labels.extend(['{}-axpenum-axp'.format(conf_)])
            else:
                res.extend(res_)
                labels.extend(['{}-axpenum-axp'.format(conf_), '{}-axpenum-cxp'.format(conf_)])

        if con_info_ is not None:
            con_stats_ = con_info_['stats'][inst]
            time_list = [con_stats_['expltimes'], con_stats_['coextimes']]
            if metric in ('error', 'kl'):
                expl_list = [con_stats_['expls-'], con_stats_[metric + 's-coex']]
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                expl_list = [con_stats_['expls-'], con_stats_['coefs-coex'][m1]]
            else:
                expl_list = [con_stats_['expls-'], con_stats_['coexes']]
            res_, times = merge_list(time_list, expl_list, issize=issize, ismetric=ismetric)
            nof_iters = max(nof_iters, len(times))
            if metric in ('error', 'kl') or 'coef' in metric:
                res.extend(res_[1:])
                labels.extend(['{}-cxpenum-axp'.format(conf_)])
            else:
                res.extend(res_)
                labels.extend(['{}-cxpenum-cxp'.format(conf_), '{}-cxpenum-axp'.format(conf_)])

        for i in range(len(res)):
            if len(res[i]) < nof_iters:
                res[i] += [res[i][-1]] * (nof_iters - len(res[i]))
            assert len(res[i]) == nof_iters

        # replace inf by the max value
        res = np.array(res)
        #res[res == float('inf')] = res[res != float('inf')].max()
        res[res == float('inf')] = 0.5#res[res != float('inf')].max()

        if metric != 'cnt':
            saved_file = title + '-{}-{}'.format(metric, inst) + '-m'
        else:
            saved_file = title + '-{}'.format(inst) + '-m'
        line_chart(res, list(range(1, nof_iters + 1)), labels, title=title + '-{}'.format(inst),
                   saved_file=saved_file, xlabel='iters', ylabel=metric)
        all_.append(saved_file + '.pdf')
    return all_

def cnt_time_nor_switch(files, conf, conf_):
    """ Count over time - normal + switch (cxpenum)"""
    pdf_files = []
    for abd_file in files:
        con_file = abd_file.replace('-abd', '-con')
        with open(abd_file, 'r') as f:
            abd_info = json.load(f)
        with open(con_file, 'r') as f:
            con_info = json.load(f)
        abd_file_ = abd_file.replace('formal', 'formal-{}'.format(conf_))
        con_file_ = con_file.replace('formal', 'formal-{}'.format(conf_))

        # try:
        #if 'map' in conf_:
        #    with open(abd_file_, 'r') as f:
        #        abd_info_ = json.load(f)
        #else:
        abd_info_ = None

        #try:
        with open(con_file_, 'r') as f:
            con_info_ = json.load(f)
        #except:
        #    con_info_ = None

        title = abd_file.rsplit('/', maxsplit=1)[-1].replace('-abd', '').rsplit('.', maxsplit=1)[0]
        pdf = visualise(abd_info, con_info, title, abd_info_, con_info_, conf_=conf_)
        pdf_files.extend(pdf)

    saved_dir = './temp/{}-{}/{}-expl'.format(conf, conf_, conf)
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    merged_file = '{}/{}-{}.pdf'.format(saved_dir, conf, conf_)
    # DeprecationWarning: PdfMerger is deprecated and will be
    # removed in pypdf 5.0.0. Use PdfWriter instead.
    with PdfMerger() as merger:
        for pdf in pdf_files:
            merger.append(pdf)
        merger.write(merged_file)

def metric_time_nor_switch(files, conf, conf_, metric='cnt', isiter=False, xlog=False):
    pdf_files = []
    for abd_file in files:
        con_file = abd_file.replace('-abd', '-con')
        if abd_file.endswith('.json'):
            with open(abd_file, 'r') as f:
                abd_info = json.load(f)
            with open(con_file, 'r') as f:
                con_info = json.load(f)
        elif abd_file.endswith('.xz'):
            with lzma.open(abd_file, 'rb') as f:
                abd_info = json.load(f)
            with lzma.open(con_file, 'rb') as f:
                con_info = json.load(f)
        abd_file_ = abd_file.replace('formal', 'formal-{}'.format(conf_))
        con_file_ = con_file.replace('formal', 'formal-{}'.format(conf_))

        # try:
        #if 'map' in conf_:
        #    with open(abd_file_, 'r') as f:
        #        abd_info_ = json.load(f)
        #else:
        abd_info_ = None

        #try:
        if con_file_.endswith('.json'):
            with open(con_file_, 'r') as f:
                con_info_ = json.load(f)
        elif con_file_.endswith('.xz'):
            with lzma.open(con_file_, 'rb') as f:
                con_info_ = json.load(f)
        else:
            print('something wrong')
            exit(1)
        #except:
        #    con_info_ = None

        title = abd_file.rsplit('/', maxsplit=1)[-1].replace('-abd', '').rsplit('.', maxsplit=1)[0]
        if isiter:
            pdf = visualise_iteration(abd_info, con_info, title, abd_info_, con_info_,
                            ismetric=True, conf_=conf_, metric=metric)
        else:
            pdf = visualise(abd_info, con_info, title, abd_info_, con_info_,
                            ismetric=True, conf_=conf_, metric=metric, xlog=xlog)
        pdf_files.extend(pdf)

    saved_dir = './temp/{}-{}/{}-metric'.format(conf, conf_, conf)
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    merged_file = '{}/{}-{}-{}{}{}.pdf'.format(saved_dir, conf, conf_, metric, '-iter' if isiter else '',
                                               '-xlog' if xlog else '')
    # DeprecationWarning: PdfMerger is deprecated and will be
    # removed in pypdf 5.0.0. Use PdfWriter instead.
    with PdfMerger() as merger:
        for pdf in pdf_files:
            merger.append(pdf)
        merger.write(merged_file)

def cnt_iter_nor_switch(files, conf, conf_):
    pdf_files = []
    for abd_file in files:
        con_file = abd_file.replace('-abd', '-con')
        with open(abd_file, 'r') as f:
            abd_info = json.load(f)
        with open(con_file, 'r') as f:
            con_info = json.load(f)
        abd_file_ = abd_file.replace('formal', 'formal-{}'.format(conf_))
        con_file_ = con_file.replace('formal', 'formal-{}'.format(conf_))

        # try:
        #    with open(abd_file_, 'r') as f:
        #        abd_info_ = json.load(f)
        # except:
        #    abd_info_ = None
        abd_info_ = None
        try:
            with open(con_file_, 'r') as f:
                con_info_ = json.load(f)
        except:
            con_info_ = None
        title = abd_file.rsplit('/', maxsplit=1)[-1].replace('-abd', '').rsplit('.', maxsplit=1)[0]
        pdf = visualise_iteration(abd_info, con_info, title, abd_info_, con_info_, conf_=conf_)
        pdf_files.extend(pdf)

    saved_dir = './temp/{}-{}/{}-expl'.format(conf, conf_, conf)
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    merged_file = '{}/{}-{}-iter.pdf'.format(saved_dir, conf, conf_)
    # DeprecationWarning: PdfMerger is deprecated and will be
    # removed in pypdf 5.0.0. Use PdfWriter instead.
    with PdfMerger() as merger:
        for pdf in pdf_files:
            merger.append(pdf)
        merger.write(merged_file)

def totsize_time_nor_switch(files, conf, conf_):
    pdf_files = []
    for abd_file in files:
        con_file = abd_file.replace('-abd', '-con')
        with open(abd_file, 'r') as f:
            abd_info = json.load(f)
        with open(con_file, 'r') as f:
            con_info = json.load(f)
        abd_file_ = abd_file.replace('formal', 'formal-{}'.format(conf_))
        con_file_ = con_file.replace('formal', 'formal-{}'.format(conf_))
        # try:
        #    with open(abd_file_, 'r') as f:
        #        abd_info_ = json.load(f)
        # except:
        #    abd_info_ = None
        abd_info_ = None
        try:
            with open(con_file_, 'r') as f:
                con_info_ = json.load(f)
        except:
            con_info_ = None
        title = abd_file.rsplit('/', maxsplit=1)[-1].replace('-abd', '').rsplit('.', maxsplit=1)[0] + '-size'
        pdf = visualise(abd_info, con_info, title, abd_info_, con_info_, issize=True, accumulate=True,
                        conf_=conf_)
        pdf_files.extend(pdf)

    saved_dir = './temp/{}-{}/{}-size'.format(conf, conf_, conf)
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    merged_file = '{}/{}-{}-totsize.pdf'.format(saved_dir, conf, conf_)
    # DeprecationWarning: PdfMerger is deprecated and will be
    # removed in pypdf 5.0.0. Use PdfWriter instead.
    with PdfMerger() as merger:
        for pdf in pdf_files:
            merger.append(pdf)
        merger.write(merged_file)


def visualise_derivative(abd_info=None, con_info=None, title='', abd_info_=None, con_info_=None, issize=False,
                         k=25, conf_='switch'):
    insts = set()
    for i, info in enumerate([abd_info, con_info, abd_info_, con_info_]):
        if info is not None:
            if len(insts) == 0:
                insts = set(info['stats'].keys())
            else:
                insts = insts.intersection(info['stats'].keys())

    insts = sorted(insts, key=lambda l: int(l[4:]))

    all_ = []
    for inst in insts:
        time_list, expl_list, labels = [], [], []

        if abd_info is not None:
            abd_stats = abd_info['stats'][inst]
            time_list.extend([abd_stats['expltimes'], abd_stats['coextimes']])
            expl_list.extend([abd_stats['expls-'], abd_stats['coexes']])
            labels.extend(['axpenum-axp', 'axpenum-cxp'])

        if con_info is not None:
            con_stats = con_info['stats'][inst]
            time_list.extend([con_stats['expltimes'], con_stats['coextimes']])
            expl_list.extend([con_stats['expls-'], con_stats['coexes']])
            labels.extend(['cxpenum-cxp', 'cxpenum-axp'])

        if abd_info_ is not None:
            abd_stats_ = abd_info_['stats'][inst]
            time_list.extend([abd_stats_['expltimes'], abd_stats_['coextimes']])
            expl_list.extend([abd_stats_['expls-'], abd_stats_['coexes']])
            labels.extend(['{}-axpenum-axp'.format(conf_), '{}-axpenum-cxp'.format(conf_)])

        if con_info_ is not None:
            con_stats_ = con_info_['stats'][inst]
            time_list.extend([con_stats_['expltimes'], con_stats_['coextimes']])
            expl_list.extend([con_stats_['expls-'], con_stats_['coexes']])
            labels.extend(['{}-cxpenum-cxp'.format(conf_), '{}-cxpenum-axp'.format(conf_)])

        derivative_list = list(map(lambda l: sliding_win(l, k=k), time_list))

        res, times = merge_list(time_list, derivative_list, issize, derivative=True)

        new_saved_file = line_chart(res, times, labels, title=title + '-win{}-{}'.format(k, inst),
                                    saved_file=title + '-{}'.format(inst), ylabel='derivative')
        all_.append(new_saved_file + '.pdf')

    return all_

def der_time_nor_switch(files, conf, conf_):
    pdf_files = []
    for abd_file in files:
        con_file = abd_file.replace('-abd', '-con')
        with open(abd_file, 'r') as f:
            abd_info = json.load(f)
        with open(con_file, 'r') as f:
            con_info = json.load(f)
        abd_file_ = abd_file.replace('formal', 'formal-{}'.format(conf_))
        con_file_ = con_file.replace('formal', 'formal-{}'.format(conf_))

        # try:
        #    with open(abd_file_, 'r') as f:
        #        abd_info_ = json.load(f)
        # except:
        #    abd_info_ = None
        abd_info_ = None
        try:
            with open(con_file_, 'r') as f:
                con_info_ = json.load(f)
        except:
            con_info_ = None

        title = abd_file.rsplit('/', maxsplit=1)[-1].replace('-abd', '').rsplit('.', maxsplit=1)[0]
        pdf = visualise_derivative(abd_info, con_info, title, abd_info_, con_info_,
                                   k=2000, conf_=conf_)
        pdf_files.extend(pdf)

    saved_dir = './temp/{}-{}/{}-derivative'.format(conf, conf_, conf)
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    merged_file = '{}/{}-{}.pdf'.format(saved_dir, conf, conf_)
    # DeprecationWarning: PdfMerger is deprecated and will be
    # removed in pypdf 5.0.0. Use PdfWriter instead.
    with PdfMerger() as merger:
        for pdf in pdf_files:
            merger.append(pdf)
        merger.write(merged_file)


def visualise_derivative_iteration(abd_info=None, con_info=None, title='', abd_info_=None, con_info_=None,
                                   issize=False, k=25, conf_='switch'):
    insts = set()
    for i, info in enumerate([abd_info, con_info, abd_info_, con_info_]):
        if info is not None:
            if len(insts) == 0:
                insts = set(info['stats'].keys())
            else:
                insts = insts.intersection(info['stats'].keys())

    insts = sorted(insts, key = lambda l: int(l[4:]))

    all_ = []
    for inst in insts:

        labels, nof_iters, res = [], 0, []

        if abd_info is not None:
            abd_stats = abd_info['stats'][inst]
            time_list = [abd_stats['expltimes'], abd_stats['coextimes']]
            expl_list = [abd_stats['expls-'], abd_stats['coexes']]
            res_, times = merge_list(time_list, expl_list, iteration=True)
            res.extend(res_)
            nof_iters = max(nof_iters, len(times))
            labels.extend(['axpenum-axp', 'axpenum-cxp'])

        if con_info is not None:
            con_stats = con_info['stats'][inst]
            time_list = [con_stats['expltimes'], con_stats['coextimes']]
            expl_list = [con_stats['expls-'], con_stats['coexes']]
            res_, times = merge_list(time_list, expl_list, issize, iteration=True)
            res.extend(res_)
            nof_iters = max(nof_iters, len(times))
            labels.extend(['cxpenum-cxp', 'cxpenum-axp'])

        if abd_info_ is not None:
            abd_stats_ = abd_info_['stats'][inst]
            time_list = [abd_stats_['expltimes'], abd_stats_['coextimes']]
            expl_list = [abd_stats_['expls-'], abd_stats_['coexes']]
            res_, times = merge_list(time_list, expl_list, issize, iteration=True)
            res.extend(res_)
            nof_iters = max(nof_iters, len(times))
            labels.extend(['{}-axpenum-axp'.format(conf_), '{}-axpenum-cxp'.format(conf_)])

        if con_info_ is not None:
            con_stats_ = con_info_['stats'][inst]
            time_list = [con_stats_['expltimes'], con_stats_['coextimes']]
            expl_list = [con_stats_['expls-'], con_stats_['coexes']]
            res_, times = merge_list(time_list, expl_list, issize, iteration=True)
            res.extend(res_)
            nof_iters = max(nof_iters, len(times))
            labels.extend(['{}-cxpenum-cxp'.format(conf_), '{}-cxpenum-axp'.format(conf_)])

        for i in range(len(res)):
            if len(res[i]) < nof_iters:
                res[i] += [0] * (nof_iters - len(res[i]))
            assert len(res[i]) == nof_iters

        derivative_list = list(map(lambda l: sliding_win_iter(l, k=k), res))

        new_saved_file = line_chart(derivative_list, list(range(1, nof_iters + 1)), labels, title=title + '-win{}-{}'.format(k, inst),
                   saved_file=title + '-{}'.format(inst), xlabel='iters', ylabel='derivative')
        all_.append(new_saved_file + '.pdf')
    return all_

def der_iter_nor_switch(files, conf, conf_):
    pdf_files = []
    for abd_file in files:
        con_file = abd_file.replace('-abd', '-con')
        with open(abd_file, 'r') as f:
            abd_info = json.load(f)
        with open(con_file, 'r') as f:
            con_info = json.load(f)
        abd_file_ = abd_file.replace('formal', 'formal-{}'.format(conf_))
        con_file_ = con_file.replace('formal', 'formal-{}'.format(conf_))

        # try:
        #    with open(abd_file_, 'r') as f:
        #        abd_info_ = json.load(f)
        # except:
        #    abd_info_ = None
        abd_info_ = None
        try:
            with open(con_file_, 'r') as f:
                con_info_ = json.load(f)
        except:
            con_info_ = None

        title = abd_file.rsplit('/', maxsplit=1)[-1].replace('-abd', '').rsplit('.', maxsplit=1)[0]
        pdf = visualise_derivative_iteration(abd_info, con_info, title, abd_info_, con_info_,
                                             k=2000, conf_=conf_)
        pdf_files.extend(pdf)

    saved_dir = './temp/{}-{}/{}-derivative'.format(conf, conf_, conf)
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    merged_file = '{}/{}-{}-iter.pdf'.format(saved_dir, conf, conf_)
    # DeprecationWarning: PdfMerger is deprecated and will be
    # removed in pypdf 5.0.0. Use PdfWriter instead.
    with PdfMerger() as merger:
        for pdf in pdf_files:
            merger.append(pdf)
        merger.write(merged_file)

def visualise_size_iteration(abd_info=None, con_info=None, title='', abd_info_=None, con_info_=None,
                             issize=False, derivative=False, k=25, conf_='switch'):
    insts = set()
    for i, info in enumerate([abd_info, con_info, abd_info_, con_info_]):
        if info is not None:
            if len(insts) == 0:
                insts = set(info['stats'].keys())
            else:
                insts = insts.intersection(info['stats'].keys())

    insts = sorted(insts, key = lambda l: int(l[4:]))

    all_ = []
    for inst in insts:
        labels, nof_iters, res = [], 0, []

        if abd_info is not None:
            abd_stats = abd_info['stats'][inst]
            time_list = [abd_stats['expltimes'], abd_stats['coextimes']]
            expl_list = [abd_stats['expls-'], abd_stats['coexes']]
            res_, times = merge_list(time_list, expl_list, derivative=derivative,
                                     iteration=True)
            res.extend(res_)
            nof_iters = max(nof_iters, len(times))
            labels.extend(['axpenum-axp', 'axpenum-cxp'])

        if con_info is not None:
            con_stats = con_info['stats'][inst]
            time_list = [con_stats['expltimes'], con_stats['coextimes']]
            expl_list = [con_stats['expls-'], con_stats['coexes']]
            res_, times = merge_list(time_list, expl_list, derivative=derivative,
                                     iteration=True)
            res.extend(res_)
            nof_iters = max(nof_iters, len(times))
            labels.extend(['cxpenum-cxp', 'cxpenum-axp'])

        if abd_info_ is not None:
            abd_stats_ = abd_info_['stats'][inst]
            time_list = [abd_stats_['expltimes'], abd_stats_['coextimes']]
            expl_list = [abd_stats_['expls-'], abd_stats_['coexes']]
            res_, times = merge_list(time_list, expl_list, derivative=derivative,
                                     iteration=True)
            res.extend(res_)
            nof_iters = max(nof_iters, len(times))
            labels.extend(['{}-axpenum-axp'.format(conf_), '{}-axpenum-cxp'.format(conf_)])

        if con_info_ is not None:
            con_stats_ = con_info_['stats'][inst]
            time_list = [con_stats_['expltimes'], con_stats_['coextimes']]
            expl_list = [con_stats_['expls-'], con_stats_['coexes']]
            res_, times = merge_list(time_list, expl_list, derivative=derivative,
                                     iteration=True)
            res.extend(res_)
            nof_iters = max(nof_iters, len(times))
            labels.extend(['{}-cxpenum-cxp'.format(conf_), '{}-cxpenum-axp'.format(conf_)])

        for i in range(len(res)):
            if len(res[i]) < nof_iters:
                res[i] += [0] * (nof_iters - len(res[i]))
            assert len(res[i]) == nof_iters

        derivative_list = list(map(lambda l: sliding_win_size(l, k=k), res))

        new_saved_file = line_chart(derivative_list, list(range(1, nof_iters + 1)), labels, title=title + '-win{}-{}'.format(k, inst),
                                    saved_file=title + '-{}'.format(inst), xlabel='iters', ylabel='size derivative')
        all_.append(new_saved_file + '.pdf')
    return all_

def size_iter_nor_switch(files, conf, conf_):
    pdf_files = []
    for abd_file in files:
        con_file = abd_file.replace('-abd', '-con')
        with open(abd_file, 'r') as f:
            abd_info = json.load(f)
        with open(con_file, 'r') as f:
            con_info = json.load(f)
        abd_file_ = abd_file.replace('formal', 'formal-{}'.format(conf_))
        con_file_ = con_file.replace('formal', 'formal-{}'.format(conf_))

        # try:
        #    with open(abd_file_, 'r') as f:
        #        abd_info_ = json.load(f)
        # except:
        #    abd_info_ = None
        abd_info_ = None
        try:
            with open(con_file_, 'r') as f:
                con_info_ = json.load(f)
        except:
            con_info_ = None

        title = abd_file.rsplit('/', maxsplit=1)[-1].replace('-abd', '').rsplit('.', maxsplit=1)[0]
        pdf = visualise_size_iteration(abd_info, con_info, title, abd_info_, con_info_, derivative=True,
                                 k=2000, conf_=conf_)
        pdf_files.extend(pdf)

    saved_dir = './temp/{}-{}/{}-size'.format(conf, conf_, conf)
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    merged_file = '{}/{}-{}-size-iter.pdf'.format(saved_dir, conf, conf_)
    # DeprecationWarning: PdfMerger is deprecated and will be
    # removed in pypdf 5.0.0. Use PdfWriter instead.
    with PdfMerger() as merger:
        for pdf in pdf_files:
            merger.append(pdf)
        merger.write(merged_file)

def metric_switch_avg(files, conf, conf_, metric='cnt', isiter=False,
                      normalise=False, normalise2=False):
    if normalise2:
        normalise = True

    pdf_files = []
    times, reses = [], []
    dt_files = []

    record = {'conf': conf,
              'conf_': conf_,
              'metric': metric,
              'normalise': normalise,
              'normalise2': normalise2,
              'datasets': {}}

    for abd_file in files:
        con_file = abd_file.replace('-abd', '-con')

        if abd_file.endswith('.json'):
            with open(abd_file, 'r') as f:
                abd_info = json.load(f)
            with open(con_file, 'r') as f:
                con_info = json.load(f)
        elif abd_file.endswith('.xz'):
            with lzma.open(abd_file, 'rb') as f:
                abd_info = json.load(f)
            with lzma.open(con_file, 'rb') as f:
                con_info = json.load(f)
        else:
            print('something wrong')
            exit(1)
        abd_file_ = abd_file.replace('formal', 'formal-{}'.format(conf_))
        con_file_ = con_file.replace('formal', 'formal-{}'.format(conf_))

        # try:
        #if 'map' in conf_:
        #    with open(abd_file_, 'r') as f:
        #        abd_info_ = json.load(f)
        #else:
        abd_info_ = None

        #try:
        if con_file_.endswith('.json'):
            with open(con_file_, 'r') as f:
                con_info_ = json.load(f)
        elif con_file_.endswith('.xz'):
            with lzma.open(con_file_, 'rb') as f:
                con_info_ = json.load(f)
        else:
            print('something wrong')
            exit(1)
        #except:
        #    con_info_ = None

        title = abd_file.rsplit('/', maxsplit=1)[-1].replace('-abd', '').rsplit('.', maxsplit=1)[0]
        if isiter:
            pass
        else:
            dt_times, dt_reses, labels = merge_multi_inst_time(abd_info, con_info, abd_info_, con_info_,
                                                               metric=metric, normalise=normalise, normalise2=normalise2)
            times.extend(dt_times)
            reses.extend(dt_reses)
            dt_reses, dt_times = merge_list(dt_times, dt_reses, ismetric=True)
            assert len(dt_reses) % len(labels) == 0

            avg_res = []
            for i, label in enumerate(labels):
                avg_res.append([])
                for j in range(i, len(dt_reses), len(labels)):
                    cur_res = dt_reses[j]
                    avg_res[-1].append(cur_res)
                avg_res[-1] = np.mean(np.array(avg_res[-1]), axis=0).tolist()

            saved_file = title + '-{}-m'.format(metric)
            avg_res_ = avg_res

            record['datasets'][con_file_] = {'labels': labels,
                                             'res': avg_res_,
                                             'times': dt_times}

            new_saved_file = line_chart(avg_res, dt_times, labels, title=title,
                                        saved_file=saved_file, ylabel=metric)
            dt_files.append(new_saved_file + '.pdf')

    #saved_dir = './temp/{}-{}/{}-metric{}{}'.format(conf, conf_, conf, '-normalise' if normalise else '',
    #                                                '2' if normalise2 else '')
    #if not os.path.isdir(saved_dir):
    #    os.makedirs(saved_dir)

    #"""dataset-level"""
    #merged_file = '{}/{}-{}-{}{}{}{}-dt.pdf'.format(saved_dir, conf, conf_, metric, '-normalise' if normalise else '',
    #                                                '2' if normalise2 else '', '-iter' if isiter else '')
    ## DeprecationWarning: PdfMerger is deprecated and will be
    ## removed in pypdf 5.0.0. Use PdfWriter instead.
    #with PdfMerger() as merger:
    #    for pdf in dt_files:
    #        merger.append(pdf)
    #    merger.write(merged_file)

    """all-dataset-level"""
    if isiter:
        pass
    else:
        reses, times = merge_list(times, reses, ismetric=True)
        assert len(reses) % len(labels) == 0

        avg_res = []
        for i, label in enumerate(labels):
            avg_res.append([])
            for j in range(i, len(reses), len(labels)):
                cur_res = reses[j]
                avg_res[-1].append(cur_res)
            avg_res[-1] = np.mean(np.array(avg_res[-1]), axis=0).tolist()


        title = abd_file.rsplit('/', maxsplit=1)[-1].replace('-abd', '').rsplit('.', maxsplit=1)[0]
        saved_file = title + '-{}'.format(metric)

        #new_saved_file = line_chart(avg_res, times, labels, title=title,
        #                            saved_file=saved_file, ylabel=metric)
        avg_res_ = avg_res

        record['whole'] = {'labels': labels,
                           'res': avg_res_,
                           'times': times}
        saved_dir = './stats_record/'
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)
        res_file = saved_dir + '{}-{}-{}{}{}.json'.format(conf, conf_, metric, '-normalise' if normalise else '',
                                         '-normalise2' if normalise2 else '')
        #print('res_file:', res_file)
        with open(res_file, 'w') as f:
            json.dump(record, f, indent=4)

def merge_multi_inst_iter(abd_info=None, con_info=None, abd_info_=None,
                          con_info_=None, metric='cnt'):
    insts = set()
    for i, info in enumerate([abd_info, con_info, abd_info_, con_info_]):
        if info is not None:
            if len(insts) == 0:
                insts = set(info['stats'].keys())
            else:
                insts = insts.intersection(info['stats'].keys())

    insts = sorted(insts, key=lambda l: int(l[4:]))

    dt_reses = []
    for inst in insts:
        labels, nof_iters, res = [], 0, []

        if abd_info is not None:
            abd_stats = abd_info['stats'][inst]
            time_list = [abd_stats['expltimes'], abd_stats['coextimes']]
            if metric in ('error', 'kl'):
                expl_list = [abd_stats[metric + 's-expl'], abd_stats['coexes']]  # , abd_stats['errors-coex']]
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                expl_list = [abd_stats['coefs-expl'][m1], abd_stats['coexes']]  #:, abd_stats['coefs-coex'][m1]]
            else:
                expl_list = [abd_stats['expls-'], abd_stats['coexes']]
            res_, times = merge_list(time_list, expl_list, ismetric=True)
            nof_iters = max(nof_iters, len(times))
            if metric in ('error', 'kl') or 'coef' in metric:
                res.extend(res_[0:1])
                labels.extend(['axpenum-axp'])
            else:
                res.extend(res_)
                labels.extend(['axpenum-axp', 'axpenum-cxp'])

        if con_info is not None:
            con_stats = con_info['stats'][inst]
            time_list = [con_stats['expltimes'], con_stats['coextimes']]
            if metric in ('error', 'kl'):
                expl_list = [con_stats[metric + 's-'], con_stats[metric + 's-coex']]
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                expl_list = [con_stats['expls-'], con_stats['coefs-coex'][m1]]
            else:
                expl_list = [con_stats['expls-'], con_stats['coexes']]
            res_, times = merge_list(time_list, expl_list, ismetric=True)
            nof_iters = max(nof_iters, len(times))
            if metric == 'error' or 'coef' in metric:
                res.extend(res_[1:])
                labels.extend(['cxpenum-axp'])
            else:
                res.extend(res_)
                labels.extend(['cxpenum-cxp', 'cxpenum-axp'])

        if abd_info_ is not None:
            abd_stats_ = abd_info_['stats'][inst]
            time_list = [abd_stats_['expltimes'], abd_stats_['coextimes']]
            if metric == 'error':
                expl_list = [abd_stats_['errors-expl'], abd_stats_['coexes']]
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                expl_list = [abd_stats_['coefs-expl'][m1], abd_stats_['coexes']]
            else:
                expl_list = [abd_stats_['expls-'], abd_stats_['coexes']]

            res_, times = merge_list(time_list, expl_list, ismetric=True)
            nof_iters = max(nof_iters, len(times))
            if metric == 'error' or 'coef' in metric:
                res.extend(res_[0:])
                labels.extend(['{}-axpenum-axp'.format(conf_)])
            else:
                res.extend(res_)
                labels.extend(['{}-axpenum-axp'.format(conf_), '{}-axpenum-cxp'.format(conf_)])

        if con_info_ is not None:
            con_stats_ = con_info_['stats'][inst]
            time_list = [con_stats_['expltimes'], con_stats_['coextimes']]
            if metric == 'error':
                expl_list = [con_stats_['expls-'], con_stats_['errors-coex']]
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                expl_list = [con_stats_['expls-'], con_stats_['coefs-coex'][m1]]
            else:
                expl_list = [con_stats_['expls-'], con_stats_['coexes']]
            res_, times = merge_list(time_list, expl_list, ismetric=True)
            nof_iters = max(nof_iters, len(times))
            if metric == 'error' or 'coef' in metric:
                res.extend(res_[1:])
                labels.extend(['{}-cxpenum-axp'.format(conf_)])
            else:
                res.extend(res_)
                labels.extend(['{}-cxpenum-cxp'.format(conf_), '{}-cxpenum-axp'.format(conf_)])

        for i in range(len(res)):
            if len(res[i]) < nof_iters:
                res[i] += [res[i][-1]] * (nof_iters - len(res[i]))
            assert len(res[i]) == nof_iters
        dt_reses.extend(res)
    for res in dt_reses:
        assert len(res) == nof_iters
    return nof_iters, dt_reses, labels

def merge_multi_inst_time(abd_info=None, con_info=None, abd_info_=None,
                          con_info_=None, metric='cnt', normalise=False,
                          normalise2=False):
    insts = set()
    for i, info in enumerate([abd_info, con_info, abd_info_, con_info_]):
        if info is not None:
            if len(insts) == 0:
                insts = set(info['stats'].keys())
            else:
                insts = insts.intersection(info['stats'].keys())

    insts = sorted(insts, key=lambda l: int(l[4:]))
    times, expls = [], []
    for inst in insts:
        time_list, expl_list, labels = [], [], []

        if abd_info is not None:
            abd_stats = abd_info['stats'][inst]
            if metric in ('error', 'kl'):
                if metric == 'error':
                    cnt_gt = axp_stats(abd_stats['expls-'])
                    temp = [measure_dist({}, cnt_gt, shape=(10,10), metric='manhattan')]
                else:
                    temp = []
                expl_list.append(temp + abd_stats[metric + 's-expl'])#, abd_stats['errors-coex']])
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                cnt_gt = axp_stats(abd_stats['expls-'])
                coef = compare_lists({}, cnt_gt,
                                     metric=m1.replace('-', '_'))
                expl_list.append( [coef] + abd_stats['coefs-expl'][m1])#, abd_stats['coefs-coex'][m1]])
            elif metric == 'nof-expls':
                expl_list.append(abd_stats['expls-'])
            else:
                expl_list.extend([abd_stats['expls-'], abd_stats['coexes']])

            if metric in ('error', 'kl', 'nof-expls') or 'coef' in metric:
                labels.extend(['axpenum-axp'])#, 'axpenum-cxp'])
                if metric == 'error' or 'coef' in metric:
                    temp = [0]
                else:
                    temp = []
                time_list.append(temp + abd_stats['expltimes'])
            else:
                labels.extend(['axpenum-axp', 'axpenum-cxp'])
                time_list.extend([abd_stats['expltimes'], abd_stats['coextimes']])

        if con_info is not None:
            con_stats = con_info['stats'][inst]
            if metric in ('error', 'kl'):
                if metric == 'error':
                    cnt_gt = axp_stats(con_stats['coexes'])
                    cnt = {}
                    temp = [measure_dist(cnt, cnt_gt, shape=(10,10), metric='manhattan')]
                else:
                    temp = []
                #expl_list.extend([con_stats['errors-expl'], con_stats['errors-coex']])
                expl_list.append(temp + con_stats[metric + 's-coex'])
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                cnt_gt = axp_stats(con_stats['coexes'])
                coef = compare_lists({}, cnt_gt,
                                     metric=m1.replace('-', '_'))
                #expl_list.extend([con_stats['coefs-expl'][m1], con_stats['coefs-coex'][m1]])
                expl_list.append([coef] + con_stats['coefs-coex'][m1])
            elif metric == 'nof-expls':
                expl_list.append(con_stats['coexes'])
            else:
                expl_list.extend([con_stats['expls-'], con_stats['coexes']])

            if metric in ('error', 'kl', 'nof-expls') or 'coef' in metric:
                labels.extend(['cxpenum-axp'])
                if metric == 'error' or 'coef' in metric:
                    temp = [0]
                else:
                    temp = []
                time_list.append(temp + con_stats['coextimes'])
            else:
                labels.extend(['cxpenum-cxp', 'cxpenum-axp'])
                time_list.extend([con_stats['expltimes'], con_stats['coextimes']])

        if abd_info_ is not None:
            abd_stats_ = abd_info_['stats'][inst]
            if metric in ('error', 'kl'):
                if metric == 'error':
                    cnt_gt = axp_stats(abd_stats_['expls-'])
                    cnt = {}
                    temp = [measure_dist(cnt, cnt_gt, shape=(10,10), metric='manhattan')]
                else:
                    temp = []
                expl_list.append(temp + abd_stats_[metric + 's-expl'])#, abd_stats_['errors-coex']])
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                cnt_gt = axp_stats(abd_stats_['expls-'])
                coef = compare_lists({}, cnt_gt,
                                     metric=m1.replace('-', '_'))
                expl_list.append([coef] + abd_stats_['coefs-expl'][m1])#, abd_stats_['coefs-coex'][m1]])
            elif metric == 'nof-expls':
                expl_list.append(abd_stats_['expls-'])
            else:
                expl_list.extend([abd_stats_['expls-'], abd_stats_['coexes']])
            if metric in ('error', 'kl', 'nof-expls') or 'coef' in metric:
                labels.extend(['{}-axpenum-axp'.format(conf_)])#, '{}-axpenum-cxp'.format(conf_)])
                if metric == 'error' or 'coef' in metric:
                    temp = [0]
                else:
                    temp = []
                time_list.append(temp + abd_stats_['expltimes'])#, abd_stats_['coextimes']])
            else:
                labels.extend(['{}-axpenum-axp'.format(conf_), '{}-axpenum-cxp'.format(conf_)])
                time_list.extend([abd_stats_['expltimes'], abd_stats_['coextimes']])

        if con_info_ is not None:
            con_stats_ = con_info_['stats'][inst]
            if metric in ('error', 'kl'):
                if metric == 'error':
                    cnt_gt = axp_stats(con_stats_['coexes'])
                    cnt = {}
                    temp = [measure_dist(cnt, cnt_gt, shape=(10,10), metric='manhattan')]
                else:
                    temp = []
                #expl_list.extend([con_stats_['errors-expl'], con_stats_['errors-coex']])
                expl_list.append(temp + con_stats_[metric + 's-coex'])
            elif 'coef' in metric:
                m0, m1 = metric.split('-')
                cnt_gt = axp_stats(con_stats_['coexes'])
                coef = compare_lists({}, cnt_gt,
                                     metric=m1.replace('-', '_'))
                #expl_list.extend([con_stats_['coefs-expl'][m1], con_stats_['coefs-coex'][m1]])
                expl_list.append([coef] + con_stats_['coefs-coex'][m1])
            elif metric == 'nof-expls':
                expl_list.append(con_stats_['coexes'])
            else:
                expl_list.extend([con_stats_['expls-'], con_stats_['coexes']])
            if metric in ('error', 'kl', 'nof-expls') or 'coef' in metric:
                labels.extend(['{}-cxpenum-axp'.format(conf_)])
                if metric == 'error' or 'coef' in metric:
                    temp = [0]
                else:
                    temp = []
                time_list.append(temp + con_stats_['coextimes'])
            else:
                labels.extend(['{}-cxpenum-cxp'.format(conf_), '{}-cxpenum-axp'.format(conf_)])
                time_list.extend([con_stats_['expltimes'], con_stats_['coextimes']])

        # replace inf by the max value
        if metric == 'kl':
            expl_list = np.array(expl_list)
            expl_list[expl_list == float('inf')] = expl_list[expl_list != float('inf')].max() * 2

        if normalise:
            time_list = normalisation(time_list, cnt=False)
            if metric == 'nof-expls':
                expl_list = normalisation(expl_list, cnt=True)

        if normalise2 and metric != 'nof-expls':
            expl_list = normalisation(expl_list, cnt=False)

        times.extend(time_list)
        expls.extend(expl_list)

    assert len(times) == len(expls)
    #assert len(times[0]) == len(expls[0]) == len(labels)

    return times, expls, labels

def normalisation(lists, cnt=False):
    if cnt:
        new_lists = []
        for slist in lists:
            new_lists.append([i/len(slist) for i in
                             range(1, len(slist)+1)])
    else:
        lists = np.array(lists)
        maxv = np.array(lists).max()
        #print(maxv)
        #for l in lists:
        #    print(l[-1])
        #print()
        new_lists = lists / maxv
    #for l, ll in zip(lists, new_lists):
    #    print('before')
    #    print(l[:100])
    #    print(l[-100:])
    #    print('after')
    #    print(ll[:100])
    #    print(ll[-100:])
    #    print()
    return new_lists

if __name__ == '__main__':
    for conf in ['mgh']: #sort
        files = sorted(glob.glob('../stats/*{}*.json*'.format(conf)))
        files = list(filter(lambda l: 'abd' in l and 'switch' not in l, files))
        confs_ = []
        diffs = [1]
        gaps = [2]
        for i in [50]:
            base = 'switch-slide{}-stype'.format(i)
            stype = 'ls'
            for gap in gaps:
                b_ = f'{base}-{stype}-gap{gap}'
                for diff in diffs:
                    conf_ = f'{b_}-diff{diff}'
                    confs_.append(conf_)

        metrics = ['error', 'coef-kendalltau', 'coef-rbo', 'kl', 'nof-expls']
        for conf_ in confs_:
            for metric in metrics:
                # integrated - normalised
                normalise2 = metric in ['error', 'nof-expls']
                metric_switch_avg(files, conf, conf_, metric=metric, normalise=True, normalise2=normalise2)
    exit()
