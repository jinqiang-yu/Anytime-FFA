import os
import sys
import glob
import json
import lzma
import re
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
import matplotlib.pyplot as plt

def find_dtname(file):
    file = file.lower()
    dtname = None
    if 'pneumonia' in file:
        dtname = 'pneumoniamnist'
    elif 'mnist' in file:
        if '1,7' in file:
            dtname = 'mnist-1v7'
        elif '1,3' in file:
            dtname = 'mnist-1v3'
    elif 'sarcasm' in file:
        dtname = 'sarcasm'
    elif 'disaster' in file:
        dtname = 'disaster'
    return dtname

def line_chart(y, x, labels, title, level='inst', saved_file=None, rotation=0,
               xlabel='Normalized Time', ylabel=None, xlog=False, ylog=False,
               switches=None, slice=1, linewidth=1, fontsize=12, metric=None,
               loc_='right', zwidth=2, zheight=2, loc1=1, loc2=3, x1=0, x2=.1,
              y1=0, y2=1, bbox_to_anchor=(1,1), ylim=None):
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': fontsize})
    if metric == 'kl':
        for v in y:
            v0 = v[0]
            for i, vv in enumerate(v):
                if vv == v0:
                    v[i] = 0.5
                else:
                    break
          
    if 'expl' in metric:
        x = [0] + x
        y_ = []
        for i, v in enumerate(y):
            y_.append([0] + v)
        y = y_
            
    # Change the style of plot
    #plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, ax = plt.subplots()
    # Create a color palette
    palette = plt.get_cmap('Set1')
    #palette = plt.get_cmap('Accent')
    
    

    if switches is None:
        switches = [[]] * len(y)

    # Plot multiple lines
    num = 1
    if labels:
        for label, v, switch in zip(labels, y, switches):
            ax.plot(x, v, marker='', color=palette(num),
                     linewidth=linewidth, alpha=0.75, label=label)
            num += 1
            if len(switch) > 0:
                sp = v[len(list(filter(lambda l: l <= switch[0], x)))-1]
                ax.plot(switch[0], sp, marker="o", markersize=5, markeredgecolor="orange",
                         markerfacecolor="orange")
    else:
         for v, switch in zip(y, switches):
            ax.plot(x, v, marker='', color=palette(num),
                     linewidth=linewidth, alpha=0.75)
            num += 1
            if len(switch) > 0:
                sp = v[len(list(filter(lambda l: l <= switch[0], x)))-1]
                ax.plot(switch[0], sp, marker="o", markersize=5, markeredgecolor="orange",
                         markerfacecolor="orange")
    
    if ylim:
        ax.set_ylim(ylim)
    #    ax.set_ylim(.4)
    #elif 'tau' in metric:
    #    ax.set_ylim(-.08)
    
    # Add legend
    if labels:
        if level == 'inst':
            fontsize_ = fontsize * 0.7
        else:
            fontsize_ = fontsize * 0.5
        if metric == 'kl':
            #ax.legend(labels, loc='lower right', ncol=1, fancybox=True, shadow=True,
            #          bbox_to_anchor=(1, 0.12), fontsize=fontsize * 0.8)
            ax.legend(labels, loc='upper center', ncol=3, fancybox=True, shadow=True,
                       fontsize=fontsize_, bbox_to_anchor=(0.5, 1.1))
        elif 'rbo' in metric or 'tau' in metric:
            #ax.legend(labels, loc='lower right', bbox_to_anchor=(1, 0.1),
            #          ncol=1, fancybox=True, shadow=True, fontsize=fontsize * 0.8)
            ax.legend(labels, loc='upper center', ncol=3, fancybox=True, shadow=True,
                       fontsize=fontsize_, bbox_to_anchor=(0.5, 1.1))
            
        elif 'expl' in metric:
            ax.legend(labels, loc='upper center', ncol=3, fancybox=True, shadow=True,
                       fontsize=fontsize_, bbox_to_anchor=(0.5, 1.1))
        elif 'error' in metric:
            #ax.legend(labels, loc='upper right', ncol=1, fancybox=True, shadow=True, fontsize=fontsize * 0.8)
            ax.legend(labels, loc='upper center', ncol=3, fancybox=True, shadow=True,
                       fontsize=fontsize_, bbox_to_anchor=(0.5, 1.1))
        else:
            ax.legend(labels, loc=1, ncol=1, fancybox=True, shadow=True, fontsize=fontsize_)

    # Add titles
    ax.set_title(title, loc='left', fontsize=12, fontweight=0, color='orange')
    ax.set_xlabel(xlabel)
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #xticks = list(filter(lambda l: l < x[-1], [i/10 for i in range(11)]))
    #ax.set_xticks()
    
    #ylog = True
    if xlog:
        #ax.xscale('log', base=10)
        ax.set_xscale('log', base=10)
    if ylog:
        #ax.xscale('log', base=10)
        ax.set_yscale('log', base=10)
        
    if title and 'size' in title:
        ax.set_ylabel("Size")
        if saved_file:
            saved_file = saved_file.replace('-size', '') + '-size'
    elif ylabel is not None:
        ax.set_ylabel(ylabel)
    #else:
    #    plt.ylabel("#Expls")

    #ax.set_xticks(rotation=rotation)
    #ax.set_xticklabels(ax.get_xticks(), rotation=rotation)
    
    
    #if slice != 1:
    #    x_max = max(x)
    #    x_ = list(filter(lambda l: l/x_max <= slice, x))
    #    y_ = [yy[:len(x_)] for yy in y]
    #else:
    x_, y_ = x, y 
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for l in ['left', 'bottom']:
        ax.spines[l].set_linewidth(1.5)
        
    #ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])
    
    '''
    if 'tau' in metric:
        bbox_to_anchor = (0.63, 0.47)
    elif 'rbo' in metric:
        bbox_to_anchor = (0.65, 0.41)
    elif 'error' in metric:
        bbox_to_anchor = (0.63, 0.55)
    elif metric == 'kl':
        bbox_to_anchor = (0.65, 0.6)
    else:
        bbox_to_anchor = (0.73, 0.38)
    '''
    
    #loc_ = 'center'
    axins = inset_axes(ax, width=zwidth, height=zheight, loc=loc_,
                       bbox_transform=ax.figure.transFigure, bbox_to_anchor=bbox_to_anchor)
                       #bbox_transform=bbox_to_anchor)#,
                      #bbox_to_anchor=bbox_to_anchor)#, bbox_transform=ax.figure.transFigure)#, #bbox_to_anchor bbox_to_anchor=(0.2, 0.55),
                      #bbox_transform=ax.figure.transFigure)
    #for l in ['left', 'bottom']:
    #    axins.spines[l].set_linewidth(.5)
    #axins.spines['left'].set_visible(False)
        
    num = 1
    for v in y_:
        axins.plot(x_, v, marker='', color=palette(num),
                 linewidth=linewidth, alpha=0.75)
        num += 1
    
    '''
    if 'tau' in metric:
        x1, x2 = -0.002, .03
        y1, y2 = -.08, .85
    elif 'rbo' in metric:
        x1, x2 = -0.002, .03
        y1, y2 = .4, .8
    elif 'error' in metric:
        x1, x2 = -0.002, .05
        y1, y2 = 0.1, 1
    elif metric == 'kl':
        x1, x2 = -0.002, .025
        y1, y2 = 0.17, 0.29
    else:
        x1, x2 = -0.002, .03
        y1, y2 = 0, 0.38
    '''
    
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    

    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.5", linestyle='--')
    
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    
    #saved_file = None
    print('saved_file:', saved_file)
    
    # Show the graph
    if saved_file:
        plt.savefig(saved_file + '.pdf', bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()
    return saved_file

def gnrt_dt_plots(info):
    if info['metric'] in ['error']:
        dtname2slice = {'mnist-1v3': 0.3,
                    'mnist-1v7': 0.6,
                    'pneumoniamnist': 0.1,
                    'sarcasm': 0.1,
                    'disaster': 0.1}
    elif info['metric'] in ['coef-kendalltau']:
        dtname2slice = {'mnist-1v3': 0.85, #0.7,
                    'mnist-1v7': 0.7,
                    'pneumoniamnist': 0.1,
                    'sarcasm': 0.1,
                    'disaster': 0.1}
    elif info['metric'] in ['coef-rbo']:
        dtname2slice = {'mnist-1v3': 0.85, #0.7,
                    'mnist-1v7': 0.6,
                    'pneumoniamnist': 0.1,
                    'sarcasm': 0.1,
                    'disaster': 0.1}
    elif info['metric'] in ['kl']:
        dtname2slice = {'mnist-1v3': 0.8, #0.7,
                    'mnist-1v7': 0.8,
                    'pneumoniamnist': 0.2,
                    'sarcasm': 0.25,
                    'disaster': 0.25}
    elif info['metric'] in ['nof-expls']:
        dtname2slice = {'mnist-1v3': 0.1,
                    'mnist-1v7': 0.6,
                    'pneumoniamnist': 0.1,
                    'sarcasm': 0.1,
                    'disaster': 0.1}
    
    loc_ = 'center'
    loc1=1
    loc2=3
    ylim = None
    
    datasets = list(info['datasets'].keys())[:]
    for dt_id, dt_file in enumerate(datasets):
        #print(info['datasets'][dt_file].keys())
        print(dt_file)
        dtname = find_dtname(dt_file)
        print(dtname)
        
        metric = info['metric']
        if 'error' in metric:
            zwidth=3#2
            zheight=2.5#2
            if dtname == 'mnist-1v3':
                x1, x2 = -0.002, .25
                #y1, y2 = 0.05, 1
                y1, y2 = 0.2, 1
                bbox_to_anchor = (0.63, 0.55)
            elif dtname == 'mnist-1v7':
                x1, x2 = -0.002, .6
                #y1, y2 = 0, 1
                y1, y2 = 0.2, 1
                bbox_to_anchor = (0.63, 0.55)
            elif 'pneu' in dtname:
                x1, x2 = -0.002, .07
                #y1, y2 = 0.15, 1
                y1, y2 = 0.3, 1
                bbox_to_anchor = (0.63, 0.55)
            elif dtname == 'sarcasm':
                x1, x2 = 0, .005
                y1, y2 = 0, 1
                bbox_to_anchor = (0.63, 0.55)
            else:
                x1, x2 = 0, .03
                y1, y2 = 0, 1
                bbox_to_anchor = (0.63, 0.55)
        elif 'tau' in metric:
            zwidth=3 #2
            zheight=2.5 #2
            if dtname == 'mnist-1v3':
                x1, x2 = 0, .55
                y1, y2 = -.2, .7
                bbox_to_anchor = (0.63, 0.47)
            elif dtname == 'mnist-1v7':
                zheight=2.3
                x1, x2 = 0, .55
                y1, y2 = -.15, .63
                bbox_to_anchor = (0.63, 0.43)
            elif 'pneu' in dtname:
                x1, x2 = 0, .015
                y1, y2 = 0, .7
                bbox_to_anchor = (0.63, 0.47)
            elif dtname == 'sarcasm':
                x1, x2 = 0, .002
                y1, y2 = .22, 1
                bbox_to_anchor = (0.63, 0.47)
            else:
                x1, x2 = -0.002, .03
                y1, y2 = -.15, 1
                bbox_to_anchor = (0.63, 0.47)
        elif 'rbo' in metric:
            zwidth=2.8#2
            zheight=2.1#2
            if dtname == 'mnist-1v3':
                zheight=2.5
                x1, x2 = 0, .55
                y1, y2 = .47, .83
                bbox_to_anchor = (0.63, 0.44)
            elif dtname == 'mnist-1v7':
                zheight=2.3
                x1, x2 = 0, .7
                y1, y2 = .53, .78
                bbox_to_anchor = (0.66, 0.41)
            elif 'pneu' in dtname:
                zheight=1.9
                x1, x2 = 0, .015
                y1, y2 = 0.44, .78
                bbox_to_anchor = (0.66, 0.41)
            elif dtname == 'sarcasm':
                zheight=1.9
                x1, x2 = 0, .0025
                y1, y2 = .3, 1
                bbox_to_anchor = (0.63, 0.37)
            else:
                zheight=1.9
                ylim = 0.4
                x1, x2 = 0, .015
                y1, y2 = .4, .9
                bbox_to_anchor = (0.66, 0.4)        
        elif metric == 'kl':
            zwidth=2.8#2
            zheight=2.1#2
            if dtname == 'mnist-1v3':
                zheight=1.7
                x1, x2 = .7, .9
                y1, y2 = .2, .55
                bbox_to_anchor = (0.63, .65)
            elif dtname == 'mnist-1v7':
                zwidth=1.7
                zheight=2.2
                x1, x2 = .4, .73
                y1, y2 = .26, .55
                bbox_to_anchor = (0.72, .62)
            elif 'pneu' in dtname:
                zheight=1.7
                zwidth=2.4
                x1, x2 = 0, .1
                y1, y2 = .12, .55
                bbox_to_anchor = (0.7, .65)
            elif dtname == 'sarcasm':
                zheight= 2.5
                x1, x2 = 0, 0.08
                y1, y2 = 0, .55
                bbox_to_anchor = (0.63, .55)
            else:
                zheight= 2.5
                x1, x2 = 0, 0.08
                y1, y2 = 0, .55
                bbox_to_anchor = (0.63, .55)
        else:
            zwidth=3#2
            zheight=2.5#2
            if dtname == 'mnist-1v3':
                zwidth=2
                zheight=1.7
                x1, x2 = 0, .05
                y1, y2 = 0, .08
                bbox_to_anchor = (0.7, 0.37)
            elif dtname == 'mnist-1v7':
                zwidth=2
                zheight=1.7
                x1, x2 = 0, .22
                y1, y2 = 0, .065
                bbox_to_anchor = (0.7, 0.37)
            elif 'pneu' in dtname:
                zwidth=2
                zheight=1.7
                x1, x2 = 0.0005, .005
                y1, y2 = 0, .1
                bbox_to_anchor = (0.7, 0.37)
            elif dtname == 'sarcasm':
                zwidth=2
                zheight=1.7
                x1, x2 = 0, .0015
                y1, y2 = 0, 1.02
                bbox_to_anchor = (0.7, 0.37)
            else:
                zwidth=2
                zheight=1.7
                x1, x2 = 0, .0021
                y1, y2 = 0, .7
                bbox_to_anchor = (0.7, 0.37)
        
        x1, x2 = 0, .03
        
        res = info['datasets'][dt_file]['res']
        times = info['datasets'][dt_file]['times']
        labels = info['datasets'][dt_file]['labels']
        if info['normalise2']:
            #ylabel = info['metric'].capitalize() + '~Proportion'
            ylabel = 'Normalized ' + info['metric'].capitalize()
        else:
            ylabel = info['metric'].capitalize()
        
        ylabel = ylabel.replace('Coef-', '').replace('rbo', 'RBO')
        ylabel = ylabel.replace('Kl', 'KL Divergence')
        ylabel = ylabel.replace('kendalltau', r"Kendall's Tau")
        #ylabel = ylabel.replace('Nof-expls', r"Nof. Explantions")
        ylabel = ylabel.replace('Nof-expls', r"Nof. Expls")
        
        
        if info['normalise']:
            #xlabel = r'\textbf{Time~Proportion}'
            xlabel = r'\textbf{Normalized Time}'
        else:
            xlabel = r'\textbf{Time~(s)}'
        #if dt_id == 0:
        #    ylabel = r'\textbf{' + ylabel + r'}'
        #else:
        #    ylabel = None
        ylabel = r'\textbf{' + ylabel + r'}'
            
        saved_dir = '../plots/'
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)

        saved_file = f'{dtname}-{conf_}-{"normalise2" if info["normalise2"] else ""}-{info["metric"]}'.replace("stype-ls-", "")
        saved_file = saved_dir + saved_file + '.pdf' #+ '-' + dtname 
        saved_file = saved_file.replace('.pdf', '').replace('--', '-')
        print(saved_file)
        #labels = [r'\textbf{MARCO}$_{axp}$', r'\textbf{MARCO}$_{cxp}$', r'\textbf{MARCO}$_{switch}$']
        #if dt_id == 0:
        #    #labels = [r'MARCO$_{axp}$', r'MARCO$_{cxp}$', r'MARCO$_{switch}$']
        #    labels = [r'MARCO-A', r'MARCO-C', r'MARCO-S']
        #else:
        #    labels = None
        labels = [r'MARCO-A', r'MARCO-C', r'MARCO-S']
        title = None
        #saved_file = None
        line_chart(y=res, x=times, labels=labels, title=title, level='dt', saved_file=saved_file, 
                   ylabel=ylabel, xlabel=xlabel, slice=dtname2slice[dtname], linewidth=2, fontsize=24,
                  metric=info['metric'], loc_=loc_, zwidth=zwidth, zheight=zheight, loc1=loc1, loc2=loc2,
                  x1=x1, x2=x2, y1=y1, y2=y2, bbox_to_anchor=bbox_to_anchor, ylim=ylim)


def gnrt_int_plot(info):
    if info['metric'] in ['error', 'coef-kendalltau', 'coef-rbo', 'kl', 'nof-expls']:
        s = 0.1 
    loc_ = 'center'
    loc1=1
    loc2=3        
    ylim = None
    
    metric = info['metric']
    if 'tau' in metric:
        ylim = 0.08
        zwidth=3 #2
        zheight=2.5 #2
        x1, x2 = -0.002, .03
        y1, y2 = -.08, .85
        bbox_to_anchor = (0.63, 0.47)
    elif 'rbo' in metric:
        ylim = 0.4
        zwidth=2.8#2
        zheight=2.1#2
        x1, x2 = -0.002, .03
        y1, y2 = .4, .8
        bbox_to_anchor = (0.65, 0.43)
    elif 'error' in metric:
        zwidth=3#2
        zheight=2.5#2
        x1, x2 = -0.002, .05
        y1, y2 = 0.1, 1
        bbox_to_anchor = (0.63, 0.55)
    elif metric == 'kl':
        zwidth=2.8#2
        zheight=2#2
        x1, x2 = -0.002, .025
        y1, y2 = 0.17, 0.29
        bbox_to_anchor = (0.65, 0.6)
    else:
        zwidth=2
        zheight=1.8
        x1, x2 = -0.002, .03
        y1, y2 = 0, 0.38
        bbox_to_anchor = (0.73, 0.38)
        
    res = info['whole']['res']
    times = info['whole']['times']
    labels = info['whole']['labels']
    
    if info['normalise2']:
        #ylabel = info['metric'].capitalize() + '~Proportion'
        ylabel = 'Normalized ' + info['metric'].capitalize()
    else:
        ylabel = info['metric'].capitalize()
        
    ylabel = ylabel.replace('Coef-', '').replace('rbo', 'RBO')
    ylabel = ylabel.replace('Kl', 'KL Divergence')
    ylabel = ylabel.replace('kendalltau', r"Kendall's Tau")
    ylabel = ylabel.replace('Nof-expls', r"Nof. Explantions")
    
    if info['normalise']:
        #xlabel = r'\textbf{Time~Proportion}'
        xlabel = r'\textbf{Normalied Time}'
    else:
        xlabel = r'\textbf{Time~(s)}'

    labels = [r'MARCO-A', r'MARCO-C', r'MARCO-S']
    title = None
    saved_dir = '../plots/'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    saved_file = f'{conf_}-{"normalise2" if info["normalise2"] else ""}-{info["metric"]}'.replace("stype-ls-", "")
    saved_file = saved_dir + saved_file + '.pdf'
    saved_file = saved_file.replace('.pdf', '').replace('--', '-')
    #print(saved_file)
    #saved_file = None
    #labels = None
    line_chart(res, times, labels, title=title, saved_file=saved_file, 
               ylabel=ylabel, slice=s, linewidth=2, fontsize=18, metric=info['metric'],
              loc_=loc_, zwidth=zwidth, zheight=zheight, loc1=loc1, loc2=loc2,
              x1=x1, x2=x2, y1=y1, y2=y2, bbox_to_anchor=bbox_to_anchor, ylim=ylim)


if __name__ == '__main__':
    confs_ = []
    diffs = [1]
    for i in [50] :
        gaps = [2]
        base = 'switch-slide{}-stype'.format(i)
        stype = 'ls'
        for gap in gaps:
            b_ = f'{base}-{stype}-gap{gap}'
            for diff in diffs:
                conf_ = f'{b_}-diff{diff}'
                confs_.append(conf_)
                        
    metrics = ['error', 'coef-kendalltau', 'coef-rbo', 'kl', 'nof-expls']

    for conf_ in confs_:
        for metric in metrics[:]:
            slide = re.search('slide\d+-', conf_).group()[5:-1]
            gap = re.search('gap\d+-', conf_).group()[3:-1]
            diff = re.search('-diff\d{1}', conf_).group()[-1]
            stats_file = f'./stats_record/mgh-switch-slide{slide}-stype-ls-gap{gap}-diff{diff}-{metric}'
            if metric in ('error', 'nof-expls'):
                stats_file = stats_file + '-normalise-normalise2.json'
            else:
                stats_file = stats_file + '-normalise.json'
            with open(stats_file, 'r') as f:
                info = json.load(f)
                
            """Generate dataset plots"""
            gnrt_dt_plots(info)
            """Generate integrated plots"""
            gnrt_int_plot(info)
        exit()
