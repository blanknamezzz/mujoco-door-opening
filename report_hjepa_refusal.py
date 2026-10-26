"""Validation-selected refusal experiment, independent physical test figure."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from report_hjepa_holdout import interval


def report(experiment,output):
    training=json.loads((experiment/'training.json').read_text())
    test=json.loads((experiment/'independent_test/heldout.json').read_text())
    replays=json.loads((experiment/'independent_test/replay.json').read_text())
    assert len(replays)==test['test_scene_count'] and all(r['selection_matches'] and r['outcome_matches'] for r in replays)
    methods=training['method_order']
    labels=['Frozen','Frozen\n+offset','Rank','Rank\n+offset','BCE','BCE\n+offset']
    keys=['old_checkpoint','reference_4','reference_2','reference_5','reference_3','reference_6']
    vals=training['validation_methods'];summaries=test['summaries'];count=test['test_scene_count']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,9));x=np.arange(3)
    axes[0,0].bar(x-.18,[vals[m]['correct_task_decisions'] for m in methods[::2]],.36,label='Zero offset',color='#4b88aa')
    axes[0,0].bar(x+.18,[vals[m]['correct_task_decisions'] for m in methods[1::2]],.36,label='Validation-fitted offset',color='#9971ab')
    axes[0,0].set(xticks=x,xticklabels=['Frozen weights','Rank continuation','BCE continuation'],ylim=(0,96),ylabel='Correct decisions /96',title='Development selection (not independent)')
    axes[0,0].legend(fontsize=9)
    scores=[summaries[k]['correct_task_decisions'] for k in keys]
    bounds=[interval(n,count) for n in scores]
    errors=np.array([[n-lo*count,hi*count-n] for n,(lo,hi) in zip(scores,bounds)]).T
    colors=['#9971ab' if n==training['selected_candidate'] else '#4b88aa' for n in methods]
    axes[0,1].bar(np.arange(6),scores,yerr=errors,capsize=3,color=colors)
    for i,n in enumerate(scores):axes[0,1].text(i,3,str(n),ha='center',color='white',weight='bold')
    axes[0,1].set(xticks=np.arange(6),xticklabels=labels,ylim=(0,count),ylabel=f'Correct decisions /{count}',title='Independent test; 95% Wilson intervals')
    x=np.arange(6)
    axes[1,0].bar(x-.18,[summaries[k]['opening_successes'] for k in keys],.36,label='Successful opening',color='#399280')
    axes[1,0].bar(x+.18,[summaries[k]['correct_rejections'] for k in keys],.36,label='Correct refusal',color='#9971ab')
    axes[1,0].set(xticks=x,xticklabels=labels,ylabel='Scene count',title='What makes a correct decision')
    axes[1,0].legend(fontsize=9)
    for offset,metric,label,color in [(-.24,'false_rejections','False refusals','#cc8e50'),(0.,'wrong_side_errors','Wrong side','#b55a30'),(.24,'impossible_attempts','Impossible attempts','#667a91')]:
        values=[summaries[k][metric] if metric!='impossible_attempts' else count-summaries[k]['feasible_scenes']-summaries[k]['correct_rejections'] for k in keys]
        axes[1,1].bar(x+offset,values,.24,label=label,color=color)
    axes[1,1].set(xticks=x,xticklabels=labels,ylabel='Scene count',title='Failure modes (collision can overlap these)')
    axes[1,1].legend(fontsize=9)
    for ax in axes.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    selected_label=labels[methods.index(training['selected_candidate'])].replace('\n',' ')
    fig.suptitle('Refusal supervision - validation selected '+selected_label+'; seed 42',fontsize=15,weight='bold')
    fig.tight_layout(rect=(0,0,1,.95));output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160);plt.close(fig)
    summary=dict(test_scenes=count,selected_candidate=training['selected_candidate'],test_used_for_selection=False,
        methods={n:summaries[k] for n,k in zip(methods,keys)},image=str(output))
    (experiment/'independent_test/report_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.experiment,a.output)
