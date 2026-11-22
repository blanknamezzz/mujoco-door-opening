"""Build the four report figures from preserved measurements and camera frames."""
from pathlib import Path
import csv
import hashlib
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parents[1]
WORK=ROOT/'runs/final_report_consistency_20261122'
WORK.mkdir(parents=True,exist_ok=True)
INPUTS={}

def remember(path):
    INPUTS[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def read(path):
    return json.loads(remember(path).read_text(encoding='utf-8'))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold',
    'axes.labelcolor':'#26394b','text.color':'#183044','xtick.color':'#26394b',
    'ytick.color':'#26394b','axes.edgecolor':'#657484'})
BLUE,TEAL,INK,GRAY,PALE='#326ba0','#21988d','#183044','#8795a3','#eaf2f7'

def save(fig,name,preview):
    fig.savefig(REPORT/name,bbox_inches='tight',metadata={'CreationDate':None})
    fig.savefig(WORK/(preview+'.png'),dpi=190,bbox_inches='tight')
    plt.close(fig)

def arrow(ax,a,b,color=GRAY,style='-|>',lw=1,**kw):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=10,
        linewidth=lw,color=color,**kw))

# Figure 1: recorded configuration and two measured continuation signals.
case=ROOT/'results/workspace_limit'
diagnosis=read(case/'workspace_limit_diagnosis.json')
with np.load(remember(case/'workspace_limit.npz'),allow_pickle=False) as data:
    qpos=data['qpos'].copy();times=data['time'].copy()
with remember(case/'workspace_limit.csv').open(encoding='utf-8',newline='') as stream:
    rows=list(csv.DictReader(stream))
assert len(rows)==len(times)
np.testing.assert_allclose(times,[float(r['time_s']) for r in rows])
opened=np.array([r['phase']=='open' for r in rows])
assert all(r['bilateral_contact']=='True' for r in rows if r['phase']=='open')
door=np.rad2deg(qpos[:,0])
margin=np.rad2deg(qpos[:,3])-diagnosis['joint_limits_deg'][1][0]
assert abs(door[-1]-diagnosis['actual_door_angle_deg'])<1e-7
assert abs(margin[-1]-1.4109706623038676)<1e-7
scene=REPORT/'figures/failures/report_joint_limit_scene.png'
if not scene.exists():
    raise FileNotFoundError('Run scripts/render_diagnostic_frame.py with MuJoCo first.')
fig=plt.figure(figsize=(6.5,2.75))
grid=fig.add_gridspec(2,2,width_ratios=[1.02,1.23],hspace=.42,wspace=.34)
image=fig.add_subplot(grid[:,0]);top=fig.add_subplot(grid[0,1]);bottom=fig.add_subplot(grid[1,1],sharex=top)
image.imshow(plt.imread(scene),interpolation='none');image.axis('off')
image.set_title('(a) Fixed-base diagnostic',loc='left',fontsize=10,pad=7)
image.text(.5,-.045,'Bilateral contact maintained',transform=image.transAxes,
    ha='center',fontsize=9,color=TEAL)
top.plot(times[opened],door[opened],color=BLUE,lw=2)
top.axhline(90,color=GRAY,lw=.9,ls='--')
top.text(9.62,90,'Target',ha='right',va='bottom',fontsize=8,color='#5c6d7d')
top.annotate(r'$42.51^\circ$',(times[-1],door[-1]),xytext=(-5,8),
    textcoords='offset points',ha='right',fontsize=9,color=BLUE)
top.set(ylim=(-3,103),yticks=[0,45,90],ylabel='Door angle (deg)')
top.set_title('(b) Opening progress',loc='left',fontsize=10,pad=7)
bottom.plot(times[opened],margin[opened],color=TEAL,lw=2)
bottom.axhline(0,color=GRAY,lw=.9,ls='--')
bottom.annotate(r'$1.41^\circ$',(times[-1],margin[-1]),xytext=(7,12),
    textcoords='offset points',ha='left',fontsize=9,color=TEAL)
bottom.set(xlim=(3.35,9.75),ylim=(-4,82),yticks=[0,40,80],
    xlabel='Simulated time (s)',ylabel='Margin (deg)')
bottom.set_title('(c) Remaining elbow travel',loc='left',fontsize=10,pad=7)
for ax in [top,bottom]:
    ax.axvline(times[-1],color='#b1bdc8',lw=.8,ls=':')
    ax.tick_params(labelsize=9,length=2)
    ax.yaxis.label.set_size(9);ax.xaxis.label.set_size(9)
plt.setp(top.get_xticklabels(),visible=False)
fig.subplots_adjust(left=.015,right=.985,top=.88,bottom=.17)
save(fig,'figures/failures/report_joint_limit.pdf','figure1_joint_limit')

# Figure 2: shared encoding, sequential decisions, and staged target construction.
fig,ax=plt.subplots(figsize=(6.5,3.05))
ax.set_xlim(0,6.5);ax.set_ylim(0,3.12);ax.axis('off')
ax.add_patch(FancyBboxPatch((.05,2.46),4.74,.56,
    boxstyle='round,pad=.02,rounding_size=.045',ec=BLUE,fc=PALE,lw=1))
ax.text(2.42,2.82,'Shared H-JEPA visual encoder',ha='center',va='center',
    fontsize=10,fontweight='bold')
ax.text(2.42,2.60,'Frozen core + rank-8 LoRA',ha='center',va='center',fontsize=9)
ax.text(5.68,2.84,'Current RGB',ha='center',fontsize=9,color=BLUE)
arrow(ax,(5.08,2.73),(4.84,2.73),color=BLUE)
ax.text(5.68,2.51,'At each stage',ha='center',va='center',fontsize=8.5,color='#5c6d7d')
labels=[('I: inspection','Inspect / direct',.05),('P: positioning','Station / refusal',1.7),
        ('G: grasping','Two grasp angles',3.35),('Controller','Prepare + open',5.0)]
for title,detail,x in labels:
    color=BLUE if title=='Controller' else TEAL
    ax.add_patch(FancyBboxPatch((x,1.66),1.43,.56,
        boxstyle='round,pad=.02,rounding_size=.04',ec=color,fc=PALE if color==BLUE else '#eaf5f2',lw=1))
    ax.text(x+.715,2.04,title,ha='center',va='center',fontsize=9.5,fontweight='bold')
    ax.text(x+.715,1.81,detail,ha='center',va='center',fontsize=9)
for _,_,x in labels[:3]:arrow(ax,(x+.715,2.43),(x+.715,2.25),color=BLUE)
for (_,_,x),(_,_,nxt) in zip(labels,labels[1:]):arrow(ax,(x+1.46,1.94),(nxt-.035,1.94))
ax.text(3.25,1.40,'Inspect if selected; navigate and reobserve before G',ha='center',fontsize=9,color='#5c6d7d')
ax.text(.05,1.14,'SUPERVISION',fontsize=9,fontweight='bold',color=TEAL)
for j,(title,detail) in enumerate([('G targets','Executed outcomes'),('P targets','Current G choice'),('I targets','Frozen P/G + cost')]):
    x=.05+2.20*j
    ax.add_patch(FancyBboxPatch((x,.48),1.92,.52,
        boxstyle='round,pad=.02,rounding_size=.04',ec=TEAL,fc='white',lw=1,ls='--'))
    ax.text(x+.96,.84,title,ha='center',va='center',fontsize=9.5,fontweight='bold',color=TEAL)
    ax.text(x+.96,.62,detail,ha='center',va='center',fontsize=9)
for x in [.05,2.25]:arrow(ax,(x+1.95,.74),(x+2.15,.74),color=TEAL)
ax.text(3.25,.12,'Stage 1: train LoRA + P/G    |    Stage 2: freeze encoder/P/G; train I',
    ha='center',fontsize=8.8,color=INK)
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
save(fig,'figures/methods/report_system.pdf','figure2_system')

# Figure 3: first front-blocked scene in the selected policy's latest test cohort.
latest=ROOT/'runs/mobile_partial_workspace_seed42_20261109/runs/p10t_seed42_20261111'
report=read(latest/'report.json');manifest=read(latest/'manifest.json')
spec=next(s for s in manifest['scenes'] if s['split']=='test' and s['layout']=='front_blocked')
replay=latest/'policy_results'/report['primary_policy']/'replays'/spec['scene_id']
trace=read(replay/'trace.json');record=read(replay/'result.json')['record']
with np.load(remember(latest/'data'/spec['scene_id']/'1/observations.npz'),allow_pickle=False) as data:
    initial=data['initial_rgb'].copy();reached=data['post_rgb'].copy()
    initial_state=data['initial_proprio'].copy();post_state=data['post_proprio'].copy()
assert initial.shape==reached.shape==(224,224,3)
assert record['success'] and record['inspection']['success'] and record['base_reset_count']==1
def camera_pose(state):
    return np.array([state[14],state[15],np.arctan2(state[16],state[17])])
np.testing.assert_allclose(camera_pose(initial_state),trace[0]['base_pose'],atol=1e-6)
np.testing.assert_allclose(camera_pose(post_state),record['inspection']['actual_base_pose'],atol=1e-6)
fig=plt.figure(figsize=(6.5,2.65))
path=fig.add_axes([.065,.18,.26,.70])
pre=fig.add_axes([.375,.21,.275,.66]);post=fig.add_axes([.695,.21,.275,.66])
for phase,color,label in [('inspection',BLUE,'Inspection'),('navigation',TEAL,'Navigation')]:
    xy=np.array([t['base_pose'][:2] for t in trace if t['phase']==phase])
    path.plot(xy[:,0],xy[:,1],color=color,lw=2,label=label,zorder=3)
cad=ET.fromstring(remember(latest/'source/scene.xml').read_text(encoding='utf-8'))
panel=cad.find("worldbody/body[@name='door']/geom[@name='door_panel']")
center=np.fromstring(panel.get('pos'),sep=' ');half=np.fromstring(panel.get('size'),sep=' ')
door_span=[float(center[0]-half[0]),float(center[0]+half[0])]
path.plot(door_span,[center[1],center[1]],color='#596b7d',lw=4,zorder=1)
for obstacle in spec['obstacles']:
    if obstacle['shape']!='box':continue
    x,y,_=obstacle['pos'];sx,sy,*_=obstacle['size'];angle=np.deg2rad(obstacle['euler_deg'][2])
    corners=np.array([[-sx,-sy],[sx,-sy],[sx,sy],[-sx,sy]])
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    path.fill(*(corners@rotation.T+[x,y]).T,color='#c4cbd3',alpha=.9,zorder=0)
start=np.array(trace[0]['base_pose'][:2]);scan=np.array(record['inspection']['actual_base_pose'][:2])
end=np.array(trace[-1]['base_pose'][:2])
path.scatter(*start,s=38,fc='white',ec=BLUE,zorder=5)
path.scatter(*scan,s=38,marker='D',color=BLUE,zorder=5)
path.scatter(*end,s=30,marker='s',color=TEAL,zorder=5)
path.annotate('A',start,xytext=(9,-1),textcoords='offset points',fontsize=10,fontweight='bold',color=BLUE)
path.annotate('B',scan,xytext=(9,-1),textcoords='offset points',fontsize=10,fontweight='bold',color=BLUE)
path.annotate('Opening\nstation',end,xytext=(9,-6),textcoords='offset points',fontsize=8,color=TEAL)
path.set(xlim=(-.65,1.35),ylim=(-1.85,1.85),xticks=[0,1],yticks=[-1,0,1],aspect='equal',
    xlabel='World x (m)',ylabel='World y (m)')
path.tick_params(labelsize=8,length=2);path.xaxis.label.set_size(9);path.yaxis.label.set_size(9)
fig.text(.025,.945,'(a) Recorded base path',fontsize=10,fontweight='bold')
for axis,rgb,x,title in [(pre,initial,.375,'(b) A: initial view'),(post,reached,.695,'(c) B: inspection view')]:
    axis.imshow(rgb,interpolation='none');axis.axis('off')
    fig.text(x,.945,title,fontsize=10,fontweight='bold')
fig.legend(*path.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.52,-.035),
    ncol=2,frameon=False,fontsize=9)
save(fig,'figures/methods/report_active_motion.pdf','figure3_active_motion')

# Figure 4: one matched test cohort, three inspection modes.
v10=ROOT/'runs/mobile_partial_workspace_seed42_20261109/runs/p10t_seed42_20261111'
metrics=read(v10/'report.json')['test']
modes=[('frozen_v9_never','Never inspect','#687b8d','o'),('standard','Learned inspection',TEAL,'D'),
       ('frozen_v9_always','Always inspect',BLUE,'o')]
fig,ax=plt.subplots(figsize=(6.5,2.65))
values={}
for key,label,color,marker in modes:
    row=metrics[key];x=row['mean_duration_s'];y=100*row['correct_decisions']/row['scenes']
    assert row['scenes']==72
    ax.scatter(x,y,s=75,color=color,marker=marker,edgecolor='white',linewidth=.9,zorder=4)
    if key=='frozen_v9_never':offset,align=(10,8),'left'
    else:offset,align=(-4,10),'right'
    ax.annotate(f'{label}\n{row["correct_decisions"]}/72; {x:.2f} s',(x,y),xytext=offset,
        textcoords='offset points',ha=align,fontsize=9,color=color)
    values[key]=dict(duration_s=x,correct_decisions=row['correct_decisions'],accuracy_percent=y)
learned=values['standard'];always=values['frozen_v9_always']
reduction=100*(1-learned['duration_s']/always['duration_s'])
assert learned['correct_decisions']==always['correct_decisions']==70
ax.hlines(learned['accuracy_percent'],learned['duration_s'],always['duration_s'],
          color='#b6c1cb',ls=':',lw=1,zorder=1)
arrow(ax,(always['duration_s'],90.5),(learned['duration_s'],90.5),color=TEAL,lw=1.2)
ax.text((always['duration_s']+learned['duration_s'])/2,82.4,
    f'{reduction:.1f}% less time\nsame task outcomes',ha='center',fontsize=9,color=TEAL)
ax.set(xlim=(0,65),ylim=(60,112),xticks=[0,10,20,30,40,50,60],yticks=[60,70,80,90,100],
    xlabel='Mean episode duration (simulated s)',ylabel='Task correctness (%)')
ax.grid(color='#e4e9ee',lw=.6,zorder=0);ax.set_axisbelow(True)
ax.tick_params(labelsize=9,length=2)
fig.subplots_adjust(left=.10,right=.985,bottom=.21,top=.91)
save(fig,'figures/results/report_inspection_cost.pdf','figure4_accuracy_time')

for name,expected in INPUTS.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
audit=dict(input_sha256=INPUTS,figure1=dict(stop_time_s=float(times[-1]),door_deg=float(door[-1]),
    elbow_margin_deg=float(margin[-1]),bilateral_contact_during_opening=True),
    figure3=dict(cohort='C',archive_id='v10',policy=report['primary_policy'],
        scene_id=spec['scene_id'],selection_rule='first test scene with front_blocked layout',
        camera_poses_match_replay=True,
        path_m=record['measured_path_length_m'],
        episode_duration_s=record['episode_duration_s'],door_span_m=door_span,camera_shape=list(initial.shape),
        initial_rgb_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        reached_rgb_sha256=hashlib.sha256(reached.tobytes()).hexdigest()),
    figure4=dict(cohort='C',archive_id='v10',modes=values,time_reduction_percent=reduction),
    original_inputs_unchanged=True)
(WORK/'figure_data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k!='input_sha256'},indent=2))
