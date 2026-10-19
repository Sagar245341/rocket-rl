"""Create publication-ready plots and a standalone interactive benchmark report."""
import argparse
import csv
import html as html_tools
import json
import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parent / 'results' / '.matplotlib'
_CACHE.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(_CACHE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
COLORS = {'ppo':'#54cbbb','guidance':'#80a9f8','random':'#f0a66c'}


def build_report(folder,model_dir,update_readme=False):
    folder,model_dir = Path(folder),Path(model_dir)
    rows = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    scenario_order = list(manifest['config']['scenarios'])
    rows.sort(key=lambda row:(row['phase'],scenario_order.index(row['scenario']),row['controller']))
    methods = []
    for phase, checkpoint in manifest.get('checkpoints',{}).items():
        metadata = checkpoint['metadata']
        methods.append(f"Phase {phase}: {metadata.get('method','PPO')}; selected checkpoint at "
                       f"{metadata['checkpoint_timesteps']:,} PPO steps of {metadata['total_timesteps']:,} trained.")
    plt.style.use('dark_background')
    nominal = [r for r in rows if r['scenario']=='nominal']
    def controller_label(phase,controller):
        if controller != 'ppo':
            return controller
        metadata = manifest.get('checkpoints',{}).get(str(phase),{}).get('metadata',{})
        if metadata.get('method') == 'behavior cloning + PPO':
            return 'imitation + PPO' if metadata['checkpoint_timesteps'] else 'imitation'
        return 'PPO'
    labels = [f"P{r['phase']} / {controller_label(r['phase'],r['controller'])}" for r in nominal]
    x = np.arange(len(nominal))
    colors = [COLORS[r['controller']] for r in nominal]
    fig,axes = plt.subplots(3,1,figsize=(12,12),layout='constrained')
    rates = np.array([r['success_rate']*100 for r in nominal])
    axes[0].bar(x,rates,color=colors)
    low = [max(0,r['success_rate']-r['success_ci_low'])*100 for r in nominal]
    high = [max(0,r['success_ci_high']-r['success_rate'])*100 for r in nominal]
    axes[0].errorbar(x,rates,yerr=[low,high],fmt='none',ecolor='white',capsize=3)
    axes[0].set(ylim=(0,110),ylabel='Safe landings (%)',title='Nominal scenarios / 95% Wilson intervals')
    axes[1].bar(x,[r['mean_reward'] for r in nominal],color=colors)
    axes[1].errorbar(x,[r['mean_reward'] for r in nominal],
                     yerr=[r['std_reward'] for r in nominal],fmt='none',ecolor='white',capsize=3)
    axes[1].set(ylabel='Episode return',title='Mean return / error bars show episode standard deviation')
    fuel = [r['successful_mean_fuel'] if r['successful_mean_fuel'] is not None else float('nan') for r in nominal]
    axes[2].bar(x,fuel,color=colors)
    axes[2].set(ylabel='Fuel units',title='Mean fuel used on successful flights only / missing bar = no successes')
    for ax in axes:
        ax.set_xticks(x,labels,rotation=25,ha='right')
        ax.set_xlim(-.6,len(nominal)-.4)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('APOGEE / Controller Benchmark',fontsize=20)
    fig.supxlabel('Fixed selected policies / one training seed per phase / paired evaluation scenarios',fontsize=10)
    fig.savefig(folder/'comparison.png',dpi=160)
    plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for phase in (1,2,3):
        path = model_dir/f'rocket_phase{phase}.monitor.csv'
        if path.exists():
            with path.open(encoding='utf-8') as file:
                next(file)
                flights = list(csv.DictReader(file))
            if flights:
                steps = np.cumsum([int(r['l']) for r in flights])
                rewards = np.array([float(r['r']) for r in flights])
                window = min(20,len(rewards))
                axes[0].plot(steps[window-1:],np.convolve(rewards,np.ones(window)/window,'valid'),label=f'Phase {phase}')
        path = model_dir/f'rocket_phase{phase}.validation.json'
        if path.exists():
            validations = json.loads(path.read_text(encoding='utf-8'))
            axes[1].plot([r['timesteps'] for r in validations],[100*r['success_rate'] for r in validations],marker='o',label=f'Phase {phase}')
    axes[0].set(title='Training return / up to 20-episode moving average',xlabel='Training timesteps',ylabel='Return')
    axes[1].set(title='Held-out validation / checkpoint selection',xlabel='Training timesteps',ylabel='Safe landings (%)',ylim=(-3,103))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.15)
    fig.savefig(folder/'training_progress.png',dpi=160)
    plt.close(fig)

    robust = [r for r in rows if r['phase']==3]
    if robust:
        scenarios = list(dict.fromkeys(r['scenario'] for r in robust))
        fig,ax = plt.subplots(figsize=(11,5),layout='constrained')
        for controller in ('ppo','guidance','random'):
            selected = {r['scenario']:r for r in robust if r['controller']==controller}
            values = [selected[s]['success_rate']*100 if s in selected else float('nan') for s in scenarios]
            ax.plot(scenarios,values,marker='o',label=controller_label(3,controller),color=COLORS[controller])
        ax.set(title='Phase 3 / robustness conditions (separate interventions)',ylabel='Safe landings (%)',ylim=(-3,103))
        ax.legend()
        ax.grid(alpha=.15)
        fig.savefig(folder/'robustness.png',dpi=160)
        plt.close(fig)
    markdown = ['# Measured controller results','',
        '| Phase | Scenario | Controller | Episodes | Safe landings (95% CI) | Mean return +/- std | Successful-flight fuel |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        fuel_text = f"{r['successful_mean_fuel']:.2f}" if r['successful_mean_fuel'] is not None else 'N/A'
        markdown.append(f"| {r['phase']} | {r['scenario']} | {r['controller']} | {r['episodes']} | "
            f"{r['success_rate']:.0%} ({r['success_ci_low']:.1%}-{r['success_ci_high']:.1%}) | "
            f"{r['mean_reward']:.2f} +/- {r['std_reward']:.2f} | {fuel_text} |")
    markdown += ['', '## Training and checkpoint selection', '', *methods, '',
        'A checkpoint selected at step 0 is the imitation warm start before PPO updates; '
        'it must not be described as a policy improved by PPO.', '',
        'Results use paired initial-condition seeds and identical reward weights across controllers. '
        'Benchmark seeds are separate from training and checkpoint-selection seeds. Each PPO phase uses one training seed; '
        'confidence intervals describe evaluated episodes for that fixed policy, not variability across retraining. '
        'Successful-flight fuel is conditional on landing and can involve different subsets of scenarios. '
        'Stress tests are distribution shifts, not aerospace safety certification. Zero observed failures does not establish a 100% population success rate.',
        '', 'Raw data: [episodes.csv](episodes.csv), [summary.csv](summary.csv), [manifest.json](manifest.json).',
        '', 'Open [dashboard.html](dashboard.html) locally for interactive filtering.']
    (folder/'REPORT.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')
    payload = json.dumps(rows).replace('<','\\u003c')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>APOGEE | Measured Results</title><style>
body{margin:0;background:#0b1424;color:#e8f0fa;font:16px system-ui}main{max-width:1200px;margin:auto;padding:36px}
h1{letter-spacing:4px}small,p{color:#a8bbd0}select{padding:10px;background:#1c3048;color:white;border:1px solid #3e5670;border-radius:6px;margin:8px}
.cards{display:flex;gap:16px;flex-wrap:wrap}.card{background:#15273b;padding:20px;border-radius:12px;flex:1;min-width:180px}.card strong{display:block;font-size:30px;color:#54cbbb}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #294058}img{width:100%;border-radius:12px;margin-top:24px}svg{width:100%;height:auto;background:#15273b;border-radius:12px;margin:20px 0}.scroll{overflow:auto}a{color:#54cbbb}</style>
<main><small>REPRODUCIBLE CONTROLLER EVALUATION</small><h1>APOGEE / RESULTS</h1>
<p>Fixed trained policies, paired scenario seeds, and explicit uncertainty. GUIDANCE is a reference controller; PPO is learned.</p>
<p>TRAINING_METHODS A selected checkpoint at step 0 represents imitation before PPO updates, not a PPO improvement.</p>
<label>Phase <select id="phase"><option value="all">All phases</option><option>1</option><option>2</option><option>3</option></select></label>
<label>Scenario <select id="scenario"><option value="all">All conditions</option></select></label>
<div class="cards"><div class="card">Evaluation episodes<strong id="episodes"></strong></div><div class="card">Reported controller groups<strong id="groups"></strong></div><div class="card">Success interval<strong>95% Wilson</strong></div></div>
<svg id="chart" role="img" aria-label="Safe landing rates by controller"></svg><div class="scroll"><table><thead><tr><th>Phase</th><th>Condition</th><th>Controller</th><th>Episodes</th><th>Safe landing rate</th><th>95% interval</th><th>Mean return</th><th>Fuel / safe flight</th></tr></thead><tbody id="rows"></tbody></table></div>
<p>Intervals describe evaluation episodes for one fixed policy, not variability across training seeds. Fuel is conditional on a safe landing; N/A means no successful flights. Phase-3 stresses are separate interventions, not increasing levels of one variable.</p>
<img src="comparison.png" alt="Nominal controller comparison"><img src="training_progress.png" alt="Training and held-out validation progress">
<img src="robustness.png" alt="Phase three robustness comparison" onerror="this.hidden=true">
<p><a href="episodes.csv">Per-episode CSV</a> / <a href="summary.csv">Summary CSV</a> / <a href="manifest.json">Reproducibility manifest</a> / <a href="REPORT.md">Full methodology</a></p></main>
<script>const data=PAYLOAD;const phase=document.querySelector('#phase'),scenario=document.querySelector('#scenario');
for(const s of [...new Set(data.map(r=>r.scenario))]){const o=document.createElement('option');o.value=s;o.textContent=s;scenario.append(o)}
function render(){const filtered=data.filter(r=>(phase.value==='all'||String(r.phase)===phase.value)&&(scenario.value==='all'||r.scenario===scenario.value));
document.querySelector('#episodes').textContent=filtered.reduce((n,r)=>n+r.episodes,0);document.querySelector('#groups').textContent=filtered.length;
const tbody=document.querySelector('#rows');tbody.replaceChildren();
for(const r of filtered){const tr=document.createElement('tr');for(const v of [r.phase,r.scenario,r.controller,r.episodes,(100*r.success_rate).toFixed(1)+'%',(100*r.success_ci_low).toFixed(1)+'–'+(100*r.success_ci_high).toFixed(1)+'%',r.mean_reward.toFixed(2),r.successful_mean_fuel===null?'N/A':r.successful_mean_fuel.toFixed(2)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}tbody.append(tr)}
const svg=document.querySelector('#chart');svg.replaceChildren();svg.setAttribute('viewBox',`0 0 1100 ${filtered.length*36+50}`);const ns='http://www.w3.org/2000/svg';
for(let i=0;i<filtered.length;i++){const r=filtered[i],y=20+i*36;const label=document.createElementNS(ns,'text');label.setAttribute('x','12');label.setAttribute('y',y+18);label.setAttribute('fill','#d6e5f5');label.textContent=`P${r.phase} / ${r.scenario} / ${r.controller}`;svg.append(label);const rect=document.createElementNS(ns,'rect');rect.setAttribute('x','330');rect.setAttribute('y',y);rect.setAttribute('width',String(650*r.success_rate));rect.setAttribute('height','23');rect.setAttribute('fill',({ppo:'#54cbbb',guidance:'#80a9f8',random:'#f0a66c'})[r.controller]);svg.append(rect);const number=document.createElementNS(ns,'text');number.setAttribute('x','990');number.setAttribute('y',y+18);number.setAttribute('fill','white');number.textContent=(100*r.success_rate).toFixed(0)+'%';svg.append(number)}}
phase.onchange=scenario.onchange=render;render();</script></html>'''.replace('PAYLOAD',payload).replace('TRAINING_METHODS',html_tools.escape(' '.join(methods)))
    (folder/'dashboard.html').write_text(html,encoding='utf-8')
    if update_readme:
        readme = ROOT/'README.md'
        content = readme.read_text(encoding='utf-8')
        start,end = '<!-- BENCHMARK_START -->','<!-- BENCHMARK_END -->'
        if start not in content or end not in content:
            raise ValueError('README benchmark markers are missing')
        lines = ['| Phase | Controller | Episodes | Safe landings (95% CI) | Mean return | Fuel / successful flight |',
                 '| --- | --- | --- | --- | --- | --- |']
        for row in nominal:
            controller = row['controller']
            if controller == 'ppo':
                metadata = manifest['checkpoints'][str(row['phase'])]['metadata']
                if metadata.get('method') == 'behavior cloning + PPO':
                    controller = 'Imitation + PPO' if metadata['checkpoint_timesteps'] else 'Imitation (step 0)'
            fuel_text = f"{row['successful_mean_fuel']:.2f}" if row['successful_mean_fuel'] is not None else 'N/A'
            lines.append(f"| {row['phase']} | {controller} | {row['episodes']} | "
                         f"{row['success_rate']:.0%} ({row['success_ci_low']:.1%}-{row['success_ci_high']:.1%}) | "
                         f"{row['mean_reward']:+.2f} | {fuel_text} |")
        lines += ['', 'Nominal conditions; one training seed per phase. Intervals condition on the fixed selected policy. '
                  'The learned policies use reference-generated imitation labels before PPO updates. '
                  'Successful-flight fuel compares potentially different successful subsets.', '', *methods]
        prefix = content.split(start,1)[0]
        suffix = content.split(end,1)[1]
        readme.write_text(prefix+start+'\n'+'\n'.join(lines)+'\n'+end+suffix,encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'docs/benchmark')
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models')
    parser.add_argument('--update-readme',action='store_true')
    args = parser.parse_args()
    build_report(args.input,args.model_dir,args.update_readme)

