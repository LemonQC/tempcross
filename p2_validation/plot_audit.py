"""Plot existing compressed P2-A time series; no simulator or policy execution."""
import argparse
import csv
import gzip
import json
from pathlib import Path


def plot(summary_path, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    episodes = json.loads(Path(summary_path).read_text())['episodes']
    cases = ['joint_reps/ur5e/000', 'joint_reps/panda/000',
             'joint_reps/panda/001', 'no_reps/panda/004']
    fig, axes = plt.subplots(len(cases), 1, figsize=(11, 10), sharex=True)
    for ax, case in zip(axes, cases):
        episode = next(r for r in episodes if r['case_id'] == case)
        with gzip.open(episode['external_timeseries_path'], 'rt') as stream:
            rows = list(csv.DictReader(stream))
        t = [int(r['pre_action_index']) for r in rows]
        ax.plot(t, [float(r['raw_gripper']) for r in rows], label='Raw gripper g', color='#255ca8')
        ax.axhline(.5, color='#a33a36', linestyle='--', label='Official close threshold')
        ax.plot(t, [(float(r['environment_gripper'])+1)/2 for r in rows],
                alpha=.6, color='#8d5eaa', label='Executed close command (0/1)')
        contact = [float(r['bilateral_contact'] == 'True') for r in rows]
        ax.fill_between(t, 0, contact, alpha=.15, color='green', label='Bilateral contact')
        ax.set(ylim=(-.1,1.3), ylabel='Gripper signal',
               title=f'{case} | recorded official success={episode["success"]}')
        ax.grid(alpha=.2)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel('Pre-action state / action index (legacy development episodes)')
    fig.suptitle('Offline reconstruction only: no new P2 physical trials', fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    if Path(out).suffix == '.svg':
        path = Path(out)
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    plot(args.summary, args.out)
