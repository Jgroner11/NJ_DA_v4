"""Run the whole pipeline for the session parameters.yaml names, in order:

    embedding_and_labels.m   bin the spikes and write the per-bin labels
    behaviour_plot.m         the behaviour figure
    run_umap.py              fit the embeddings
    main.py                  the interactive plots and the clip sweep

    python pipeline.py
    python pipeline.py --from umap     # start partway, the earlier outputs kept
    python pipeline.py --end 600       # passed on to main.py, for a quick run

The two MATLAB scripts run in one MATLAB session, so the .mat file is loaded
once: behaviour_plot.m finds it already in the workspace. The Python scripts run
under this same interpreter, so they see whatever environment pipeline.py does.

Each step has to succeed before the next starts; the first failure stops the run.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (name for --from, script, language), in the order they run
STEPS = [
    ('labels', 'embedding_and_labels.m', 'matlab'),
    ('behaviour', 'behaviour_plot.m', 'matlab'),
    ('umap', 'run_umap.py', 'python'),
    ('clips', 'main.py', 'python'),
]
NAMES = [name for name, _, _ in STEPS]

parser = argparse.ArgumentParser(description='Run every step for the session in parameters.yaml.')
parser.add_argument('--from', dest='start', choices=NAMES, default=NAMES[0],
                    help='the step to start at (default: the first)')
parser.add_argument('--end', type=float, default=None,
                    help="passed to main.py: only clips starting before this time (s)")
args = parser.parse_args()


def run(command, label):
    print(f'\n=== {label} ===', flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        sys.exit(f'{label} failed (exit code {result.returncode}); stopping')
    print(f'=== {label} done in {time.perf_counter() - started:.0f} s ===', flush=True)


def batches(steps):
    """Consecutive MATLAB steps grouped together, so they share one MATLAB session."""
    grouped = []
    for _, script, language in steps:
        if language == 'matlab' and grouped and grouped[-1][0] == 'matlab':
            grouped[-1][1].append(script)
        else:
            grouped.append((language, [script]))
    return grouped


for language, scripts in batches(STEPS[NAMES.index(args.start):]):
    if language == 'matlab':
        # -sd so the scripts' relative paths resolve from the project root,
        # whatever MATLAB's own startup folder is set to
        body = '; '.join(Path(script).stem for script in scripts)
        run(['matlab', '-sd', str(ROOT), '-batch', body], ' + '.join(scripts))
    else:
        (script,) = scripts
        extra = ['--end', str(args.end)] if script == 'main.py' and args.end is not None else []
        run([sys.executable, script, *extra], script)

print('\npipeline finished')
