"""
====================
Experiment Workflow
====================

Handles experiment specifications for run_papers.py
"""

import os
import argparse

# directories
from chemotaxis import EXPERIMENT_OUT_DIR


def make_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def add_arguments(experiments_library):
    parser = argparse.ArgumentParser(description='chemotaxis paper experiments')
    parser.add_argument(
        'experiment_id',
        type=str,
        choices=list(experiments_library.keys()),
        help='experiment id corresponds to figure number from chemotaxis paper')
    return parser.parse_args()


def workflow(experiments_library):
    """
    Execute experiments from the command line
    """

    out_dir = os.path.join(EXPERIMENT_OUT_DIR, 'chemotaxis')
    make_dir(out_dir)

    args = add_arguments(experiments_library)

    if args.experiment_id:
        # retrieve preset experiment
        experiment_id = str(args.experiment_id)
        experiment_type = experiments_library[experiment_id]

        if callable(experiment_type):
            control_out_dir = os.path.join(out_dir, experiment_id)
            make_dir(control_out_dir)
            experiment_type(control_out_dir)
        elif isinstance(experiment_type, list):
            # iterate over list with multiple experiments
            for sub_experiment_id in experiment_type:
                control_out_dir = os.path.join(out_dir, sub_experiment_id)
                make_dir(control_out_dir)
                exp = experiments_library[sub_experiment_id]
                try:
                    exp(control_out_dir)
                except:
                    print('{} experiment failed'.format(sub_experiment_id))
    else:
        print('provide experiment number')
