'''
====================
Chemotaxis Experiments
====================

Chemotaxis provides several pre-configured :py:class:`Experiments`
with different chemotactic agents and environments.
'''

from __future__ import absolute_import, division, print_function

import os
import copy
import sys
import argparse
import numpy as np
from arpeggio import (
    RegExMatch,
    ParserPython,
    OneOrMore,
)

from vivarium.library.dict_utils import deep_merge
from vivarium.core.emitter import (
    time_indexed_timeseries_from_data,
    timeseries_from_data
)
from vivarium.core.composition import (
    agent_environment_experiment,
    make_agents,
    simulate_experiment,
    plot_agents_multigen,
    process_in_compartment,
    EXPERIMENT_OUT_DIR,
)

# compartments
from vivarium.compartments.static_lattice import StaticLattice
from vivarium.compartments.chemotaxis_minimal import ChemotaxisMinimal
from vivarium.compartments.chemotaxis_master import ChemotaxisMaster
from vivarium.compartments.chemotaxis_flagella import (
    ChemotaxisVariableFlagella,
    ChemotaxisExpressionFlagella,
    ChemotaxisODEExpressionFlagella,
)

# processes
from vivarium.processes.coarse_motor import MotorActivity
from vivarium.processes.multibody_physics import agent_body_config
from vivarium.processes.static_field import make_field

# control
from vivarium.core.control import Control



# make an agent from a lone MotorActivity process
# MotorActivityAgent = MotorActivity
MotorActivityAgent = process_in_compartment(
    MotorActivity,
    topology={
        'external': ('boundary',),
        'internal': ('cell',)})

# environment defaults
DEFAULT_ENVIRONMENT_TYPE = StaticLattice
DEFAULT_LIGAND_ID = 'glc__D_e'  # BiGG id for external glucose
DEFAULT_BOUNDS = [1000, 3000]
DEFAULT_AGENT_LOCATION = [0.5, 0.1]

# exponential field parameters
FIELD_SCALE = 4.0
EXPONENTIAL_BASE = 1.3e2
FIELD_CENTER = [0.5, 0.0]

# get initial ligand concentration based on location
LOC_DX = (DEFAULT_AGENT_LOCATION[0] - FIELD_CENTER[0]) * DEFAULT_BOUNDS[0]
LOC_DY = (DEFAULT_AGENT_LOCATION[1] - FIELD_CENTER[1]) * DEFAULT_BOUNDS[1]
DIST = np.sqrt(LOC_DX ** 2 + LOC_DY ** 2)
INITIAL_LIGAND = FIELD_SCALE * EXPONENTIAL_BASE ** (DIST / 1000)


def get_exponential_env_config():
    # multibody process config
    multibody_config = {
        'bounds': DEFAULT_BOUNDS}

    # static field config
    field_config = {
        'molecules': [DEFAULT_LIGAND_ID],
        'gradient': {
            'type': 'exponential',
            'molecules': {
                DEFAULT_LIGAND_ID: {
                    'center': FIELD_CENTER,
                    'scale': FIELD_SCALE,
                    'base': EXPONENTIAL_BASE}}},
        'bounds': DEFAULT_BOUNDS}

    return {
        'multibody': multibody_config,
        'field': field_config}

def get_linear_env_config():
    # field parameters
    slope = 1.0
    base = 1e-1
    field_center = [0.5, 0.0]

    # multibody process config
    multibody_config = {
        'animate': False,
        'bounds': DEFAULT_BOUNDS}

    # static field config
    field_config = {
        'molecules': [DEFAULT_LIGAND_ID],
        'gradient': {
            'type': 'linear',
            'molecules': {
                DEFAULT_LIGAND_ID: {
                    'base': base,
                    'center': field_center,
                    'slope': slope,
                }
            }
        },
        'bounds': DEFAULT_BOUNDS}

    return {
        'multibody': multibody_config,
        'field': field_config}

DEFAULT_ENVIRONMENT_CONFIG = {
    'type': DEFAULT_ENVIRONMENT_TYPE,
    'config': get_exponential_env_config()}

DEFAULT_AGENT_CONFIG = {
    'ligand_id': DEFAULT_LIGAND_ID,
    'initial_ligand': INITIAL_LIGAND,
    'external_path': ('global',),
    'agents_path': ('..', '..', 'agents'),
    'daughter_path': tuple()}

def set_environment_config(config={}):
    # override default environment config
    return deep_merge(dict(DEFAULT_ENVIRONMENT_CONFIG), config)

def set_agent_config(config={}):
    # override default agent config
    return deep_merge(dict(DEFAULT_AGENT_CONFIG), config)


# configs with faster timescales, to support close agent/environment coupling
FAST_TIMESCALE = 0.001
tumble_jitter = 4000
# fast timescale minimal agents
FAST_MOTOR_CONFIG = set_agent_config({
        'tumble_jitter': tumble_jitter,
        'time_step': FAST_TIMESCALE})
FAST_MINIMAL_CHEMOTAXIS_CONFIG = set_agent_config({
    'receptor': {
        'time_step': FAST_TIMESCALE},
    'motor': {
        'tumble_jitter': tumble_jitter,
        'time_step': FAST_TIMESCALE}})

# fast timescale environment
FAST_TIMESCALE_ENVIRONMENT_CONFIG = set_environment_config({
    'config': {
        'multibody': {'time_step': FAST_TIMESCALE},
        'field': {'time_step': FAST_TIMESCALE} }})

# agent types
compartment_library = {
    'motor': {
        'name': 'motor',
        'type': MotorActivityAgent,
        'config': FAST_MOTOR_CONFIG,
    },
    'minimal': {
        'name': 'minimal',
        'type': ChemotaxisMinimal,
        'config': FAST_MINIMAL_CHEMOTAXIS_CONFIG,
    },
    'variable': {
        'name': 'variable',
        'type': ChemotaxisVariableFlagella,
        'config': DEFAULT_AGENT_CONFIG
    },
    'expression': {
        'name': 'expression',
        'type': ChemotaxisExpressionFlagella,
        'config': DEFAULT_AGENT_CONFIG
    },
    'ode': {
        'name': 'ode_expression',
        'type': ChemotaxisODEExpressionFlagella,
        'config': DEFAULT_AGENT_CONFIG
    },

    # Environments
    'chemotaxis_environment': {
        'name': 'chemotaxis_environment',
        'type': DEFAULT_ENVIRONMENT_TYPE,
        'config': get_exponential_env_config()
    }
}

# preset experimental configurations
experiment_library = {
    'minimal': {
        'agents_config': [
            {
                'number': 6,
                'name': 'minimal',
                'type': ChemotaxisMinimal,
                'config': DEFAULT_AGENT_CONFIG,
            }
        ],
        'environment_config': DEFAULT_ENVIRONMENT_CONFIG,
        'simulation_settings': {
            'total_time': 30,
            'emit_step': 0.1,
        },
    },
    'ode': {
        'agents_config': [
            {
                'number': 1,
                'name': 'ode_expression',
                'type': ChemotaxisODEExpressionFlagella,
                # 'config': DEFAULT_AGENT_CONFIG,
                'config': deep_merge(
                    dict(DEFAULT_AGENT_CONFIG),
                    {'growth_rate': 0.0005})  # fast growth
            }
        ],
        'environment_config': DEFAULT_ENVIRONMENT_CONFIG,
        'simulation_settings': {
            'total_time': 50,
            'emit_step': 1.0,
        },
    },
    'master': {
        'agents_config': [
            {
                'number': 1,
                'name': 'master',
                'type': ChemotaxisMaster,
                'config': DEFAULT_AGENT_CONFIG
            }
        ],
        'environment_config': DEFAULT_ENVIRONMENT_CONFIG,
        'simulation_settings': {
            'total_time': 30,
            'emit_step': 0.1,
        },
    },
    'variable': {
        'agents_config': [
            {
                'number': 1,
                'name': '{}_flagella'.format(n_flagella),
                'type': ChemotaxisVariableFlagella,
                'config': set_agent_config({'n_flagella': n_flagella})
            }
            for n_flagella in [0, 3, 6, 9, 12]
        ],
        'environment_config': DEFAULT_ENVIRONMENT_CONFIG,
        'simulation_settings': {
            'total_time': 720,
            'emit_step': 0.1,
        },
    },
    'motor': {
        'agents_config': [
            {
                'type': MotorActivityAgent,
                'name': 'motor',
                'number': 1,
                'config': FAST_MOTOR_CONFIG,
            }
        ],
        'environment_config': FAST_TIMESCALE_ENVIRONMENT_CONFIG,
        'simulation_settings': {
            'total_time': 120,
            'emit_step': FAST_TIMESCALE,
        },
    },
    'fast_minimal': {
        'agents_config': [
            {
                'number': 1,
                'name': 'motor + receptor',
                'type': ChemotaxisMinimal,
                'config': FAST_MINIMAL_CHEMOTAXIS_CONFIG,
            }
        ],
        'environment_config': FAST_TIMESCALE_ENVIRONMENT_CONFIG,
        'simulation_settings': {
            'total_time': 60,
            'emit_step': FAST_TIMESCALE,
        },
    },
    'mixed': {
        'agents_config': [
            {
                'type': ChemotaxisMinimal,
                'name': 'motor + receptor',
                'number': 1,
                'config': FAST_MINIMAL_CHEMOTAXIS_CONFIG,
            },
            {
                'type': MotorActivityAgent,
                'name': 'motor',
                'number': 1,
                'config': FAST_MOTOR_CONFIG,
            }
        ],
        'environment_config': FAST_TIMESCALE_ENVIRONMENT_CONFIG,
        'simulation_settings': {
            'total_time': 300,
            'emit_step': FAST_TIMESCALE,
        },
    },
    'many_mixed': {
        'agents_config': [
            {
                'type': MotorActivityAgent,
                'name': 'motor',
                'number': 5,
                'config': FAST_MOTOR_CONFIG,
            },
            {
                'type': ChemotaxisMinimal,
                'name': 'motor + receptor',
                'number': 5,
                'config': FAST_MINIMAL_CHEMOTAXIS_CONFIG,
            },
        ],
        'environment_config': FAST_TIMESCALE_ENVIRONMENT_CONFIG,
        'simulation_settings': {
            'total_time': 300,
            'emit_step': FAST_TIMESCALE*10,
        },
    },
}




def main():
    workflow = Control(
        compartment_library=compartment_library,
        experiment_library=experiment_library
        )

    workflow.execute()


if __name__ == '__main__':
    main()