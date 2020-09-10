from __future__ import absolute_import, division, print_function

import os

from vivarium.library.dict_utils import (
    deep_merge
)
from vivarium.library.units import units
from vivarium.core.process import Generator
from vivarium.core.composition import (
    simulate_compartment_in_experiment,
    COMPARTMENT_OUT_DIR)

# processes
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.tree_mass import TreeMass
from chemotaxis.plots.transport_metabolism import analyze_transport_metabolism
from cell.processes.division_volume import DivisionVolume
from cell.processes.metabolism import (
    Metabolism,
    get_iAF1260b_config)
from cell.processes.convenience_kinetics import (
    ConvenienceKinetics,
    get_glc_lct_config)
from cell.processes.ode_expression import (
    ODE_expression,
    get_lacy_config)


NAME = 'transport_metabolism'
TIMESTEP = 5


def default_metabolism_config():
    config = get_iAF1260b_config()

    # set flux bond tolerance for reactions in ode_expression's lacy_config
    metabolism_config = {
        'time_step': TIMESTEP,
        'initial_mass': 1339.0,  # fg of metabolite pools
        'tolerance': {
            'EX_glc__D_e': [1.05, 1.0],
            'EX_lcts_e': [1.05, 1.0]}}
    config.update(metabolism_config)
    return config


def default_expression_config():
    # glc lct config from ode_expression
    config = get_lacy_config()

    # redo regulation with BiGG id for glucose
    regulators = [
        ('external', 'glc__D_e'),
        ('internal', 'lcts_p')]
    regulation = {
        'lacy_RNA': 'if (external, glc__D_e) > 0.005 and (internal, lcts_p) < 0.05'}  # inhibited in this condition
    transcription_leak = {
        'rate': 7e-5,  #5e-5,
        'magnitude': 1e-6}
    reg_config = {
        'time_step': TIMESTEP,
        'regulators': regulators,
        'regulation': regulation,
        'transcription_leak': transcription_leak,
    }
    config.update(reg_config)
    return config


def default_transport_config():
    config = get_glc_lct_config()
    txp_config = {
        'time_step': TIMESTEP,
        'kinetic_parameters': {
            'EX_glc__D_e': {
                ('internal', 'EIIglc'): {
                    ('external', 'glc__D_e'): 2e-1,  # k_m for external [glc__D_e]
                }
            },
            'EX_lcts_e': {
                ('internal', 'LacY'): {
                    ('external', 'lcts_e'): 1e-1,
                }
            }
        }
    }
    deep_merge(config, txp_config)
    return config


class TransportMetabolism(Generator):
    """
    Transport/Metabolism Compartment, with ODE expression
    """
    name = NAME
    defaults = {
        'boundary_path': ('boundary',),
        'agents_path': ('agents',),
        'daughter_path': tuple(),
        'fields_path': ('fields',),
        'dimensions_path': ('dimensions',),
        'division': {},
        'transport': default_transport_config(),
        'metabolism': default_metabolism_config(),
        'expression': default_expression_config(),
        'divide': True
    }

    def __init__(self, config=None):
        super(TransportMetabolism, self).__init__(config)

    def generate_processes(self, config):
        daughter_path = config['daughter_path']
        agent_id = config['agent_id']

        # Transport
        transport = ConvenienceKinetics(config['transport'])

        # Metabolism
        # get target fluxes from transport, and update constrained_reaction_ids
        metabolism_config = config['metabolism']
        target_fluxes = transport.kinetic_rate_laws.reaction_ids
        metabolism_config.update({'constrained_reaction_ids': target_fluxes})
        metabolism = Metabolism(metabolism_config)

        # Gene expression
        expression = ODE_expression(config['expression'])

        # Mass deriver
        mass_deriver = TreeMass({})

        # Division
        division_condition = DivisionVolume({})

        processes = {
            'transport': transport,
            'metabolism': metabolism,
            'expression': expression,
            'mass_deriver': mass_deriver,
            'division': division_condition,
        }

        # divide process set to true, add meta-division processes
        if config['divide']:
            meta_division_config = dict(
                {},
                daughter_path=daughter_path,
                agent_id=agent_id,
                compartment=self)
            meta_division = MetaDivision(meta_division_config)
            processes['meta_division'] = meta_division

        return processes


    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        agents_path = config['agents_path']
        fields_path = config['fields_path']
        dimensions_path = config['dimensions_path']
        external_path = boundary_path + ('external',)
        topology = {
            'transport': {
                'internal': ('cytoplasm',),
                'external': external_path,
                'fields': ('null',),  # metabolism's exchange is used
                'fluxes': ('flux_bounds',),
                'global': boundary_path,
                'dimensions': dimensions_path,
            },
            'metabolism': {
                'internal': ('cytoplasm',),
                'external': external_path,
                'reactions': ('reactions',),
                'fields': fields_path,
                'flux_bounds': ('flux_bounds',),
                'global': boundary_path,
                'dimensions': dimensions_path,
            },
            'expression': {
                'counts': ('cytoplasm_counts',),
                'internal': ('cytoplasm',),
                'external': external_path,
                'global': boundary_path,
            },
            'mass_deriver': {
                'global': boundary_path,
            },
            'division': {
                'global': boundary_path,
            },
        }
        if config['divide']:
            topology.update({
                'meta_division': {
                    'global': boundary_path,
                    'agents': agents_path,
                }})
        return topology


# simulate
def test_txp_mtb_ge():
    default_test_setting = {
        'environment': {
            'volume': 1e-12 * units.L,
            'ports': {
                'fields': ('fields',),
                'external': ('boundary', 'external'),
                'dimensions': ('dimensions',),
                'global': ('boundary',),
            }},
        'timestep': 1,
        'total_time': 10}

    agent_id = '0'
    compartment = TransportMetabolism({'agent_id': agent_id})
    return simulate_compartment_in_experiment(compartment, default_test_setting)

def get_metabolism_initial_state(
    scale_concentration=1,
    override={}
):
    # get external state from iAF1260b metabolism
    config = get_iAF1260b_config()
    metabolism = Metabolism(config)
    molecules = {
        mol_id: conc * scale_concentration
        for mol_id, conc in metabolism.initial_state['external'].items()
    }
    for mol_id, conc in override.items():
        molecules[mol_id] = conc
    return molecules

def simulate_transport_metabolism(config={}):
    end_time = config.get('end_time', 2520)  # 2520 sec (42 min) is the expected doubling time in minimal media
    environment_volume = config.get('environment_volume', 1e-14)

    # make the compartment
    agent_id = '0'
    compartment = TransportMetabolism({
        'agent_id': agent_id,
        'divide': False})

    # make timeline initial state
    initial_state = get_metabolism_initial_state(
        # scale_concentration=1000,
        # override={'glc__D_e': 1.0, 'lcts_e': 1.0}
    )
    initial_state = {
        ('external', mol_id): conc
        for mol_id, conc in initial_state.items()}
    timeline = [
        (0, initial_state),
        # (200, initial_state),
        (end_time, {})]

    # run simulation
    sim_settings = {
        # 'initial_state': initial_state,
        'environment': {
            'volume': environment_volume * units.L,
            'ports': {
                'fields': ('fields',),
                'external': ('boundary', 'external'),
                'dimensions': ('dimensions',),
                'global': ('boundary',),
            }},
        'timeline': {
            'timeline': timeline,
            'ports': {
                'external': ('boundary', 'external'),
                'global': ('boundary',)}}}
    return simulate_compartment_in_experiment(compartment, sim_settings)


if __name__ == '__main__':
    out_dir = os.path.join(COMPARTMENT_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    config ={
        'end_time': 2520,
        'environment_volume': 1e-12,
    }
    timeseries = simulate_transport_metabolism(config)
    analyze_transport_metabolism(timeseries, config, out_dir)
