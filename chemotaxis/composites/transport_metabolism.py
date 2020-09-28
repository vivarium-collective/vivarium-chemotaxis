"""
==================================================
Transport Metabolism and Gene Expression Composite
==================================================
"""

import os
import argparse

from vivarium.library.units import units
from vivarium.core.process import Generator
from vivarium.core.emitter import path_timeseries_from_embedded_timeseries
from vivarium.core.composition import (
    simulate_compartment_in_experiment,
    save_flat_timeseries,
    load_timeseries,
    assert_timeseries_close,
)
from vivarium.plots.simulation_output import plot_simulation_output

# processes
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.tree_mass import TreeMass
from vivarium_cell.processes.division_volume import DivisionVolume
from vivarium_cell.processes.metabolism import (
    Metabolism,
    get_minimal_media_iAF1260b,
)
from vivarium_cell.processes.metabolism import get_iAF1260b_config as get_iAF1260b_path_config
from vivarium_cell.processes.convenience_kinetics import ConvenienceKinetics
from vivarium_cell.processes.ode_expression import ODE_expression

# plots
from chemotaxis.plots.transport_metabolism import plot_glc_lcts_environment

# directories
from chemotaxis import COMPOSITE_OUT_DIR, REFERENCE_DATA_DIR


NAME = 'transport_metabolism'


def get_iAF1260b_config():
    """
    :py:class:`Metabolism` configuration for with iAF1260b BiGG model,
    initial_mass, and tolerances set on the glucose/lactose exchange
    reactions.
    """

    config = get_iAF1260b_path_config()

    # flux bound tolerance for reactions in glucose_lactose_transport_config
    metabolism_config = {
        'initial_mass': 1339.0,  # fg of metabolite pools
        'tolerance': {
            'EX_glc__D_e': [1.05, 1.0],
            'EX_lcts_e': [1.05, 1.0]}}
    config.update(metabolism_config)
    return config


def get_lacY_expression_config():
    """
    :py:class:`ODE_expression` configuration for expression of glucose
    and lactose transporters
    """

    # expression
    transcription_rates = {
        'lacy_RNA': 5e-6}
    translation_rates = {
        'LacY': 2e-4}
    protein_map = {
        'LacY': 'lacy_RNA'}
    degradation_rates = {
        'lacy_RNA': 3e-3,  # a single RNA lasts about 5 minutes
        'LacY': 3e-5}  # proteins degrade ~100x slower

    # regulation
    regulators = [
        ('external', 'glc__D_e'),
        ('internal', 'lcts_p')]
    regulation_condition = {
        'lacy_RNA': 'if [(external, glc__D_e) > 0.05 '  # limiting concentration of glc
                    'or (internal, lcts_p) < 0.05]'}  # internal lcts is hypothesized to disinhibit lacY transcription
    transcription_leak = {
        'rate': 1e-4,
        'magnitude': 1e-7}

    # initial state
    initial_state = {
        'internal': {
            'lacy_RNA': 0.0,
            'LacY': 0.0},
        'external': {
            'glc__D_e': 8.0,
            'lcts_e': 8.0}}

    return {
        'transcription_rates': transcription_rates,
        'translation_rates': translation_rates,
        'degradation_rates': degradation_rates,
        'protein_map': protein_map,
        'regulators': regulators,
        'regulation': regulation_condition,
        'transcription_leak': transcription_leak,
        'initial_state': initial_state}


def get_glucose_lactose_transport_config():
    """
    :py:class:`ConvenienceKinetics` configuration for simplified glucose
    and lactose transport.Glucose uptake simplifies the PTS/GalP system
    to a single uptake kinetic with ``glc__D_e_external`` as the only
    cofactor.
    """
    transport_reactions = {
        'EX_glc__D_e': {
            'stoichiometry': {
                ('internal', 'g6p_c'): 1.0,
                ('external', 'glc__D_e'): -1.0,
                ('internal', 'pep_c'): -1.0,
                ('internal', 'pyr_c'): 1.0},
            'is reversible': False,
            'catalyzed by': [
                ('internal', 'EIIglc')]},
        'EX_lcts_e': {
            'stoichiometry': {
                ('external', 'lcts_e'): -1.0,
                ('internal', 'lcts_p'): 1.0},
            'is reversible': False,
            'catalyzed by': [
                ('internal', 'LacY')]}}

    transport_kinetics = {
        'EX_glc__D_e': {
            ('internal', 'EIIglc'): {
                ('external', 'glc__D_e'): 1e-1,  # (mM) k_m for glc
                ('internal', 'pep_c'): None,  # k_m = None makes a reactant non-limiting
                'kcat_f': 1e2}},
        'EX_lcts_e': {
            ('internal', 'LacY'): {
                ('external', 'lcts_e'): 1e-1,  # (mM) k_m for lcts
                'kcat_f': 1e2}}}

    transport_initial_state = {
        'internal': {
            'EIIglc': 1.0e-3,  # (mmol/L)
            'g6p_c': 0.0,
            'pep_c': 1.0e-1,
            'pyr_c': 0.0,
            'LacY': 0,
            'lcts_p': 0.0},
        'external': {
            'glc__D_e': 10.0,
            'lcts_e': 10.0}}

    transport_ports = {
        'internal': [
            'g6p_c', 'pep_c', 'pyr_c', 'EIIglc', 'LacY', 'lcts_p'],
        'external': [
            'glc__D_e', 'lcts_e']}

    return {
        'reactions': transport_reactions,
        'kinetic_parameters': transport_kinetics,
        'initial_state': transport_initial_state,
        'ports': transport_ports}


def get_metabolism_initial_external_state(
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


class TransportMetabolismExpression(Generator):
    """ Transport/Metabolism/Expression composite

    Metabolism is an FBA BiGG model, transport is a kinetic model with
    convenience kinetics, gene expression is an ODE model
    """
    name = NAME
    defaults = {
        'boundary_path': ('boundary',),
        'agents_path': ('agents',),
        'daughter_path': tuple(),
        'fields_path': ('fields',),
        'dimensions_path': ('dimensions',),
        'division': {},
        'transport': get_glucose_lactose_transport_config(),
        'metabolism': get_iAF1260b_config(),
        'expression': get_lacY_expression_config(),
        'divide': True,
    }

    def __init__(self, config=None):
        super(TransportMetabolismExpression, self).__init__(config)

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
        division_condition = DivisionVolume(config['division'])

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
def test_txp_mtb_ge(out_dir='out'):
    timeseries = run_txp_mtb_ge(
        env_volume=1e-12,
        total_time=10
    )
    path_timeseries = path_timeseries_from_embedded_timeseries(timeseries)
    save_flat_timeseries(path_timeseries, out_dir)
    reference = load_timeseries(
        os.path.join(REFERENCE_DATA_DIR, NAME + '.csv'))
    assert_timeseries_close(path_timeseries, reference)


def run_txp_mtb_ge(
    env_volume=1e-12,
    total_time=10,
    minimal_media=get_minimal_media_iAF1260b()
):
    # make the compartment
    compartment = TransportMetabolismExpression({
        'agent_id': '0',
        'divide': False})

    # configure simulation
    default_test_setting = {
        'environment': {
            'volume': env_volume * units.L,
            'concentrations': minimal_media,
            'ports': {
                'fields': ('fields',),
                'external': ('boundary', 'external'),
                'dimensions': ('dimensions',),
                'global': ('boundary',)}},
        'total_time': total_time}

    # run simulation
    return simulate_compartment_in_experiment(compartment, default_test_setting)


if __name__ == '__main__':
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    parser = argparse.ArgumentParser(description='transport metabolism')
    parser.add_argument('--shift', '-s', action='store_true', default=False)
    args = parser.parse_args()

    if args.shift:
        out_dir = os.path.join(out_dir, 'shift')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        minimal_media = get_metabolism_initial_external_state(
            scale_concentration=100,
            override={'glc__D_e': 1.0, 'lcts_e': 1.0})
        environment_volume = 1e-13
    else:
        minimal_media = get_minimal_media_iAF1260b()
        environment_volume = 1e-6

    # simulate
    total_time = 2500
    timeseries = run_txp_mtb_ge(
        env_volume=environment_volume,
        total_time=total_time,
        minimal_media=minimal_media)

    # print resulting growth
    volume_ts = timeseries['boundary']['volume']
    mass_ts = timeseries['boundary']['mass']
    print('volume growth: {}'.format(volume_ts[-1] / volume_ts[1]))
    print('mass growth: {}'.format(mass_ts[-1] / mass_ts[1]))

    # plot
    # simulation plot
    plot_settings = {
        'max_rows': 30,
        'remove_flat': True,
        'remove_zeros': True,
        'skip_ports': ['null', 'reactions']}
    plot_simulation_output(timeseries, plot_settings, out_dir)

    # glucose-lactose plot
    settings = {
        'internal_path': ('cytoplasm',),
        'external_path': ('boundary', 'external'),
        'global_path': ('boundary',),
        'environment_volume': environment_volume}
    plot_glc_lcts_environment(timeseries, settings, out_dir)
