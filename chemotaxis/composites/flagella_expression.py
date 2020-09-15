"""
==============================
Flagella Expression Composites
==============================
"""

import os
import argparse
import random
import uuid
import copy
import pytest
import numpy as np

# vivarium-core imports
from vivarium.core.process import Generator
from vivarium.core.composition import (
    simulate_compartment_in_experiment,
    plot_simulation_output,
    save_flat_timeseries,
    load_timeseries,
    assert_timeseries_close,
)
from vivarium.core.emitter import path_timeseries_from_embedded_timeseries
from vivarium.library.units import units

# data
from cell.data.nucleotides import nucleotides
from cell.data.amino_acids import amino_acids
from cell.plots.gene_expression import plot_timeseries_heatmaps
from chemotaxis.data.chromosomes.flagella_chromosome import FlagellaChromosome

# processes
from cell.processes.transcription import Transcription, UNBOUND_RNAP_KEY
from cell.processes.translation import Translation, UNBOUND_RIBOSOME_KEY
from cell.processes.degradation import RnaDegradation
from cell.processes.complexation import Complexation
from cell.processes.convenience_kinetics import ConvenienceKinetics
from cell.processes.metabolism import Metabolism, get_iAF1260b_config
from cell.processes.division_volume import DivisionVolume
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.tree_mass import TreeMass

# composites
from cell.composites.gene_expression import GeneExpression
from chemotaxis.composites.transport_metabolism import (
    glucose_lactose_transport_config,
    default_metabolism_config,
)

# plots
from cell.plots.gene_expression import plot_gene_expression_output

# directories
from chemotaxis import COMPOSITE_OUT_DIR, REFERENCE_DATA_DIR


NAME = 'flagella_gene_expression'



flagella_schema_override = {
    'transcription': {
        'proteins': {
            'CpxR': {'_divider': 'set'},
            'CRP': {'_divider': 'set'},
            'Fnr': {'_divider': 'set'},
            'endoRNAse': {'_divider': 'set'},
        }
    },
    'translation': {
        'proteins': {
            'CpxR': {'_divider': 'set'},
            'CRP': {'_divider': 'set'},
            'Fnr': {'_divider': 'set'},
            'endoRNAse': {'_divider': 'set'},
        }
    },
}


def get_flagella_expression_config(config):
    flagella_data = FlagellaChromosome(config)
    chromosome_config = flagella_data.chromosome_config
    sequences = flagella_data.chromosome.product_sequences()

    return {
        'transcription': {
            'sequence': chromosome_config['sequence'],
            'templates': chromosome_config['promoters'],
            'genes': chromosome_config['genes'],
            'transcription_factors': flagella_data.transcription_factors,
            'promoter_affinities': flagella_data.promoter_affinities,
            'polymerase_occlusion': 30,
            'elongation_rate': 50},

        'translation': {
            'sequences': flagella_data.protein_sequences,
            'templates': flagella_data.transcript_templates,
            'concentration_keys': ['CRP', 'flhDC', 'fliA'],
            'transcript_affinities': flagella_data.transcript_affinities,
            'elongation_rate': 22,
            'polymerase_occlusion': 50},

        'degradation': {
            'sequences': sequences,
            'catalytic_rates': {
                'endoRNAse': 0.02},
            'michaelis_constants': {
                'transcripts': {
                    'endoRNAse': {
                        transcript: 1e-23
                        for transcript in chromosome_config['genes'].keys()}}}},

        'complexation': {
            'monomer_ids': flagella_data.complexation_monomer_ids,
            'complex_ids': flagella_data.complexation_complex_ids,
            'stoichiometry': flagella_data.complexation_stoichiometry,
            'rates': flagella_data.complexation_rates},

        '_schema': copy.deepcopy(flagella_schema_override)
    }


# initial state for flagella gene expression compartment
def get_flagella_expression_initial_state(ports={}):
    expression_config = get_flagella_metabolism_initial_state(ports)

    # add molecules
    molecules = {}
    for nucleotide in nucleotides.values():
        molecules[nucleotide] = 5e6
    for amino_acid in amino_acids.values():
        molecules[amino_acid] = 1e6
    molecules['ATP'] = 5e7

    expression_config.update({
        ports.get('molecules', 'molecules'):
            molecules})
    return expression_config


# initial state for flagella expression with metabolism
def get_flagella_metabolism_initial_state(ports={}):
    flagella_data = FlagellaChromosome()
    chromosome_config = flagella_data.chromosome_config

    # justification for 15 ribosomes: E. coli has ~7000-70000 ribosomes
    # (http://book.bionumbers.org/how-many-ribosomes-are-in-a-cell/),
    # and 4 flagella would make up ~0.0002 of total mass. Which indicates
    # 2-14 ribosomes are required if allocation is proportional to mass.
    # there are ~2000 RNAPs in E. coli (Bremer, and Dennis 1996)
    return {
        ports.get('transcripts', 'transcripts'): {
            gene: 0
            for gene in chromosome_config['genes'].keys()
        },
        ports.get('proteins', 'proteins'): {
            'CpxR': 10,
            'CRP': 10,
            'Fnr': 10,
            'endoRNAse': 1,
            'flagella': 4,
            UNBOUND_RIBOSOME_KEY: 15,
            UNBOUND_RNAP_KEY: 15,
        },
    }


# flagella expression compartment
def FlagellaGeneExpression(config):
    """
    Make a gene expression compartment with flagella expression data
    """
    flagella_expression_config = get_flagella_expression_config(config)
    return GeneExpression(flagella_expression_config)


# flagella expression compartment with transport and metabolism
class FlagellaExpressionMetabolism(Generator):
    """ Flagella stochastic expression with metabolism """

    name = 'flagella_expression_metabolism'
    defaults = {
        'boundary_path': ('boundary',),
        'fields_path': ('fields',),
        'dimensions_path': ('dimensions',),
        'agents_path': ('agents',),
        'daughter_path': tuple(),
        'flagella_chromosome': {},
        'transport': glucose_lactose_transport_config(),
        'metabolism': default_metabolism_config(),
        'initial_mass': 0.0 * units.fg,
        'expression_time_step': 10,
        'divide': True,
    }

    def __init__(self, config=None):
        if config is None:
            config = {}
        # get flagella expression config and update config
        chromosome_config = config.get(
            'flagella_chromosome',
            self.defaults['flagella_chromosome'])
        flagella_expression_config = get_flagella_expression_config(
            chromosome_config)
        config.update(flagella_expression_config)

        super(FlagellaExpressionMetabolism, self).__init__(config)

        if 'agent_id' not in self.config:
            self.config['agent_id'] = str(uuid.uuid1())

    def generate_processes(self, config):
        daughter_path = config['daughter_path']
        agent_id = config['agent_id']

        # get the configs
        transcription_config = config['transcription']
        translation_config = config['translation']
        degradation_config = config['degradation']
        complexation_config = config['complexation']

        # update expression timestep
        transcription_config.update({'time_step': config['expression_time_step']})
        translation_config.update({'time_step': config['expression_time_step']})
        degradation_config.update({'time_step': config['expression_time_step']})
        complexation_config.update({'time_step': config['expression_time_step']})

        # make the expression processes
        transcription = Transcription(transcription_config)
        translation = Translation(translation_config)
        degradation = RnaDegradation(degradation_config)
        complexation = Complexation(complexation_config)
        mass_deriver = TreeMass(config.get('mass_deriver', {
            'initial_mass': config['initial_mass']}))

        # Transport
        transport_config = config['transport']
        transport = ConvenienceKinetics(transport_config)
        target_fluxes = transport.kinetic_rate_laws.reaction_ids

        # Metabolism
        # add target fluxes from transport
        metabolism_config = config.get('metabolism')
        metabolism_config.update({'constrained_reaction_ids': target_fluxes})
        metabolism = Metabolism(metabolism_config)

        # Division condition
        division_condition = DivisionVolume({})

        processes = {
            'metabolism': metabolism,
            'transport': transport,
            'mass_deriver': mass_deriver,
            'transcription': transcription,
            'translation': translation,
            'degradation': degradation,
            'complexation': complexation,
            'division': division_condition
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
        fields_path = config['fields_path']
        dimensions_path = config['dimensions_path']
        agents_path = config['agents_path']
        external_path = boundary_path + ('external',)

        topology = {
            'mass_deriver': {
                'global': boundary_path,
            },
            'transcription': {
                'chromosome': ('chromosome',),
                'molecules': ('molecules',),
                'proteins': ('proteins',),
                'transcripts': ('transcripts',),
                'factors': ('concentrations',),
                'global': boundary_path,
            },
            'translation': {
                'ribosomes': ('ribosomes',),
                'molecules': ('molecules',),
                'transcripts': ('transcripts',),
                'proteins': ('proteins',),
                'concentrations': ('concentrations',),
                'global': boundary_path,
            },
            'degradation': {
                'transcripts': ('transcripts',),
                'proteins': ('proteins',),
                'molecules': ('molecules',),
                'global': boundary_path,
            },
            'complexation': {
                'monomers': ('proteins',),
                'complexes': ('proteins',),
                'global': boundary_path,
            },
            'transport': {
                'internal': ('molecules',),
                'external': external_path,
                'fields': ('null',),  # metabolism's exchange is used
                'fluxes': ('flux_bounds',),
                'global': boundary_path,
                'dimensions': dimensions_path,
            },
            'metabolism': {
                'internal': ('molecules',),
                'external': external_path,
                'fields': fields_path,
                'reactions': ('reactions',),
                'flux_bounds': ('flux_bounds',),
                'global': boundary_path,
                'dimensions': dimensions_path,
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


# simulation
def run_flagella_compartment(
        compartment,
        total_time=2500,
        initial_state=None,
        out_dir='out'):

    # get flagella data
    flagella_data = FlagellaChromosome()

    # run simulation
    settings = {
        'total_time': total_time,
        'emit_step': 10,
        'initial_state': initial_state}
    timeseries = simulate_compartment_in_experiment(compartment, settings)

    # save reference timeseries
    save_flat_timeseries(timeseries, out_dir)

    # plot gene expression figure
    plot_config = {
        'name': 'flagella_expression',
        'ports': {
            'transcripts': 'transcripts',
            'proteins': 'proteins',
            'molecules': 'molecules'}}
    plot_gene_expression_output(
        timeseries,
        plot_config,
        out_dir)

    # plot just-in-time figure
    plot_config2 = plot_config.copy()
    plot_config2.update({
        'name': 'flagella',
        'plot_ports': {
            'transcripts': list(flagella_data.chromosome_config['genes'].keys()),
            'proteins': flagella_data.complexation_monomer_ids + flagella_data.complexation_complex_ids,
            'molecules': list(nucleotides.values()) + list(amino_acids.values())}})
    plot_timeseries_heatmaps(
        timeseries,
        plot_config2,
        out_dir)

    # plot basic sim output
    plot_settings = {
        'max_rows': 30,
        'remove_zeros': True,
        'skip_ports': ['chromosome', 'ribosomes']}
    plot_simulation_output(
        timeseries,
        plot_settings,
        out_dir)


# tests
def test_flagella_metabolism(seed=1):
    random.seed(seed)
    np.random.seed(seed)

    # make the compartment and state
    compartment = FlagellaExpressionMetabolism({'divide': False})
    initial_state = get_flagella_metabolism_initial_state()
    name = compartment.name
    # run simulation
    settings = {
        'total_time': 60,
        'emit_step': 10,
        'initial_state': initial_state,
    }
    timeseries = simulate_compartment_in_experiment(compartment, settings)

    # remove non-numerical data for timeseries comparison, convert to path_timeseries
    del timeseries['chromosome']
    del timeseries['ribosomes']
    del timeseries['dimensions']
    del timeseries['boundary']['divide']
    del timeseries['fields']
    del timeseries['null']
    path_timeseries = path_timeseries_from_embedded_timeseries(timeseries)

    # # save reference timeseries
    # out_dir = os.path.join(COMPOSITE_OUT_DIR, name)
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # save_flat_timeseries(path_timeseries, out_dir)

    reference = load_timeseries(
        os.path.join(REFERENCE_DATA_DIR, name + '.csv'))
    assert_timeseries_close(path_timeseries, reference)


@pytest.mark.slow
def test_flagella_expression():
    flagella_compartment = FlagellaGeneExpression({})

    # initial state for flagella complexation
    initial_state = get_flagella_expression_initial_state()
    initial_state['proteins'].update({
        'Ribosome': 100,  # plenty of ribosomes
        'flagella': 0,
        # required flagella components
        'flagellar export apparatus': 1,
        'flagellar motor': 1,
        'fliC': 1,
        'flgL': 1,
        'flgK': 1,
        'fliD': 5,
        'flgE': 120
    })

    # run simulation
    random.seed(0)  # set seed because process is stochastic
    settings = {
        'total_time': 1000,
        'emit_step': 100,
        'initial_state': initial_state}
    timeseries = simulate_compartment_in_experiment(flagella_compartment, settings)

    print(timeseries['proteins']['flagella'])
    final_flagella = timeseries['proteins']['flagella'][-1]
    # this should have been long enough for flagellar complexation to occur
    assert final_flagella > 0


if __name__ == '__main__':
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    parser = argparse.ArgumentParser(description='flagella expression')
    parser.add_argument('--metabolism', '-m', action='store_true', default=False, )
    args = parser.parse_args()

    if args.metabolism:
        mtb_out_dir = os.path.join(out_dir, 'metabolism')
        if not os.path.exists(mtb_out_dir):
            os.makedirs(mtb_out_dir)
        # get initial state
        initial_state = get_flagella_metabolism_initial_state()

        # configure the compartment
        flagella_config = {'divide': False}
        compartment = FlagellaExpressionMetabolism(flagella_config)

        # run sim
        total_time = 2500
        run_flagella_compartment(
            compartment=compartment,
            total_time=total_time,
            initial_state=initial_state,
            out_dir=out_dir)
    else:
        # get initial state
        initial_state = get_flagella_expression_initial_state()

        # configure the compartment
        flagella_chromosome_config = {}
        compartment = FlagellaGeneExpression(flagella_chromosome_config)

        # run sim
        total_time = 5000  # 2500
        run_flagella_compartment(
            compartment=compartment,
            total_time=total_time,
            initial_state=initial_state,
            out_dir=out_dir)
