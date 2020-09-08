import copy

from cell.plots.multibody_physics import plot_temporal_trajectory, plot_agent_trajectory, plot_motility
from cell.processes.static_field import make_field
from vivarium.core.composition import plot_agents_multigen
from vivarium.core.emitter import time_indexed_timeseries_from_data, timeseries_from_data


def plot_chemotaxis_experiment(
        data,
        field_config,
        out_dir,
        filename=''):

    # multigen agents plot
    plot_settings = {
        'agents_key': 'agents',
        'max_rows': 30,
        'skip_paths': [
            ('boundary', 'mass'),
            ('boundary', 'length'),
            ('boundary', 'width'),
            ('boundary', 'location'),
        ]}
    plot_agents_multigen(data, plot_settings, out_dir, 'agents')

    # trajectory and motility
    indexed_timeseries = time_indexed_timeseries_from_data(data)
    field = make_field(field_config)
    trajectory_config = {
        'bounds': field_config['bounds'],
        'field': field,
        'rotate_90': True}

    plot_temporal_trajectory(copy.deepcopy(indexed_timeseries), trajectory_config, out_dir, filename + 'temporal')
    plot_agent_trajectory(copy.deepcopy(indexed_timeseries), trajectory_config, out_dir, filename + 'trajectory')

    embdedded_timeseries = timeseries_from_data(data)
    plot_motility(embdedded_timeseries, out_dir, filename + 'motility_analysis')
