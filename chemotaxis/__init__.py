import os

from vivarium.core.registry import process_registry

# make paths
package_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PROCESS_OUT_DIR = os.path.join(package_path, 'out', 'processes')
COMPOSITE_OUT_DIR = os.path.join(package_path, 'out', 'composites')
EXPERIMENT_OUT_DIR = os.path.join(package_path, 'out', 'experiments')
REFERENCE_DATA_DIR = os.path.join(package_path, 'chemotaxis', 'reference_data')

# register processes