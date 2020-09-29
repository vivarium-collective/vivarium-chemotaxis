# vivarium-chemotaxis

[Vivarium-chemotaxis](https://github.com/vivarium-collective/vivarium-chemotaxis) is a library for the multi-scale 
model of chemotaxis described in:
Agmon, E., Spangler, R.K. "A Multi-Scale Approach to Modeling E. coli Chemotaxis". (accepted).

Supplementary materials are available [here](doc/AgmonSpangler2020_supplementary.pdf).

Visit [the Vivarium Core
documentation](https://vivarium-core.readthedocs.io/) to learn how to
use the core Vivarium engine to create computational biology models.


![vivarium-chemotaxis](doc/_static/ecoli_master.png)
**ChemotaxisMaster Composite**, with processes for metabolism (MTB), transport (TXP), transcription (TSC), 
translation (TRL), complexation (CXN), degradation (DEG), proton motive force (PMF), flagella activity (FLG), 
and chemoreceptor activity (CHE).

## setup
Please refer to the Vivarium Core documentation for more complete instructions.

Make a python environment with Python 3 (pyenv is recommended) and install dependencies. 

First install numpy:
```
$ pip install numpy
```

Then the remaining requirements:
```
$ pip install -r requirements.txt
```

## run individual processes and composites
Each process file under `chemotaxis/processes` can run on its own. Some of these have their own command line options.
For example, call the `chemoreptor_cluster` process with:
```
$ python chemotaxis/processes/chemoreptor_cluster.py
```

Composites with multiple integrated processes can also be executed on their own:
```
$ python chemotaxis/composites/chemotaxis_flagella.py
```

## experiments
All experiments from the paper "A Multi-Scale Approach to Modeling E. coli Chemotaxis" 
are available in the file `chemotaxis/experiments/paper_experiments.py`. Run them from 
the command line by specifying the corresponding figure number.
```
$ python chemotaxis/experiments/paper_experiments.py 7b
``` 

## tests
Tests are performed with pytest. Simply call the following to ensure everything is working properly:
```
$ pytest
```

To run only the fast tests:
```
$ pytest -m 'not slow'
```

## logging
Logging is done with python logging. To print out logging information, run a simulation with:
```
$ LOGLEVEL=INFO python chemotaxis/..
```