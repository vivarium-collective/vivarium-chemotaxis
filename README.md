# vivarium-chemotaxis

[vivarium-chemotaxis](https://github.com/vivarium-collective/vivarium-chemotaxis) is a library for chemotaxis-specific
vivarium processes.

## setup
Make a python environment and install dependencies. 

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
All experiments from the paper "A Multi-Scale Approach to Modeling E. coli Chemotaxis" are available in the same file.
Run them from the command line by specifying the figure number ('3b', '5a', '5b', '5c', '6a', '6b', '6c', '7a', '7b', '7c', '7d')
```
$ python chemotaxis/experiments/run_paper.py 7a
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