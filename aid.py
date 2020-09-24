""" 
Aid
---
This is the Aid module.
Host of useful manipulation methods for Furgo Metocean Consultancy - Americas.
"""
import numpy as np
import pandas as pd
import datetime as dtm

class Aid():
    """ Support variables"""
    bk = __import__('bokeh', globals(), locals(), [], 0)
    

class Erddap(object):

    def __init__(self, server='http://10.1.1.17:8080/erddap',
                 protocol='tabledap', response='csv', dataset_id=None,
                 constraints=None, variables=None):
        self._base = Erddap.erddap_instance(server, protocol, response)
        self.dataset_id = dataset_id
        self.constraints = constraints
        self.variables = variables

    def erddap_instance(server='http://10.1.1.17:8080/erddap',
                        protocol='tabledap', response='csv'):
        from erddapy import ERDDAP
        return ERDDAP(server=server, protocol=protocol, response=response)
    
    @property
    def dataset_id(self):
        return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, i):
        self._dataset_id = i
        self._base.dataset_id = i

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, c):
        self._constraints = c
        self._base.constraints = c

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, v):
        self._variables = v
        self._base.variables = v

    def _base_constraints(id, start=None, end=None):
        """ Create a constraints dictionary."""
        time = dtm.datetime.utcnow()
        time = time.replace(second=0, microsecond=0)
        if start is None:
            start = time - dtm.timedelta(hours=72)
        if end is None:
            end = time
        return {'id=': id, 'time>=': start, 'time<=': end}

    def to_pandas(self):
        dateparser = lambda x: dtm.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")
        kw = {'index_col': 'time', 'date_parser': dateparser,
              'skiprows': [1], 'response': 'csv'}
        return self._base.to_pandas(**kw)

    def vars_in_dataset(self, dataset):
        v = self._base._get_variables(dataset)
        v.pop('NC_GLOBAL', None)
        return [i for i in v]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
