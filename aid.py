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
    """ Suport bokeh figures"""
    import importlib
    figure = getattr(importlib.import_module('bokeh.plotting'), 'figure')


class Bokeh(object):
    """ Suport bokeh figures"""
    from bokeh.plotting import figure
    from bokeh.models import ColorBar, HoverTool
    from bokeh.transform import linear_cmap
    from bokeh.models.formatters import DatetimeTickFormatter

    def mk_fig(title='Initial title', x_axis_label='Initial xlabel',
               y_axis_label='Initial ylabel', height=350, width=1100,
               x_axis_type='datetime', **kw):
        sizing_mode = kw.get('sizing_mode', 'scale_both')
        f = Bokeh.figure(title=title, height=height, width=width,
                         x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                         x_axis_type=x_axis_type, sizing_mode=sizing_mode)
        return Bokeh.set_fig(f, **kw)

    def set_fig(f, title_fontsize='16pt', title_fontstyle='normal',
                axislabel_fontsize='14pt', axislabel_fontstyle='normal', **kw):
        font = kw.get('axis_label_text_font', 'segoe ui')
        f.axis.axis_label_text_font = font
        f.title.text_font_size = title_fontsize
        f.axis.axis_label_text_font_size = axislabel_fontsize
        f.axis.major_label_text_font_size = axislabel_fontsize
        f.axis.axis_label_text_font_style = axislabel_fontstyle
        if hasattr(f.x_range, 'range_padding'):
            f.x_range.range_padding = 0
        if hasattr(f.y_range, 'range_padding'):
            f.y_range.range_padding = 0
        dt = {'months': ["%Y/%m/%d"], 'days': ["%b/%d"],
              'hours': ["%d %H:%M"], 'minutes': ["%H:%M:%S"]}
        f.xaxis.formatter = Bokeh.DatetimeTickFormatter(**dt)
        f.xgrid.grid_line_color = None
        f.ygrid.grid_line_color = None
        return f

    def _get_HoverTool(ycoordlabel='', zcoordlabel='',
                       line_policy='nearest', image=False):
        if image:
            H = {'tooltips': [("Date", "@dt{  %Y-%m-%d %H:%M}"),
                              (ycoordlabel, '$y'), (zcoordlabel, '@image')],
                 'formatters': {'@dt': 'datetime'}}
        else:
            H = {'tooltips': [("Date", "@x{  %Y-%m-%d %H:%M}"),
                              (ycoordlabel, '$y')],
                 'formatters': {'@x': 'datetime'}, 'line_policy': line_policy}
        return Bokeh.HoverTool(**H)


class Erddap(object):
    """ Support FUGRO Erddap connections and requests"""

    def __init__(self, server='http://10.1.1.17:8080/erddap',
                 protocol='tabledap', response='csv', dataset_id=None,
                 constraints=None, variables=None):
        self._base = Erddap.erddap_instance(server, protocol, response)
        self.dataset_id = dataset_id
        self.constraints = constraints
        self.variables = variables

    def __repr__(self):
        return (f'Instance at server: {self._base.server}\n'
                f'Dataset ID: {self._base.dataset_id}\n'
                f'Constraints: {self._base.constraints}\n'
                f'Variables: {self._base.variables}')

    def __str__(self):
        return 'ERDDAP instance for a specific server endpoint.'

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
        def dateparser(x): return dtm.datetime.strptime(
            x, "%Y-%m-%dT%H:%M:%SZ")
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
