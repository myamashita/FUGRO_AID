""" 
Aid
---
This is the Aid module.
Host of useful manipulation methods for Furgo Metocean Consultancy - Americas.
"""
import numpy as np
import pandas as pd
import datetime as dtm

__version__ = '1.0.0'
print(f'This module version is {__version__}\nno updates available.')


class Aid():
    """ Support variables"""
    bk = __import__('bokeh', globals(), locals(), [], 0)
    """ Suport bokeh figures"""
    import importlib
    figure = getattr(importlib.import_module('bokeh.plotting'), 'figure')


class Bokeh(object):
    """ Suport bokeh figures"""
    import bokeh.palettes as bp
    from bokeh.plotting import figure
    from bokeh.models import ColorBar, HoverTool, Range1d, LinearAxis
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

    def plot_Colour_Flood(data, X_tooltip='Date', Y_tooltip='Altitude',
                          Z_tooltip='Intensity', vmin=0, vmax=100,
                          f=None, **kw):
        """Additional Keyword arguments to be passed to the functions:
            bokeh.mk_fig --> 'title'; 'x_axis_label'; 'y_axis_label'; 'height';
                             'width'; 'x_axis_type'; 'axis_label_text_font'
            bokeh.plot_Colour_Flood --> 'X_tooltip'; 'Y_tooltip'; 'Z_tooltip';
                                  'Dir_color'; 'Dir_legend'
        """
        if f is None:
            f = Bokeh.mk_fig(
                title=kw.pop('title', 'Initial title'),
                x_axis_label=kw.pop('xlabel', 'Initial xlabel'),
                y_axis_label=kw.pop('ylabel', 'Initial ylabel'),
                height=kw.pop('height', 350), width=kw.pop('width', 1100),
                x_axis_type=kw.pop('x_axis_type', 'datetime'), **kw)

        f.add_tools(Bokeh._get_HoverTool(
            X_tooltip, Y_tooltip, Z_tooltip, image=True))
        mapper = Bokeh.linear_cmap(field_name='y', palette=kw.get(
            'palette', Bokeh.bp.Turbo256), low=vmin, high=vmax)
        f.image(source=data, image='image', x='x', y='y', dw='dw', dh='dh',
                palette=kw.get('palette', Bokeh.bp.Turbo256))
        cbar = Bokeh.ColorBar(
            color_mapper=mapper['transform'], width=18, location=(0, 0),
            major_tick_line_color='#000000', label_standoff=12,
            major_label_text_font_size=kw.get('axislabel_fontsize', '14pt'))
        f.add_layout(cbar, 'right')
        return f

    def plot_IntDir(dfI, dfD, f=None, **kw):
        """Additional Keyword arguments to be passed to the functions:
            bokeh.mk_fig --> 'title'; 'x_axis_label'; 'y_axis_label'; 'height';
                             'width'; 'x_axis_type'; 'axis_label_text_font'
            bokeh.plot_IntDir --> 'Int_color'; 'Int_legend', 'Dir_label';
                                  'Dir_color'; 'Dir_legend'
        """
        if f is None:
            f = Bokeh.mk_fig(
                title=kw.pop('title', 'Initial title'),
                x_axis_label=kw.pop('xlabel', 'Initial xlabel'),
                y_axis_label=kw.pop('ylabel', 'Initial ylabel'),
                height=kw.pop('height', 350), width=kw.pop('width', 1100),
                x_axis_type=kw.pop('x_axis_type', 'datetime'), **kw)
        Int_color = kw.get('Int_color', '#6788B1')
        Int_legend = kw.get('Int_legend', 'Intensity')
        I = f.line(dfI.index, dfI, color=Int_color, muted_color=Int_color,
                   muted_alpha=0.2, line_width=2, legend_label=Int_legend)
        H = Bokeh._get_HoverTool(ycoordlabel=Int_legend, renderer=I)
        f.add_tools(H)
        Dir_label = kw.get('Dir_label', 'Direction [°T]')
        Dir_color = kw.get('Dir_color', '#D9BE89')
        Dir_legend = kw.get('Dir_legend', 'Direction')
        f.extra_y_ranges = {"Dir": Bokeh.Range1d(start=0, end=360)}
        D = f.circle(dfD.index, dfD, color=Dir_color, muted_color=Dir_color,
                     legend_label=Dir_legend, y_range_name="Dir",
                     muted_alpha=0.2)
        f.add_layout(Bokeh.LinearAxis(y_range_name="Dir",
                                      axis_label=Dir_label), 'right')
        H = Bokeh._get_HoverTool(ycoordlabel=Dir_legend, renderer=D)
        f.add_tools(H)
        f.x_range = Bokeh.Range1d(dfI.index[0], dfI.index[-1])
        f.y_range = Bokeh.Range1d(dfI.min(), dfI.max())
        f.legend.click_policy = "mute"
        return Bokeh.set_fig(f, **kw)

    def _get_HoverTool(xcoordlabel='Date', ycoordlabel='', zcoordlabel='',
                       line_policy='nearest', image=False, renderer=None):
        if image:
            H = {'tooltips': [(xcoordlabel, "@dt{  %Y-%m-%d %H:%M}"),
                              (ycoordlabel, '$y'), (zcoordlabel, '@image')],
                 'formatters': {'@dt': 'datetime'}}
        else:
            H = {'renderers': [renderer],
                 'tooltips': [(xcoordlabel, "@x{  %Y-%m-%d %H:%M}"),
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
