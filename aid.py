""" 
Aid
---
This is the Aid module.
Host of useful manipulation methods for Furgo Metocean Consultancy - Americas.
"""
import requests
import numpy as np
import pandas as pd
import datetime as dtm

releases = requests.get(
    r'https://api.github.com/repos/myamashita/FUGRO_AID/releases/latest')
lastest = releases.json()['tag_name']
__version__ = '0.1.0'
print(f'This module version is {__version__}.\n'
      f'The lastest version is {lastest}.')


class Aid():
    """ Support variables"""

    def datetime_2matlab(dt):
        """Convert a python datetime to matlab datenum
        >>> Aid.datetime_2matlab(dtm.datetime(2020, 1, 1, 5, 8, 15, 735000))
        737791.214071007
        """
        n = dt.toordinal() + 366
        td = dt - dt.min
        return n + td.seconds / 86400 + td.microseconds / 86400000000

    def datetime64_2matlab(dt):
        """Convert a datetime64[ns] to matlab datenum
        >>> Aid.datetime64_2matlab(np.datetime64('2020-02-12T19:33:15.000000'))
        737833.8147569444
        >>> Aid.datetime64_2matlab(np.datetime64('2020-01-01 05:08:15.735'))
        737791.214071007
        """
        return Aid.datetime_2matlab(dt.astype('M8[ms]').astype('O'))

    def round_dt(dt, resolution=dtm.timedelta(hours=1), up=True):
        """
        >>> Aid.round_dt(dtm.datetime(2020, 2, 12, 19, 33, 15),\
                         dtm.timedelta(minutes=10))
        datetime.datetime(2020, 2, 12, 19, 40)
        >>> Aid.round_dt(dtm.datetime(2020, 1, 1, 5, 8, 15),\
                         dtm.timedelta(minutes=10), False)
        datetime.datetime(2020, 1, 1, 5, 0)
        """
        a, b = divmod((dt - dt.min).seconds, resolution.seconds)
        if up:
            n = dtm.timedelta(seconds=(
                resolution.seconds * (1 - 0**int(b)) - int(b)))
        else:
            n = dtm.timedelta(seconds=(-int(b)))
        return dt + n

    def datenum_2datetime(datenum):
        """
        Convert Matlab datenum into Python datetime.
        :param datenum: Date in datenum format
        :return:        Datetime object corresponding to datenum.
        >>> Aid.datenum_to_datetime(737833.81486962)
        datetime.datetime(2020, 2, 12, 19, 33, 24, 735170)
        """
        return dtm.datetime.fromordinal(int(datenum) - 366) +\
            dtm.timedelta(days=datenum % 1)

    def describe(df, len_exp, df_dir=None):

        if df.ndim == 2 and df.columns.size == 0:
            raise ValueError("Cannot describe a DataFrame without columns")

        df_out = pd.DataFrame(columns=['Return', 'Max', 'Mean', 'Min', 'Std',
                                       'DirMax', 'DateMax', 'DateMin'])

        def describe_1d(s, len_exp, df_out):
            df_out.loc[f'{s.name}', 'Return'] = (s.count() / len_exp) * 100.0
            df_out.loc[f'{s.name}', 'Max'] = s.max()
            df_out.loc[f'{s.name}', 'Mean'] = s.mean()
            df_out.loc[f'{s.name}', 'Min'] = s.min()
            df_out.loc[f'{s.name}', 'Std'] = s.std()
            df_out.loc[f'{s.name}', 'DateMax'] = s.idxmax(skipna=True)
            df_out.loc[f'{s.name}', 'DateMin'] = s.idxmin(skipna=True)
            return df_out

        if df.ndim == 1:
            df_out = describe_1d(df, len_exp, df_out)
        else:
            for key in df.columns:
                df_out = describe_1d(df.loc[:, key], len_exp, df_out)

        if df_dir is not None:
            if not isinstance(df, type(df_dir)):
                raise ValueError("Wrong input types")
            if df.ndim == 1:
                df_out.loc[df.name, 'DirMax'] = df_dir.loc[
                    df_out['DateMax'].values[0]]
            elif df_dir.shape[1] == 1:
                for key in df.columns:
                    df_out.loc[f'{key}', 'DirMax'] = df_dir.loc[
                        df_out.loc[f'{key}', 'DateMax']].values[0]
            elif df.shape[1] == df_dir.shape[1]:
                for n, key in enumerate(df.columns):
                    df_out.loc[f'{key}', 'DirMax'] = df_dir.loc[
                        df_out.loc[f'{key}', 'DateMax']].values[n]

        return df_out

    def highlight_bins(s, range_min=98., range_max=99.):
        '''
        highlight a Series in tree interval.
        orange <= range_min
        yellow <= range_max
        green > range_max
        how to use:
        df.style.apply(highlight_bins)
        '''
        return ['background-color: #FD8060' if v <= range_min
                else 'background-color: #FEE191' if v <= range_max
                else 'background-color: #B0D8A4' for v in s]

    def id2uv(icomp, dcomp, str_conv='cart'):
        """Convert Intensity and Direction to U and V """
        if str_conv == 'cart':  # Vector components in Cartesian convention
            ucomp = icomp * np.cos(np.deg2rad(dcomp))
            vcomp = icomp * np.sin(np.deg2rad(dcomp))
        elif str_conv == 'meteo':  # Vector components in Meteo convention
            ucomp = icomp * np.sin(np.deg2rad(dcomp + 180.))
            vcomp = icomp * np.cos(np.deg2rad(dcomp + 180.))
        elif str_conv == 'ocean':  # Vector components in Ocean convention
            ucomp = icomp * np.sin(np.deg2rad(dcomp))
            vcomp = icomp * np.cos(np.deg2rad(dcomp))
        return ucomp, vcomp

    def uv2id(ucomp, vcomp, str_conv='cart'):
        """Convert U and V to Intensity and Direction"""
        icomp = np.sqrt(ucomp ** 2 + vcomp ** 2)
        if str_conv == 'cart':  # Direction in Cartesian convention
            dcomp = np.rad2deg(np.arctan2(vcomp, ucomp)) % 360.
        elif str_conv == 'meteo':  # Direction in Meteo convention
            dcomp = (np.rad2deg(np.arctan2(ucomp, vcomp)) - 180.) % 360.
        elif str_conv == 'ocean':  # Direction in Ocean convention
            dcomp = np.rad2deg(np.arctan2(ucomp, vcomp)) % 360.
        return icomp, dcomp


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

    def plot_Var(data, marker_type='+', f=None, **kw):
        """Additional Keyword arguments to be passed to the functions:
            bokeh.mk_fig --> 'title'; 'x_axis_label'; 'y_axis_label'; 'height';
                             'width'; 'x_axis_type'; 'axis_label_text_font'
            bokeh.plot_Var --> 'data_color'; 'data_legend'
        """
        if f is None:
            f = Bokeh.mk_fig(
                title=kw.pop('title', 'Initial title'),
                x_axis_label=kw.pop('xlabel', 'Initial xlabel'),
                y_axis_label=kw.pop('ylabel', 'Initial ylabel'),
                height=kw.pop('height', 350), width=kw.pop('width', 1100),
                x_axis_type=kw.pop('x_axis_type', 'datetime'), **kw)
        data_args = {'x': data.index, 'y': data,
                     'color': kw.get('data_color', '#6788B1'),
                     'muted_color': kw.pop('data_color', '#6788B1'),
                     'muted_alpha': 0.2,
                     'line_width': 2}

        _MARKER_SHORTCUTS = {"*": "asterisk",
                             "+": "cross",
                             "o": "circle",
                             "o+": "circle_cross",
                             "o.": "circle_dot",
                             "ox": "circle_x",
                             "oy": "circle_y",
                             "-": "dash",
                             ".": "dot",
                             "v": "inverted_triangle",
                             "^": "triangle",
                             "^.": "triangle_dot"}

        if marker_type in _MARKER_SHORTCUTS:
            marker_type = _MARKER_SHORTCUTS[marker_type]
            G = getattr(f, marker_type)(**data_args, **kw)
        elif marker_type == 'line':
            G = getattr(f, 'line')(**data_args, **kw)
        H = Bokeh._get_HoverTool(
            ycoordlabel=kw.get('legend_label', 'legend_label'), renderer=G)
        f.add_tools(H)
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
        return {'id=': id, 'time>=': start, 'time<': end}

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
