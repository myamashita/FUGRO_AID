import requests
import os
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dtm
import matplotlib.pyplot as plt

releases = requests.get(
    r'https://api.github.com/repos/myamashita/FUGRO_AID/releases/latest')
lastest = releases.json()['tag_name']
__version__ = '0.4.1'
print(f'This module version is {__version__}.\n'
      f'The lastest version is {lastest}.')


class Aid():
    """
    .. hlist::
        :columns: 5

        * :func:`datenum_2datetime`
        * :func:`datetime_2matlab`
        * :func:`datetime64_2matlab`
        * :func:`describe`
        * :func:`highlight_bins`
        * :func:`id2uv`
        * :func:`round_dt`
        * :func:`round_time`
        * :func:`uv2id`

    """

    def datenum_2datetime(dtnum: float) -> dtm.datetime:
        """Convert Matlab datenum into Python datetime

        Args:
            dtnum (``float``): Date in datenum format
        Returns:
            ``datetime.datetime`` object corresponding to ``datenum``
        >>> Aid.datenum_2datetime(737833.81486962)
        datetime.datetime(2020, 2, 12, 19, 33, 24, 735000)
        """
        return dtm.datetime.fromordinal(int(dtnum) - 366) +\
            dtm.timedelta(microseconds=round(dtnum % 1 * 86400000) * 1000)

    def datetime_2matlab(dt: dtm.datetime) -> float:
        """Convert a python datetime to matlab datenum

        Args:
            dt (``datetime.datetime``): Date in datetime format
        Returns:
            ``Datenum`` float corresponding to ``Datetime``
        >>> Aid.datetime_2matlab(dtm.datetime(2020, 1, 1, 5, 8, 15, 735000))
        737791.214071007
        """
        n = dt.toordinal() + 366
        td = dt - dt.min
        return n + td.seconds / 86400 + td.microseconds / 86400000000

    def datetime64_2matlab(dt: np.datetime64) -> float:
        """Convert a datetime64[ns] to matlab datenum

        Args:
            dt (``numpy.datetime64``): Date in datenum format
        Returns:
            ``Datenum`` float corresponding to ```numpy.datetime64``
        >>> Aid.datetime64_2matlab(np.datetime64('2020-02-12T19:33:15.000000'))
        737833.8147569444
        >>> Aid.datetime64_2matlab(np.datetime64('2020-01-01 05:08:15.735'))
        737791.214071007
        """
        return Aid.datetime_2matlab(dt.astype('M8[ms]').astype('O'))

    def describe(df: pd.DataFrame, len_exp: int,
                 df_dir: pd.DataFrame = None) -> pd.DataFrame:
        """Describe a dataframe in statistical table::

        ["Return", "Max", "Mean", "Min", "Std", "DirMax", "DateMax", "DateMin"]

        Args:
            df (``pandas.core.frame.DataFrame``): Entry to describe
            len_exp (int): expected length of entry
            df_dir (``pandas.core.frame.DataFrame``): Optional direction entry
        Returns:
            ``pandas.core.frame.DataFrame`` described
        .. note::
            **Return**:  Percentage of data return ->
            ``(Valid Values / Expected Length) * 100``

            **DirMax** is returned if ``df_dir`` is declared
        """
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
        '''Conditional formatting of a DataFrame or Series.

        It take scalars, DataFrames or Series, and return
        like-indexed DataFrames or Series with CSS "attribute: value"
        pairs for the values.

        .. note::
            There is a 3 colors selected

            orange <= range_min

            yellow <= range_max

            green > range_max

            how to use::

                df.style.apply(Aid.highlight_bins)

                df.style.apply(Aid.highlight_bins, args=[94., 96])

                df.style.apply(Aid.highlight_bins, args=[94., 96],
                               subset=['Return', 'Std', 'DirMax' ])
        '''
        return ['background-color: #FD8060' if v <= range_min
                else 'background-color: #FEE191' if v <= range_max
                else 'background-color: #B0D8A4' for v in s]

    def id2uv(icomp, dcomp, str_conv='cart'):
        """Convert Intensity and Direction to U and V

        :param icomp: Intensity
        :type icomp: ``numpy.ndarray`` or ``FrameOrSeries``
        :param dcomp: Direction at declared convention (degrees).
        :type dcomp: ``numpy.ndarray`` or ``FrameOrSeries``
        :param str_conv: Origin convention
            (ex: 'cart' - Cartesian or 'meteo' - Meteorologic
            or 'ocean' - Oceanographic)
        :type str_conv: str
        :return: + Zonal component, parallel to the abscissa axis
            + Meridional component, parallel to the coordinate axis
        :rtype: ``numpy.ndarray`` or ``FrameOrSeries``

        >>> u, v = Aid.id2uv(
        ...    np.array([1., 1., 1., 1., 1., 1., 1., 1.]),
        ...    np.array([0., 45., 90., 135., 180., 225., 270., 315.]),
        ...    'cart')
        >>> print(u)
        [ 1.00000000e+00  7.07106781e-01  6.12323400e-17 -7.07106781e-01
         -1.00000000e+00 -7.07106781e-01 -1.83697020e-16  7.07106781e-01]
        >>> print(v)
        [ 0.00000000e+00  7.07106781e-01  1.00000000e+00  7.07106781e-01
          1.22464680e-16 -7.07106781e-01 -1.00000000e+00 -7.07106781e-01]
        >>> u, v = Aid.id2uv(
        ...    np.array([1., 1., 1., 1., 1., 1., 1., 1.]),
        ...    np.array([270., 225., 180., 135., 90., 45., 0., 315.]),
        ...    'meteo')
        >>> print(u)
        [ 1.00000000e+00  7.07106781e-01 -2.44929360e-16 -7.07106781e-01
         -1.00000000e+00 -7.07106781e-01  1.22464680e-16  7.07106781e-01]
        >>> print(v)
        [ 3.06161700e-16  7.07106781e-01  1.00000000e+00  7.07106781e-01
         -1.83697020e-16 -7.07106781e-01 -1.00000000e+00 -7.07106781e-01]
        >>> u, v = Aid.id2uv(
        ...    np.array([1., 1., 1., 1., 1., 1., 1., 1.]),
        ...    np.array([90., 45., 0., 315., 270., 225., 180., 135.]),
        ...    'ocean')
        >>> print(u)
        [ 1.00000000e+00  7.07106781e-01  0.00000000e+00 -7.07106781e-01
         -1.00000000e+00 -7.07106781e-01  1.22464680e-16  7.07106781e-01]
        >>> print(v)
        [ 6.12323400e-17  7.07106781e-01  1.00000000e+00  7.07106781e-01
         -1.83697020e-16 -7.07106781e-01 -1.00000000e+00 -7.07106781e-01]
        """
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

    def round_dt(dt, resolution=dtm.timedelta(hours=1), up=True):
        """Round datetime to up or down
        This function is useful to create a expected range of time

        >>> Aid.round_dt(dtm.datetime(2020, 2, 12, 19, 33, 15),
        ...              dtm.timedelta(minutes=10))
        datetime.datetime(2020, 2, 12, 19, 40)
        >>> Aid.round_dt(dtm.datetime(2020, 1, 1, 5, 8, 15),
        ...              dtm.timedelta(minutes=10), False)
        datetime.datetime(2020, 1, 1, 5, 0)
        """
        a, b = divmod((dt - dt.min).seconds, resolution.seconds)
        if up:
            n = dtm.timedelta(seconds=(
                resolution.seconds * (1 - 0**int(b)) - int(b)))
        else:
            n = dtm.timedelta(seconds=(-int(b)))
        return dt + n

    def round_time(dt, dtDelta=dtm.timedelta(seconds=5)):
        """
        Round a datetime object to a multiple of a timedelta

        Args:
            dt (``datetime.datetime``): Date in datetime format
            dtDelta (``datetime.timedelta``): timedelta object
        Returns:
            ``datetime.datetime`` rounded

        >>> Aid.round_time(dtm.datetime(2020, 2, 12, 19, 33, 15),
        ...              dtm.timedelta(minutes=10))
        datetime.datetime(2020, 2, 12, 19, 30)

        .. note::

            Author:

            Thierry Husson 2012 - Use it as you want but don't blame me.

            Stijn Nevens 2014 - Changed to use only datetime objects
        """
        roundTo = dtDelta.total_seconds()
        seconds = (dt - dt.min).seconds
        # // is a floor division, not a comment on following line:
        rounding = (seconds + roundTo / 2) // roundTo * roundTo
        return dt + dtm.timedelta(0, rounding - seconds, -dt.microsecond)

    def uv2id(ucomp, vcomp, str_conv='cart'):
        """Convert U and V to Intensity and Direction

        :param ucomp: Zonal component, parallel to the abscissa axis
        :type ucomp: ``numpy.ndarray`` or ``FrameOrSeries``
        :param vcomp: Meridional component, parallel to the coordinate axis
        :type vcomp: ``numpy.ndarray`` or ``FrameOrSeries``
        :param str_conv: Origin convention
            (ex: 'cart' - Cartesian or 'meteo' - Meteorologic
            or 'ocean' - Oceanographic)
        :type str_conv: str
        :return: + Intensity
            + direction at the desired convention (degrees)
        :rtype: ``numpy.ndarray`` or ``FrameOrSeries``

        >>> i, d = Aid.uv2id(
        ...    np.array([1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0, 0.5]),
        ...    np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1, -0.5]),
        ...    'cart')
        >>> print(d)
        [  0.  45.  90. 135. 180. 225. 270. 315.]
        >>> i, d = Aid.uv2id(
        ...    np.array([1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0, 0.5]),
        ...    np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1, -0.5]),
        ...    'meteo')
        >>> print(d)
        [270. 225. 180. 135.  90.  45.   0. 315.]
        >>> i, d = Aid.uv2id(
        ...    np.array([1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0, 0.5]),
        ...    np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1, -0.5]),
        ...    'ocean')
        >>> print(d)
        [ 90.  45.   0. 315. 270. 225. 180. 135.]
        """
        icomp = np.sqrt(ucomp ** 2 + vcomp ** 2)
        if str_conv == 'cart':  # Direction in Cartesian convention
            dcomp = np.rad2deg(np.arctan2(vcomp, ucomp)) % 360.
        elif str_conv == 'meteo':  # Direction in Meteo convention
            dcomp = (np.rad2deg(np.arctan2(ucomp, vcomp)) - 180.) % 360.
        elif str_conv == 'ocean':  # Direction in Ocean convention
            dcomp = np.rad2deg(np.arctan2(ucomp, vcomp)) % 360.
        return icomp, dcomp


class Bokeh(object):
    """
    .. hlist::
        :columns: 5

        * :func:`mk_fig`
        * :func:`plot_Colour_Flood`
        * :func:`plot_IntDir`
        * :func:`plot_QCresults`
        * :func:`plot_Stick`
        * :func:`plot_Var`
        * :func:`set_fig`

    """

    import bokeh.palettes as bp
    from bokeh.plotting import figure
    from bokeh.models import ColorBar, HoverTool, Range1d, LinearAxis
    from bokeh.transform import linear_cmap
    from bokeh.models.formatters import DatetimeTickFormatter

    def mk_fig(title='Initial title', x_axis_label='Initial xlabel',
               y_axis_label='Initial ylabel', height=350, width=1100,
               x_axis_type='datetime', **kw):
        """Method to simplifies figure creation

        Args:

          title (``str``): plot_title
          x_axis_label (``str``): A label for the x-axis
          y_axis_label (``str``): A label for the y-axis
          height (``int``): plot_height
          width (``int``): plot_width
          x_axis_type (``str``): The type of the x-axis
          **kw : Additional Keyword arguments to be passed to set_fig::

            title_fontsize='16pt', title_fontstyle='normal', axislabel_fontsize='14pt',
            axislabel_fontstyle='normal', axis_label_text_font='segoe ui'

        Returns:
        ``Bokeh.figure``
        """  # nopep8
        sizing_mode = kw.get('sizing_mode', 'scale_both')
        f = Bokeh.figure(title=title, height=height, width=width,
                         x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                         x_axis_type=x_axis_type, sizing_mode=sizing_mode)
        return Bokeh.set_fig(f, **kw)

    def plot_Colour_Flood(data, X_tooltip='Date', Y_tooltip='Altitude',
                          Z_tooltip='Intensity', vmin=0, vmax=100,
                          f=None, **kw):
        """Function to plot colour flood

        Args:
            data (``dict``): dict with keys::

                'image': data in array_like, 'dt':The range in x-coordinates,
                'x': The x-coordinates to locate the image anchors,
                'y': The y-coordinates to locate the image anchors,
                'dw': The width of the plot region that the image will occupy,
                'dh': The height of the plot region that the image will occupy

            str, X_tooltip, Y_tooltip, Z_tooltip : string to tooltip
            float, vmin, vmax : min/max values in colorbar range
            f (``Bokeh.figure``): optional figure
            **kw: Additional Keyword arguments to be passed to the functions:

                function: bokeh.mk_fig::

                    title, x_axis_label, y_axis_label, height, width, x_axis_type, axis_label_text_font

                function: bokeh.plot_Colour_Flood::

                    'palette': default pallete "Bokeh.bp.Turbo256"
                    'axislabel_fontsize': colorbar font size "14pt"

        Returns:
            ``Bokeh.figure``

        Example::

          N = 500
          x = np.linspace(0, 10, N)
          xx, yy = np.meshgrid(x, x)
          d = np.sin(xx) * np.cos(yy)
          X = pd.date_range(dtm.datetime(2020,2,12,19,30), 
                            dtm.datetime(2020,2,12,19,30) + dtm.timedelta(minutes=499), freq='Min')
          XX, YY = np.meshgrid(X, x)
          data = dict(image=[d], dt=[XX], x=[dtm.datetime(2020,2,12,19,30)], y=[0], dw=[500*60000], dh=[10])
          fig = Bokeh.plot_Colour_Flood(data, Y_tooltip='y', Z_tooltip='value', vmin=-1, vmax=1,
                                        title='Example', x_axis_label='')

        """  # nopep8
        if f is None:
            f = Bokeh.mk_fig(
                title=kw.pop('title', 'Initial title'),
                x_axis_label=kw.pop('x_axis_label', 'Initial xlabel'),
                y_axis_label=kw.pop('y_axis_label', 'Initial ylabel'),
                height=kw.pop('height', 350), width=kw.pop('width', 1100),
                x_axis_type=kw.pop('x_axis_type', 'datetime'), **kw)

        f.add_tools(Bokeh._get_HoverTool(
            X_tooltip, Y_tooltip, Z_tooltip, image=True))
        mapper = Bokeh.linear_cmap(field_name='y', palette=kw.get(
            'palette', Bokeh.bp.Turbo256), low=vmin, high=vmax)
        f.image(source=data, image='image', x='x', y='y', dw='dw', dh='dh',
                color_mapper=mapper['transform'])
        cbar = Bokeh.ColorBar(
            color_mapper=mapper['transform'], width=18, location=(0, 0),
            major_tick_line_color='#000000', label_standoff=12,
            major_label_text_font_size=kw.get('axislabel_fontsize', '14pt'))
        f.add_layout(cbar, 'right')
        return f

    def plot_IntDir(dfI, dfD, f=None, **kw):
        """Function to plot intensity and direction

        Args:
            dfI(``pandas.core.series.Series``): Intensity series
            dfD(``pandas.core.series.Series``): Direction series
            f(``Bokeh.figure``): optional figure
            **kw: Additional Keyword arguments to be passed to the functions::
                function: bokeh.mk_fig::

                    title, x_axis_label, y_axis_label, height, width, x_axis_type, axis_label_text_font

                function: bokeh.plot_IntDir::

                    "Int_label": Y-label in left side plot
                    "Int_color":  Intensity color in plot
                    "Int_legend": Intensity legend
                    "Dir_label": Y-label in rigth side plot
                    "Dir_color": Direction color in plot
                    "Dir_legend": Direction legend

        Returns:
            ``Bokeh.figure``
        """  # nopep8
        if f is None:
            f = Bokeh.mk_fig(
                title=kw.pop('title', 'Initial title'),
                x_axis_label=kw.pop('x_axis_label', 'Initial xlabel'),
                y_axis_label=kw.pop('y_axis_label', ''),
                height=kw.pop('height', 350), width=kw.pop('width', 1100),
                x_axis_type=kw.pop('x_axis_type', 'datetime'), **kw)
        Int_color = kw.get('Int_color', '#6788B1')
        Int_legend = kw.get('Int_legend', 'Intensity')
        f.yaxis.axis_label = kw.get('Int_label', 'Intensity [m/s]')
        G = f.line(dfI.index, dfI, color=Int_color, muted_color=Int_color,
                   muted_alpha=0.2, line_width=2, legend_label=Int_legend)
        H = Bokeh._get_HoverTool(ycoordlabel=Int_legend, renderer=G)
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

    def plot_QCresults(data, var_name, results, title, test_name, **kw):
        """Function to plot QC results of IOOS QARTOD tests

        Args:
            data (``pandas.core.frame.DataFrame``): Raw data 
            var_name (``str``): Name of the variable
            results (``OrderedDict``): The results of a QcConfig.run
            title (``str``): Title of the plot
            test_name (``str``): Name of the test to plot
            **kw : Additional Keyword arguments to be passed to the functions:

                function: bokeh.mk_fig::

                    x_axis_label, height, width, x_axis_type

        Returns:
            ``Bokeh.figure``
        """  # nopep8

        time = data.index
        obs = data[var_name]
        qc_test = results['qartod'][test_name]

        qc_pass = np.ma.masked_where(qc_test != 1, obs)
        qc_suspect = np.ma.masked_where(qc_test != 3, obs)
        qc_fail = np.ma.masked_where(qc_test != 4, obs)
        qc_notrun = np.ma.masked_where(qc_test != 2, obs)
        qc_noteval = np.ma.masked_where(qc_test != 9, obs)

        f = Bokeh.mk_fig(
            title=f'{test_name}: {title}',
            x_axis_label=kw.pop('x_axis_label', 'Initial xlabel'),
            y_axis_label='Observation Value',
            height=kw.pop('height', 350), width=kw.pop('width', 1100),
            x_axis_type=kw.pop('x_axis_type', 'datetime'))

        f.line(time, obs, legend_label='obs',
               color='#A6CEE3', muted_alpha=0.1)
        f.circle(time, qc_notrun, size=2, legend_label='qc not run',
                 color='gray', alpha=0.3, muted_alpha=0.2)
        f.circle(time, qc_pass, size=4, legend_label='qc pass',
                 color='green', alpha=0.5, muted_alpha=0.2)
        f.circle(time, qc_suspect, size=4, legend_label='qc suspect',
                 color='orange', alpha=0.7, muted_alpha=0.2)
        f.circle(time, qc_fail, size=6, legend_label='qc fail',
                 color='red', alpha=1.0, muted_alpha=0.2)
        f.circle(time, qc_noteval, size=6, legend_label='qc not eval',
                 color='gray', alpha=1.0, muted_alpha=0.2)

        f.legend.click_policy = "mute"
        f.add_layout(f.legend[0], 'right')
        f.add_tools(Bokeh.HoverTool(
            tooltips=[("index", "$index"),
                      ("Date", "@x{  %Y-%m-%d %H:%M}"), ("y", '$y')],
            formatters={'@x': 'datetime'}, line_policy='nearest'))
        return f

    def plot_Stick(dfI, dfD, dw, str_conv='ocean', f=None, add_cbar=True,
                   **kw):
        """Function to plot vector.
        Each stick plot contains three pieces of information: direction, time, and magnitude (strength).
        Default ocean convention

        Args:
            dfI(``pandas.core.series.Series``): Intensity series
            dfD(``pandas.core.series.Series``): Direction series in **degrees**
            dw (``int``): width of data units (measured in the x-direction)
            str_conv (``str``): 'meteo' (from) or 'ocean' (towards)
            f(``Bokeh.figure``): optional figure
            add_cbar (``bool``): True to add color bar
            **kw: Additional Keyword arguments to be passed to the functions::
                function: bokeh.mk_fig::

                    title, x_axis_label, y_axis_label, height, width, x_axis_type, axis_label_text_font

                function: bokeh.plot_stick::

                    'vmin': minumum value in colorbar range
                    'vmax': maximum value in colorbar range
                    'y_coord': The y-coordinates to start the sticks

        Returns:
            ``Bokeh.figure``
        """  # nopep8
        vmin = kw.pop('vmin', min(dfI))
        vmax = kw.pop('vmax', max(dfI))
        y_coord = kw.pop('y_coord', 0)

        if str_conv == 'meteo':  # Direction in Meteo convention FROM
            dfD = (-dfD + 270) % 360
        elif str_conv == 'ocean':  # Direction in Ocean convention TOWARDS
            dfD = (-dfD + 90) % 360

        if f is None:
            f = Bokeh.mk_fig(
                title=kw.pop('title', 'Initial title'),
                x_axis_label=kw.pop('x_axis_label', 'Initial xlabel'),
                y_axis_label=kw.pop('y_axis_label', 'Initial ylabel'),
                height=kw.pop('height', 350), width=kw.pop('width', 1100),
                x_axis_type=kw.pop('x_axis_type', 'datetime'), **kw)

        mapper = Bokeh.linear_cmap(
            field_name='length',
            palette=kw.get('palette', Bokeh.bp.Turbo256),
            low=vmin, high=vmax * dw * 200)

        f.ray(dfI.index, np.repeat(y_coord, dfI.size), length=dfI * dw * 200,
              angle=dfD, angle_units="deg", color=mapper, line_width=2)
        cbar_map = Bokeh.linear_cmap(
            field_name='length',
            palette=kw.get('palette', Bokeh.bp.Turbo256),
            low=vmin, high=vmax)
        cbar = Bokeh.ColorBar(
            color_mapper=cbar_map['transform'], width=18, location=(0, 0),
            major_tick_line_color='#000000', label_standoff=12,
            major_label_text_font_size='14pt')
        if add_cbar:
            f.add_layout(cbar, 'right')
        return f

    def plot_Var(data, marker_type='+', f=None, **kw):
        """Function to plot a single variable. Using a marker type or line

        Args:
            data(``pandas.core.series.Series``): Series to plot
            marker_type (``str``): "*", "+", "o", "o+", "o.", "ox", "oy", "-", ".", "v", "^", "^."
            f (``Bokeh.figure``): optional figure
            **kw: Additional Keyword arguments to be passed to the functions:

                function: bokeh.mk_fig::

                    title, x_axis_label, y_axis_label, height, width, x_axis_type, axis_label_text_font

                function:: bokeh.plot_Var::

                    'data_color': color of line or marker
                    'legend_label': legend label

        Returns:
            ``Bokeh.figure``
        """  # nopep8
        if f is None:
            f = Bokeh.mk_fig(
                title=kw.pop('title', 'Initial title'),
                x_axis_label=kw.pop('x_axis_label', 'Initial xlabel'),
                y_axis_label=kw.pop('y_axis_label', 'Initial ylabel'),
                height=kw.pop('height', 350), width=kw.pop('width', 1100),
                x_axis_type=kw.pop('x_axis_type', 'datetime'), **kw)
        data_args = {'x': data.index, 'y': data,
                     'color': kw.get('data_color', '#6788B1'),
                     'muted_color': kw.pop('data_color', '#6788B1'),
                     'muted_alpha': 0.2, 'line_width': 2}

        _MARKER_SHORTCUTS = {
            "*": "asterisk", "+": "cross", "o": "circle", "o+": "circle_cross",
            "o.": "circle_dot", "ox": "circle_x", "oy": "circle_y",
            "-": "dash", ".": "dot", "v": "inverted_triangle", "^": "triangle",
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

    def set_fig(f, title_fontsize='16pt', title_fontstyle='normal',
                axislabel_fontsize='14pt', axislabel_fontstyle='normal', **kw):
        """Function to set the figure

        Args:
          f (``Bokeh.figure``): figure to be set
          title_fontsize (``str``): set title font size
          title_fontstyle (``str``): set title font style
          axislabel_fontsize (``str``): set axis label font size
          axislabel_fontstyle (``str``): set axis label font style
          **kw: Additional Keyword arguments to be passed to the functions:

              function: bokeh.set_fig::

                axis_label_text_font (str): set axis label text font. Default: 'segoe ui'

        Retunrs:
            ``Bokeh.figure``
        """  # nopep8
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
    """Create a Class instance to facilitate connection and requests from FUGRO Erddap

        Arguments given to the class intantiation operator::

          server (str): Default server "http://10.1.1.17:8080/erddap" (FUGRO ERDDAP)
          protocol (str): Default protocol "tabledap"
          response (str): Default response "csv"
          dataset_id (str): set dataset id
          constraints (dict): set constraints
          variables (list): set variables to requests

    .. hlist::
        :columns: 5

        * :func:`to_pandas`
        * :func:`vars_in_dataset`
    """  # nopep8

    def __init__(self, server='http://10.1.1.17:8080/erddap',
                 protocol='tabledap', response='csv', dataset_id=None,
                 constraints=None, variables=None):
        self._base = Erddap._erddap_instance(server, protocol, response)
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

    def _erddap_instance(server='http://10.1.1.17:8080/erddap',
                         protocol='tabledap', response='csv'):
        """Create a erddap instance"""
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
        """Function to get data from Erddap and return in ``pandas.core.frame.DataFrame``

        index column with ``dtype="datetime64[ns]"``

        >>> e = Erddap(dataset_id='geosOceanorMet',
        ...    constraints={'time>=': dtm.datetime.utcnow()-dtm.timedelta(hours=1)})
        >>> df = e.to_pandas()
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>
        """  # nopep8
        def dateparser(x):
            return dtm.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")
        kw = {'index_col': 'time', 'date_parser': dateparser,
              'skiprows': [1], 'response': 'csv'}
        return self._base.to_pandas(**kw)

    def vars_in_dataset(self, dataset):
        """Function to get variables in a specific dataset

        >>> e = Erddap()
        >>> var = e.vars_in_dataset('geosOceanorAquadopp')
        >>> var.sort()
        >>> print(var)
        ['AqDir', 'AqSpd', 'Label', 'Misc', 'Source', 'depth', 'id', 'latitude', 'longitude', 'time']
        >>> var = e.vars_in_dataset('geosOceanorConductivity')
        >>> var.sort()
        >>> print(var)
        ['Conductivity', 'Label', 'Misc', 'Source', 'depth', 'id', 'latitude', 'longitude', 'time']
        """  # nopep8
        v = self._base._get_variables(dataset)
        v.pop('NC_GLOBAL', None)
        return [i for i in v]


class Mat(object):
    """Class to facilitate matlab operations for FUGRO Softwares

    .. hlist::
        :columns: 5

        * :func:`fmdm_meta`
        * :func:`mets_structure`
        * :func:`mets_meta`
        * :func:`mets_data`
        * :func:`merge_local_flag`
        * :func:`split_local_flag`
        * :func:`read_mets_data`
        * :func:`export_qc_metis`
        * :func:`save`

    """  # nopep8

    from scipy.io import savemat, loadmat

    def fmdm_meta(lat=0, lon=0, waterdepth=0, Contract='Contract'):
        """Create a dictionary with metadata necessary for FMDM.

        Args:
            lat(``float``): Latitude
            lon(``float``): Longitude
            waterdepth(``float``): waterdepth
            Contract(``str``): Contract
        Returns:
            ``dict`` Data corresponding to ``metadata``
        """  # nopep8
        Data = {}
        Data['Metadata'] = {}
        Data['Metadata']['Lat'] = lat
        Data['Metadata']['Lon'] = lon
        Data['Metadata']['Waterdepth'] = waterdepth
        Data['Metadata']['Contract'] = Contract
        return Data

    def mets_structure(filename='raw_data.mat', time=[0], data_type='ADCP'):
        """Create a dictionary with structure necessary for METS.

        Args:
            filename(``str``): filename 
            time(``list``): List of matlab datenum
            data_type(``str``): output data type for METS
        Returns:
            ``dict`` Data corresponding to ``METS structure``
        """  # nopep8

        DTYPES = ['ADCP', 'MCAL', 'CCAL', 'TCAL']
        if data_type not in DTYPES:
            raise ValueError("Invalid data type. Expected one of: %s" % DTYPES)
        name = os.getcwd() + f'\\{filename}'
        Data = {}
        Data["qc"] = {}
        Data["data"] = {}
        Data["metadata"] = {}
        Data["time"] = time
        Data["filename"] = name
        Data["irec"] = np.arange(len(time)) + 1  # indice_reccord
        Data["type"] = data_type
        Data["metadata"]['SDOFC'] = time[0]
        Data["metadata"]['EDOFC'] = time[-1]
        Data["metadata"]["NREC"] = len(time)
        Data["qc"]['flag'] = {}
        Data["qc"]['flag']['global'] = np.array(
            [['FalseStart', 1], ['FalseEnd', 2], ['General', 3]], dtype=object)
        Data["qc"]['flag']['local'] = np.array(
            [['PGood', 1], ['Spline', 2], ['ErrVel', 3], ['Gao', 4],
             ['VelSp', 5], ['OK', 6], ['Block', 7], ['Manual', 8]],
            dtype=object)
        Data["qc"]['global'] = np.zeros(len(time), dtype=np.uint8)
        return Data

    def mets_meta(Data={'metadata': {}}, **kw):
        """
        Update or generate metadata in METS structure.

        ``One-to-one relationship. keyword arguments and METS arguments.``

        TITLE = Job Name

        REFNO = Contract Number

        INSTR = Instrument Name

        INTRAT = Sample Rate

        MAGVAR = MagVar (-West/+East)

        POSL = Position

        POSD = Location (Descriptive)

        ZONE = Time Zone

        MSERS = Sensor. Serial Nros

        HTMSL = Height of ground

        HEIGHT = Height of Sensor

        SENS1 = Type of First extra sensor

        SENS2 = Type of second extra sensor

        SENS3 = Type of third extra sensor

        SPLIN = Splined

        KFREQ = Head Frequency (kHz)

        DATUM = DATUM

        WDEP = Water Depth (m)

        SDEP = Current Meter Sensor Height (m above bed)

        SERNO = Instrument Serial Number

        NDEP = Dephts array

        NCELL = Number of bins

        Args:
            Data(``dict``): Dictionary with METS structure 
            **kw: Additional Keyword arguments to be passed to the functions::

                TITLE, REFNO, INSTR, INTRAT, MAGVAR, POSL, POSD, ZONE,
                MSERS, HTMSL, HEIGHT, SENS1, SENS2, SENS3, SPLIN, KFREQ,
                DATUM, WDEP, SDEP, SERNO, NDEP, NCELL 

         Returns:
            ``dict`` Data corresponding to ``METS structure``

        """  # nopep8

        types = {
            'ADCP': ['KFREQ', 'SERNO', 'SPLIN', 'WDEP', 'NDEP', 'NCELL'],
            'MCAL': ['HEIGHT', 'HTMSL', 'MSERS', 'SENS1', 'SENS2', 'SENS3'],
            'CCAL': ['WDEP', 'SDEP', 'SERNO'],
            'TCAL': ['DATUM', 'HEIGHT', 'SDEP', 'SERNO', 'WDEP']}
        if 'metadata' not in Data:
            raise KeyError("Invalid METS structure. Missing: metadata")
        for k in types[Data['type']]:
            try:
                Data["metadata"][k] = kw.pop(k)
            except KeyError:
                raise KeyError(
                    f'Keyword Argument REQUIRED {types[Data["type"]]}')

        Data["metadata"]["TITLE"] = kw.pop('TITLE', '')
        Data["metadata"]["REFNO"] = kw.pop('REFNO', '')
        Data["metadata"]["INSTR"] = kw.pop('INSTR', '')
        Data["metadata"]["INTRAT"] = kw.pop('INTRAT', 6000)
        Data["metadata"]["MAGVAR"] = kw.pop('MAGVAR', 0)
        Data["metadata"]["POSL"] = kw.pop('POSL', '')
        Data["metadata"]["POSD"] = kw.pop('POSD', '')
        Data["metadata"]["ZONE"] = kw.pop('ZONE', 'UTC')

        return Data

    def mets_data(name, units, param_type, isBined, value, Data={'data': {}}):

        if 'data' not in Data:
            raise KeyError("Invalid METS structure. Missing: data")
        struct = {}
        struct["units"] = units
        struct["type"] = param_type
        struct["isBinned"] = bool(isBined)
        struct["value"] = value
        Data['data'][name] = struct

        if 'localMapping' not in Data['qc']:
            Data['qc']['localMapping'] = np.array([[name, 1]], dtype=object)
        cols = (np.size(value[0])) ** isBined
        loc_arr = np.array(
            [np.zeros(len(value), dtype=np.uint16), ] * cols).transpose()
        if 'local' not in Data['qc']:
            qc_local = np.empty((1, 1), dtype=object)
            qc_local[0, 0] = loc_arr
            Data['qc']['local'] = qc_local
        if name in Data['qc']['localMapping']:
            idx = np.where(Data['qc']['localMapping'] == name)[0][0]
            Data['qc']['local'][0, idx] = loc_arr
        else:
            n = np.size(Data["qc"]["local"])
            qc_local = np.empty((1, n + 1), dtype=object)
            qc_local[0, :n] = Data["qc"]["local"][0]
            qc_local[0, n] = loc_arr
            Data['qc']['local'] = qc_local
        if name not in Data['qc']['localMapping']:
            Lmap = Data['qc']['localMapping']
            Data['qc']['localMapping'] = np.append(
                Lmap, [name, name]).reshape((len(Lmap) + 1, 2))
            Data['qc']['localMapping'][len(Lmap), 1] = Lmap[-1, 1] + 1
        return Data

    def merge_local_flag(data, merge=[]):

        Lmap = Data['qc']['localMapping']
        for i in merge:
            a = [np.where(Lmap == name)[0][0] for name in i]
            Data['qc']['localMapping'][a, 1] = Lmap[a[0], 1]

        values = set(Data['qc']['localMapping'][:, 1])
        qc_local = np.empty((1, len(values)), dtype=object)
        for new, old in enumerate(values, start=1):
            b = np.where(Data['qc']['localMapping'] == old)[0]
            Data['qc']['localMapping'][b, 1] = new
            qc_local[0, new - 1] = Data['qc']['local'][0, old - 1]

        Data['qc']['local'] = qc_local
        return Data

    def split_local_flag(data, split=[]):

        Lmap = Data['qc']['localMapping']
        n = Lmap[-1, 1] + 1
        for i in split:
            a = [np.where(Lmap == name)[0][0] for name in i]
            old = Lmap[a, 1][0]
            Data['qc']['localMapping'][a, 1] = n
            nl = np.size(Data["qc"]["local"])
            qc_local = np.empty((1, nl + 1), dtype=object)
            qc_local[0, :nl] = Data["qc"]["local"][0]
            qc_local[0, nl] = Data["qc"]["local"][0][old - 1]
            n += 1
            Data['qc']['local'] = qc_local
        values = set(Data['qc']['localMapping'][:, 1])
        qc_local = np.empty((1, len(values)), dtype=object)
        for new, old in enumerate(values, start=1):
            b = np.where(Data['qc']['localMapping'] == old)[0]
            Data['qc']['localMapping'][b, 1] = new
            qc_local[0, new - 1] = Data['qc']['local'][0, old - 1]
        Data['qc']['local'] = qc_local

        return Data

    def read_mets_data(matfile):
        """Read matfile (with METS structure) insert into dataset.

        Args:
            matfile(``str``): Name of the raw .mat file
        Returns:
            ``xarray.core.dataset.Dataset`` Dataset corresponding to ``matfile``
        """  # nopep8
        MAT = Mat.loadmat(matfile)
        vfunc = np.vectorize(Aid.datenum_2datetime)
        QC = MAT['dataset']['qc'][0, 0]
        DATA = MAT['dataset']['data'][0, 0]
        time = MAT['dataset']['time'][0, 0].ravel()
        META = MAT['dataset']['metadata'][0, 0]
        DS = xr.Dataset(data_vars={
            'global_mask_0': (('date'), QC['global'][0, 0].ravel())},
            coords={'date': vfunc(time)})
        qc_flag = {}
        for i, j in QC['localMapping'][0, 0]:
            qc_flag.update({f'{i[0]}': j[0, 0]})
        for i in DATA.dtype.names:
            name = f'local_mask_{qc_flag[i]}'
            flag = QC['local'][0, 0][0, int(qc_flag[i]) - 1]
            if DATA[i].ravel()[0]['isBinned'][0, 0].ravel():
                level = MAT['dataset']['metadata'][0, 0]['NDEP'][0, 0].ravel()
                ds = xr.Dataset(data_vars={
                    i: (('date', 'level'), DATA[i].ravel()[0]['value'][0, 0]),
                    name: (('date', 'level'), flag)},
                    coords={'date': vfunc(time), 'level': level})
            else:
                ds = xr.Dataset(data_vars={
                    i: (('date'), DATA[i].ravel()[0]['value'][0, 0].ravel()),
                    name: (('date'), flag.ravel())},
                    coords={'date': vfunc(time)})
            attrs = DS.attrs
            attrs.update(
                {f'{i}_unit': DATA[i].ravel()[0]['units'][0, 0].ravel()[0]})
            if name not in attrs:
                attrs.update({name: f'{i}'})
            else:
                attrs.update({name: f'{attrs[name]}, {i}'})
            DS = xr.merge([DS, ds])
            DS.attrs = attrs
        for i in META.dtype.names:
            attrs = DS.attrs
            values = META[i].ravel()[0].ravel()
            if len(values) > 1:
                attrs.update({f'{i}': values})
            else:
                attrs.update({f'{i}': values[0]})
            DS.attrs = attrs
        Qc._get_false_start_end_time(DS)
        return DS

    def export_qc_metis(raw_matfile, ds, qc_matfile):
        """Save a MATLAB-style .mat file (with METS structure).

        Args:
            raw_matfile(``str``): Name of the raw .mat file
            ds(``xarray.core.dataset.Dataset``): Dataset with matfile variables and flags
            qc_matfile(``str``): Name of the output qced .mat file
        """  # nopep8
        MAT = Mat.loadmat(raw_matfile, mat_dtype=True)
        new = MAT['dataset']
        for i in ds.data_vars:
            if 'global_mask' in i:
                new["qc"][0, 0]['global'][0, 0] = ds[i].values
            elif i.startswith('local_mask'):
                n = int(i.split('_')[-1])
                new['qc'][0, 0]['local'][0, 0][0, n - 1] = ds[i].values
            else:
                new['data'][0, 0][i][0, 0]['value'][0, 0] = ds[i].values
                new['data'][0, 0][i][0, 0]['units'][0, 0] = ds.attrs.get(
                    f'{i}_unit')
        Mat.save(qc_matfile, {'dataset': MAT['dataset']})

    def save(fname, mdict):
        """Save a dictionary of names and arrays into a MATLAB-style .mat file.

        Args:
            fname(``str``): Name of the .mat file
            mdict(``dict``): Dictionary from which to save matfile variables
        """  # nopep8
        Mat.savemat(fname, mdict, do_compression=True, oned_as='column')


class Hycom(object):
    """Class to get Hycom datasets

    .. hlist::
        :columns: 5

        * :func:`gom_url`
        * :func:`gom_ds`
        * :func:`subset_gom`
        * :func:`loop_current_time_rng`
        * :func:`plot_gom_LC`
        * :func:`plot_LC`
        * :func:`plot_gom_sst`
        * :func:`plot_gom_ssh`
        * :func:`plot_gom_vel`
        * :func:`plot_map`

    """  # nopep8

    import cftime
    import cartopy.crs as ccrs
    plt.rcParams['font.sans-serif'] = "Segoe ui"

    class nf(float):
        def __repr__(self):
            s = f'{self:.1f}'
            return f'{self:.0f}' if s[-1] == '0' else s

    def gom_url(dt: dtm.datetime) -> str:
        """Return url experiment

        Args:
            dt (``datetime.datetime``): Date in datetime format
        Returns:
            ``str`` URL string corresponding to ``Datetime``
        >>> Hycom.gom_url(dtm.datetime(2020, 1, 1))
        'https://tds.hycom.org/thredds/dodsC/GOMu0.04/expt_90.1m000'
        """
        lim = [dtm.datetime(2013, 1, 1), dtm.datetime(2014, 4, 1),
               dtm.datetime(2019, 1, 1)]
        if dt < lim[0]:
            url = "https://tds.hycom.org/thredds/dodsC/GOMu0.04/expt_50.1"
        elif (dt >= lim[0]) & (dt < lim[1]):
            url = "https://tds.hycom.org/thredds/dodsC/GOMl0.04/expt_31.0"
        elif (dt >= lim[1]) & (dt < lim[2]):
            url = 'https://tds.hycom.org/thredds/dodsC/GOMl0.04/expt_32.5'
        else:
            url = "https://tds.hycom.org/thredds/dodsC/GOMu0.04/expt_90.1m000"
        return url

    def gom_ds(dt: dtm.datetime) -> xr.Dataset:
        """Return xarray Dataset experiment in GOM

        Args:
            dt (``datetime.datetime``): Date in datetime format
        Returns:
            ``xarray.core.dataset.Dataset`` using respective experiment
        """
        if ((dt < dtm.datetime(2013, 1, 1, 0)) |
                (dt >= dtm.datetime(2019, 1, 1))):
            ds = xr.open_dataset(Hycom.gom_url(dt), decode_times=False)
            time_fix = Hycom.cftime.num2date(
                ds.time, units=ds.time.units, calendar='gregorian')
            ds = ds.assign_coords(time=time_fix)
            ds = ds.rename({'surf_el': 'ssh'})
        else:
            ds = xr.open_dataset(Hycom.gom_url(dt), decode_times=True)
            ds = ds.rename({'MT': 'time', 'Depth': 'depth', 'u': 'water_u',
                            'v': 'water_v', 'temperature': 'water_temp',
                            'Latitude': 'lat', 'Longitude': 'lon'})
        return ds

    def _get_grid(lon, lat, grid):
        if isinstance(grid, int):
            kw = {'lon': slice(lon - (grid / 2), lon + (grid / 2)),
                  'lat': slice(lat - (grid / 2), lat + (grid / 2))}
        elif len(grid) == 4:
            kw = {'lon': slice(grid[0], grid[1]),
                  'lat': slice(grid[2], grid[3])}
        elif len(grid) == 2:
            kw = {'lon': slice(lon - (grid[0] / 2), lon + (grid[0] / 2)),
                  'lat': slice(lat - (grid[1] / 2), lat + (grid[1] / 2))}
        return kw

    def subset_gom(dt, lon, lat, grid=2,
                   var=['water_u', 'water_v', 'water_temp', 'ssh']):
        """Return a subset xarray in GOM with one timestamp and some variables

        Args:
            dt (``datetime.datetime``): Date in datetime format
            lon (``float``): Central longitude
            lat (``float``): Central latitude
            grid (``float | list``): Grid between central point
        Returns:
            ``xarray.core.dataset.Dataset`` using respective constraints
        """
        kw = Hycom._get_grid(lon, lat, grid)
        ds = Hycom.gom_ds(dt)
        ds_out = ds.sel(time=dt, **kw)
        return ds_out[var]

    def loop_current_time_rng(start, end, lon, lat, grid=2, **kw):

        def mk_plots(start, end, eof, lon, lat, grid, fol, **kw):
            ds = Hycom.gom_ds(start)
            while start < eof:
                ds_time = ds.sel(time=start, method='nearest')
                kwg = Hycom._get_grid(lon, lat, grid)
                ds_ = ds_time.sel(**kwg)
                fig, ax = Hycom.plot_LC(ds_, lon, lat, **kw)
                plt.savefig(f'{fol}/LC_gom_{start.strftime("%Y%m%dT%HZ")}.png',
                            bbox_inches='tight', pad_inches=0.05)
                plt.close()
                start += dtm.timedelta(days=1)
                if (start > end) | (start > eof):
                    break
            return start

        fol = kw.get(
            'folder', dtm.datetime.utcnow().strftime("%Y%m%d_%H%M%SUTC"))
        if not os.path.exists(fol):
            os.mkdir(fol)

        eof_time = [dtm.datetime(2013, 1, 1, 0), dtm.datetime(2014, 4, 1),
                    dtm.datetime(2019, 1, 1), dtm.datetime.utcnow()]
        A = [start < i for i in eof_time]
        B = [end >= i for i in [dtm.datetime(1993, 1, 1, 0)] + eof_time[:-1]]
        DS = [a * b for a, b in zip(A, B)]

        for i, ds in zip(eof_time, DS):
            if ds:
                start = mk_plots(start, end, i, lon, lat, grid, fol, **kw)

    def plot_gom_LC(time, lon, lat, save=True, **kw):
        ds = Hycom.subset_gom(time, lon, lat, kw.pop('grid', 6))
        fig, ax = Hycom.plot_LC(ds, lon, lat, **kw)
        if save:
            plt.savefig(f'LC_gom_{time.strftime("%Y%m%dT%HZ")}.png',
                        bbox_inches='tight', pad_inches=0.05)

    def plot_LC(ds, lon, lat, **kw):

        pooling = kw.pop('pooling', [])
        points = kw.pop('points', True)
        quiver = kw.pop('quiver', True)
        cmap = kw.pop('cmap', plt.cm.bwr)
        vmin = kw.pop('vmin', ds['ssh'].min(dim=("lat", "lon")).item())
        vmax = kw.pop('vmax', ds['ssh'].max(dim=("lat", "lon")).item())
        temp = ds['water_temp'].sel(depth=200.0)

        fig, ax = Hycom.plot_map(
            ds['ssh'], ds['ssh'].attrs['units'], cmap, vmin, vmax)
        ax.set_title(f'LC GOM {ds.time.dt.strftime("%Y-%m-%d %HZ").item()}')

        if points:
            ax = Hycom._points(ax, lon, lat, pooling)

        CS = ax.contour(temp.lon, temp.lat, temp, [20],
                        linewidths=3, colors='darkgreen')
        CS.levels = [Hycom.nf(val) for val in CS.levels]
        ax.clabel(CS, CS.levels, inline=True, fmt=r'%r degC', fontsize=13)
        CS.levels = [Hycom.nf(val) for val in CS.levels]
        CS = ax.contour(temp.lon, temp.lat, ds['ssh'], [0.17],
                        linewidths=3, colors='k')
        ax.clabel(CS, CS.levels, inline=True, fmt=r'%r m', fontsize=12)

        ax.legend([plt.Rectangle((0, 0), 1, 1, fc='k'),
                   plt.Rectangle((0, 0), 1, 1, fc='darkgreen')],
                  ['SSH 17 cm', '20 deg C at 200m'], loc='upper right',
                  framealpha=0.9)
        if quiver:
            u = ds['water_u'].sel(depth=0)
            v = ds['water_v'].sel(depth=0)
            ax = Hycom._quiver(ax, u, v)
        return fig, ax

    def plot_gom_sst(time, lon, lat, save=True, **kw):

        ds = Hycom.subset_gom(
            time, lon, lat, kw.pop('grid', 2), ['water_temp'])
        sst = ds['water_temp'].sel(depth=0)

        cmap = kw.pop('cmap', plt.cm.rainbow)
        vmin = kw.pop('vmin', sst.min(dim=("lat", "lon")).item())
        vmax = kw.pop('vmax', sst.max(dim=("lat", "lon")).item())
        pooling = kw.pop('pooling', [])
        points = kw.pop('points', True)

        fig, ax = Hycom.plot_map(
            sst, ds['water_temp'].attrs['units'], cmap, vmin, vmax)
        ax.set_title(f'SST {time.strftime("%Y-%m-%d %HZ")}')

        if points:
            ax = Hycom._points(ax, lon, lat, pooling)

        if save:
            plt.savefig(f'SST_gom_{time.strftime("%Y%m%dT%HZ")}.png',
                        bbox_inches='tight', pad_inches=0.05)

    def plot_gom_ssh(time, lon, lat, save=True, **kw):

        ds = Hycom.subset_gom(time, lon, lat, kw.pop('grid', 2), ['ssh'])

        cmap = kw.pop('cmap', plt.cm.bwr)
        vmin = kw.pop('vmin', ds['ssh'].min(dim=("lat", "lon")).item())
        vmax = kw.pop('vmax', ds['ssh'].max(dim=("lat", "lon")).item())
        pooling = kw.pop('pooling', [])
        points = kw.pop('points', True)

        fig, ax = Hycom.plot_map(
            ds['ssh'], ds['ssh'].attrs['units'], cmap, vmin, vmax)
        ax.set_title(f'SSH {time.strftime("%Y-%m-%d %HZ")}')

        if points:
            ax = Hycom._points(ax, lon, lat, pooling)
        if save:
            plt.savefig(f'SSH_gom_{time.strftime("%Y%m%dT%HZ")}.png',
                        bbox_inches='tight', pad_inches=0.05)

    def plot_gom_vel(time, lon, lat, depth=0, quiver=True, save=True, **kw):

        ds = Hycom.subset_gom(time, lon, lat, kw.pop(
            'grid', 2), ['water_u', 'water_v'])
        if depth not in ds.depth:
            print('Depth not present in Hycom')
            return

        u = ds['water_u'].sel(depth=depth)
        v = ds['water_v'].sel(depth=depth)
        vel = np.sqrt(u**2 + v**2)

        cmap = kw.pop('cmap', plt.cm.rainbow)
        vmin = kw.pop('vmin', vel.min(dim=("lat", "lon")).item())
        vmax = kw.pop('vmax', vel.max(dim=("lat", "lon")).item())
        pooling = kw.pop('pooling', [])
        points = kw.pop('points', True)

        fig, ax = Hycom.plot_map(
            vel, ds['water_u'].attrs['units'], cmap, vmin, vmax)
        ax.set_title(
            f'VEL {time.strftime("%Y-%m-%d %HZ")} {depth}m')
        if points:
            ax = Hycom._points(ax, lon, lat, pooling)
        if quiver:
            ax = Hycom._quiver(ax, u, v)
        if save:
            plt.savefig(
                f'VEL_gom_{time.strftime("%Y%m%dT%HZ")}_{depth}m.png',
                bbox_inches='tight', pad_inches=0.05)

    def _points(ax, lon, lat, pooling):
        ax.plot(lon, lat, 'oy')
        for pos in pooling:
            ax.plot(pos[0], pos[1], marker='o', color='brown')
        return ax

    def _quiver(ax, u, v):
        ax.quiver(u.lon[::4], u.lat[::4], u[::4, ::4],
                  v[::4, ::4], pivot='middle', color='blue')
        return ax

    def plot_map(var, label, cmap, vmin, vmax):
        xx, yy = np.meshgrid(var.lon, var.lat)
        fig, ax = plt.subplots(
            figsize=(8.5, 11), dpi=70,
            subplot_kw={"projection": Hycom.ccrs.PlateCarree()})
        ax.set_extent([xx[0, 0], xx[0, -1], yy[0, 0], yy[-1, 0]])
        pcm = ax.pcolormesh(xx, yy, var, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.coastlines()
        ax.gridlines(draw_labels=True, edgecolor='gray', linewidth=.75)
        axpos = ax.get_position()
        px = axpos.x0
        py = axpos.y0 - 0.06
        cax_width = axpos.width
        cax_height = 0.02
        pos_cax = fig.add_axes([px, py, cax_width, cax_height])
        fig.colorbar(pcm, cax=pos_cax, orientation='horizontal', label=label)

        return fig, ax


class Qc(object):
    """Class to QC in xarray dataset for FUGRO Softwares

    .. hlist::
        :columns: 5

        * :func:`false_start_end`
        * :func:`get_expected_date`
        * :func:`get_QCed`

    """  # nopep8
    def _get_values_in(num):
        """get values inside METS flags"""
        v_in = []

        def factor(n):
            v = 0
            while n % 2 == 0:
                n = n / 2
                v += 1
            return 2**v
        while num > 0:
            i = factor(num)
            num = num - i
            v_in.append(i)
        return v_in

    def _remove_false_start_end_time(ds):
        da = ds['global_mask_0']
        ds['global_mask_0'].values = Qc._remove_values(da, 1)
        ds['global_mask_0'].values = Qc._remove_values(da, 2)
        Qc._get_false_start_end_time(ds)
        return ds

    def _remove_values(da, num):
        flags = [i - num if num in Qc._get_values_in(
            i) else i for i in da.values]
        da = np.asarray(flags, dtype=np.uint16)
        return da

    def _apply_values(da, idx, num):
        flags = [i if num in Qc._get_values_in(
            i) else i + num for i in da[idx]]
        da[idx] = np.asarray(flags, dtype=np.uint16)
        return da

    def _get_false_start_end_time(ds):
        fmt = "%Y-%m-%d %H:%M:%S"
        idx = sum([True for j in ds.global_mask_0.values
                   if 1 in Qc._get_values_in(j)])
        start = ds.isel(date=idx).date.dt.strftime(fmt).values
        fin_idx = sum([True for j in ds.global_mask_0.values
                       if 2 not in Qc._get_values_in(j)]) - 1
        end = ds.isel(date=fin_idx).date.dt.strftime(fmt).values
        attrs = ds.attrs
        attrs.update({f'False_Start_time > ': f'{start}'})
        attrs.update({f'False_End_time < ': f'{end}'})
        return ds

    def false_start_end(ds, start, end):
        Qc._remove_false_start_end_time(ds)
        ini_idx = ds['date'] < np.datetime64(start)
        ds['global_mask_0'] = Qc._apply_values(ds['global_mask_0'], ini_idx, 1)
        fin_idx = ds['date'] > np.datetime64(end)
        ds['global_mask_0'] = Qc._apply_values(ds['global_mask_0'], fin_idx, 2)
        attrs = ds.attrs
        attrs.update({f'False_Start_time > ': f'{start}'})
        attrs.update({f'False_End_time < ': f'{end}'})
        return ds

    def get_expected_date(ds):
        freq = ds.attrs['INTRAT'] / 10
        start = ds.attrs['False_Start_time > ']
        end = ds.attrs['False_End_time < ']
        expected_date = pd.date_range(start, end, freq=f'{freq}S')
        return expected_date

    def get_QCed(ds):
        var_list = [i for i in list(ds.keys()) if 'mask' not in i]
        flag_list = [i for i in list(ds.attrs.keys()) if 'local_mask' in i]
        DF = pd.DataFrame()
        for i in var_list:
            for j in flag_list:
                if i in ds.attrs[j]:
                    break
            VAR_Qced = ds[i].where(ds.global_mask_0 + ds[j] == 0)
            if 'level' in VAR_Qced.dims:
                df = VAR_Qced.to_dataframe().unstack('level')
                df.columns = [
                    f'{col[0]}{abs(int(col[1]))}m' for col in df.columns]
            else:
                df = VAR_Qced.to_dataframe()
            DF = pd.concat([DF, df], axis=1)
        DF = DF.reindex(Qc.get_expected_date(ds))
        return DF


if __name__ == "__main__":
    import doctest
    doctest.testmod()
