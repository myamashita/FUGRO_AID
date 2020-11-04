import requests
import numpy as np
import pandas as pd
import datetime as dtm

releases = requests.get(
    r'https://api.github.com/repos/myamashita/FUGRO_AID/releases/latest')
lastest = releases.json()['tag_name']
__version__ = '0.1.1'
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
        datetime.datetime(2020, 2, 12, 19, 33, 24, 735170)
        """
        return dtm.datetime.fromordinal(int(dtnum) - 366) +\
            dtm.timedelta(days=dtnum % 1)

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
        """  # nopep8
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
                x_axis_label=kw.pop('xlabel', 'Initial xlabel'),
                y_axis_label=kw.pop('ylabel', ''),
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

                    xlabel, height, width, x_axis_type

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
            x_axis_label=kw.pop('xlabel', 'Initial xlabel'),
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
                x_axis_label=kw.pop('xlabel', 'Initial xlabel'),
                y_axis_label=kw.pop('ylabel', 'Initial ylabel'),
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
                x_axis_label=kw.pop('xlabel', 'Initial xlabel'),
                y_axis_label=kw.pop('ylabel', 'Initial ylabel'),
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
        """
        def dateparser(x): return dtm.datetime.strptime(
            x, "%Y-%m-%dT%H:%M:%SZ")
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
