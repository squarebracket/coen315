import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import EngFormatter
from matplotlib.cbook import Grouper

plt.rc('legend', labelspacing=0.2)
plt.rc('text', usetex=True)
plt.rc('text.latex', unicode=True)

color_palette_2 = sns.color_palette('Set2', 10)
# color_palette_2.pop(2)


def reformat(s):
    m = re.match(r'([A-Z]*)\(([A-Z0-9a-z_]*?)(_inv)?\)', s)
    if m.group(3):
        name = '$%s_{\\overline{%s}}$' % (m.group(1).upper(), m.group(2).title())
    else:
        name = '$%s_{%s}$' % (m.group(1).upper(), m.group(2).title())
    return name, m.group(1)


def make_iter(obj):
    if not hasattr(obj, '__iter__'):
        return [obj]
    else:
        if isinstance(obj, str):
            return [obj]
        else:
            return obj


class GNUCapSimulationData(object):
    
    def __init__(self, csv_file, index_col=None, **parse_args):
        self.df = pd.read_csv(csv_file, **parse_args)
        if index_col:
            self.df.index = self.df.pop(index_col)
        
    def get_properties(self, vout_node='out', psu_name='VDD', 
                       signal_node='in'):
        vout_col = 'V(%s)' % vout_node
        psu_current_col = 'I(%s)' % psu_name
        psu_power_col = 'P(%s)' % psu_name
        
        voh = self.df[vout_col].max()
        vol = self.df[vout_col].min()
        total_current = self.df[psu_current_col].sum()
        total_power = self.df[psu_power_col].sum()
        
        prop_times = self.get_propagations(vout_node=vout_node, 
                                           signal_node=signal_node)
        avg_prop = sum(prop_times) / len(prop_times)
        
        return (vol, voh, total_current, total_power, avg_prop)
    
    def get_propagations(self, vout_node='out', signal_node='in', return_full=False):
        vin_col = 'V(%s)' % signal_node
        vout_col = 'V(%s)' % vout_node
        
        # No need for all the columns; we'll work on a copy with
        # just what we need
        prop_df = self.df[[vin_col, vout_col]]
        
        in_half = (prop_df[vin_col].max() - prop_df[vin_col].min()) / 2
        out_half = (prop_df[vout_col].max() - prop_df[vout_col].min()) / 2
        
        transition_times = prop_df[prop_df[vin_col] == in_half].index
        # Add last time index to bracket last transition time
        transition_times = transition_times.append(prop_df.index[-1:])
        
        prop_times = []
        
        for i in range(0, len(transition_times)-1):
            start_time = transition_times[i]
            end_time = transition_times[i+1]

            # Slice the DataFrame into just this time slice
            time_slice = prop_df.loc[start_time:end_time, :]
            if time_slice.iat[0, 1] < out_half:
                # expect the output to go high
                point = time_slice[time_slice[vout_col] > out_half].iloc[:1]
            elif time_slice.iat[0, 1] > out_half:
                # expect the output to go low
                point = time_slice[time_slice[vout_col] < out_half].iloc[:1]
            else:
                raise ValueError('Unable to determine direction of output')
            
            if not point.empty:
                cross_time = point.index[0]
                cross_voltage = point.values[0][1]
                if return_full:
                    prop_times.append((start_time, cross_time))
                else:
                    prop_time = cross_time - start_time
                    prop_times.append(prop_time)
            
        return prop_times
            
    def plot_timing_diagram(self, subplot_data, start_time=None, end_time=None, 
                            y1_label=None, sharey=True, line_style={}, sec_line_style={},
                            y1_lim=None, hspace=0, unit=None, yscale=1.2):

        sns.set_style('whitegrid')
        fig, axes = plt.subplots(len(subplot_data), sharex=True, sharey=sharey)
        fig.set_size_inches(7, yscale*len(subplot_data))
        fig.subplots_adjust(hspace=hspace)
        
        df = self.df.loc[start_time:end_time, :]

        for (axis, columns) in zip(axes, subplot_data):
            if type(columns) == list:
                for column in columns:
                    if column == columns[0]:
                        self._plot_timing_diagram(df, column, axis, **line_style)
                    else:
                        self._plot_timing_diagram(df, column, axis, **sec_line_style)
            else:
                self._plot_timing_diagram(df, columns, axis, **line_style)

            self._tweak_formats(axis, unit)

        if sharey:
            axes[len(subplot_data)/2].set_ylabel(y1_label)
        if y1_lim:
            [axis.set_ylim(y1_lim) for axis in axes]
        return fig
    
    def _plot_timing_diagram(self, df, column, axis, **line_style):
        label, group = reformat(column)
        line, = axis.plot(df.index, df[column], label=label, **line_style)
        return line, label

    def plot2(self, subplot_data, start_time=None, end_time=None, sharey=True,
              line_style={}, left_unit=None, right_unit=None, yscale=2,
              left_line_style={}, right_line_style={}, hspace=None):
        left_line_style.update(line_style)
        right_line_style.update(line_style)

        sns.set_style('white')
        fig, axes = plt.subplots(nrows=len(subplot_data), ncols=1, sharex=True, sharey=sharey)
        fig.set_size_inches(7, yscale*len(subplot_data))

        df = self.df.loc[start_time:end_time, :]

        grp = Grouper()
        # help(grp)

        prev_y = None

        for (left_axis, col_spec) in zip(axes, subplot_data):
            lines = []
            labels = []

            print(left_axis._sharey)
            print(left_axis.get_shared_y_axes().get_siblings(left_axis))

            right_axis = left_axis.twinx()
            right_axis._adjustable = 'datalim'

            if prev_y is not None:
                right_axis._sharey = prev_y
                right_axis._shared_y_axes.join(prev_y, right_axis)
                right_axis.cla()
            print(right_axis.get_shared_y_axes().get_siblings(right_axis))
            prev_y = right_axis
            right_axis.autoscale_view(scaley=True)
            right_axis.set_color_cycle(color_palette_2)
            for column in make_iter(col_spec[1]):
                line, label = self._plot_timing_diagram(df, column, right_axis, **right_line_style)
                lines.append(line)
                labels.append(label)
            for column in make_iter(col_spec[0]):
                line, label = self._plot_timing_diagram(df, column, left_axis, **left_line_style)
                lines.append(line)
                labels.append(label)

            self._tweak_formats2(left_axis, left_unit)

            self._tweak_formats2(right_axis, right_unit)
            right_axis.legend(lines, labels, loc='center left', bbox_to_anchor=(1.1, 0.5))
            if left_axis.is_last_row():
                x_formatter = EngFormatter(unit='s', places=0)
                left_axis.xaxis.set_major_formatter(x_formatter)
                left_axis.set_xlabel(r'$\mathrm{Time}$')
            print ('a\n')
        print(grp.__dict__)
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        fig.subplots_adjust(hspace=hspace)
        return fig

    def plot3(self, subplot_data, time_slices=None, sharey=True, sharex=True,
              line_style={}, left_unit=None, right_unit=None, yscale=2,
              left_line_style={}, right_line_style={}, hspace=None):
        left_line_style.update(line_style)
        right_line_style.update(line_style)

        sns.set_style('white')

        if not time_slices:
            time_slices = ((None, None), )
        else:
            sharex = False

        print(time_slices)

        fig, axes = plt.subplots(nrows=len(subplot_data), ncols=len(time_slices), sharex=sharex, sharey=sharey)
        fig.set_size_inches(7, yscale*len(subplot_data))

        print(axes)

        grp = Grouper()

        prev_y = None

        for (dur_spec, col_spec) in zip(axes, subplot_data):
            for (left_axis, time_slice) in zip(dur_spec, time_slices):
                right_axis = left_axis.twinx()
                right_axis._adjustable = 'datalim'
                if prev_y is not None:
                    right_axis._sharey = prev_y
                    right_axis._shared_y_axes.join(prev_y, right_axis)
                    # right_axis.cla()
                prev_y = right_axis
                # right_axis.autoscale_view(scaley=True)
                left_axis.right = right_axis

        for (dur_spec, col_spec) in zip(axes, subplot_data):
            for (left_axis, time_slice) in zip(dur_spec, time_slices):
                print(time_slice)
                df = self.df.loc[time_slice[0]:time_slice[1], :]
                lines = []
                labels = []

                print(left_axis._sharey)
                print(left_axis.get_shared_y_axes().get_siblings(left_axis))

                right_axis = left_axis.right
                # right_axis = left_axis.twinx()
                # right_axis._adjustable = 'datalim'
                #
                # if prev_y is not None:
                #     right_axis._sharey = prev_y
                #     right_axis._shared_y_axes.join(prev_y, right_axis)
                # print(right_axis.get_shared_y_axes().get_siblings(right_axis))
                # prev_y = right_axis
                # right_axis.autoscale_view(scaley=True)
                right_axis.set_color_cycle(color_palette_2)
                for column in make_iter(col_spec[1]):
                    line, label = self._plot_timing_diagram(df, column, right_axis, **right_line_style)
                    lines.append(line)
                    labels.append(label)
                for column in make_iter(col_spec[0]):
                    line, label = self._plot_timing_diagram(df, column, left_axis, **left_line_style)
                    lines.append(line)
                    labels.append(label)

                self._tweak_formats2(left_axis, left_unit)


                self._tweak_formats3(right_axis, right_unit)
                if left_axis.is_last_col():
                    right_axis.legend(lines, labels, loc='center left', bbox_to_anchor=(1.7, 0.5))
                if left_axis.is_last_row():
                    x_formatter = EngFormatter(unit='s', places=0)
                    left_axis.xaxis.set_major_formatter(x_formatter)
                    left_axis.set_xlabel(r'$\mathrm{Time}$')
                left_axis.locator_params(axis='x', nbins=3)
                # right_axis.locator_params(axis='x', nbins=1)
            print('\n')
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        fig.subplots_adjust(hspace=hspace, wspace=0.28)
        return fig

    def _tweak_formats(self, axis, unit):
            # formatting adjustments to ensure proper plot
            box = axis.get_position()
            axis.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            if unit:
                y_formatter = EngFormatter(unit=unit, places=1)
                axis.yaxis.set_major_formatter(y_formatter)
            axis.set_ymargin(0.05)
            axis.label_outer()
            axis.locator_params(axis='both', tight=True)
            axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axis.spines['bottom'].set_visible(True)
            if axis.is_last_row():
                axis.set_xlabel(r'$\mathrm{Time}$')
                x_formatter = EngFormatter(unit='s', places=0)
                axis.xaxis.set_major_formatter(x_formatter)

    def _tweak_formats2(self, axis, unit):
            # formatting adjustments to ensure proper plot
            box = axis.get_position()
            axis.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            if unit and axis.is_last_col():
                y_formatter = EngFormatter(unit=unit, places=1)
                axis.yaxis.set_major_formatter(y_formatter)
            axis.set_ymargin(0.05)
            axis.label_outer()
            axis.locator_params(axis='both', tight=True)
            axis.grid(axis='x')

            # axis.spines['bottom'].set_visible(False)
            # axis.spines['top'].set_visible(False)
            axis.spines['bottom'].update(dict(color='#cccccc', linewidth=0.5))
            axis.spines['top'].update(dict(color='#cccccc', linewidth=0.5))

            axis.spines['left'].set_visible(False)
            axis.spines['right'].set_visible(False)

            # axis.spines['left'].update(dict(color='#aaaaaa', linewidth=0.5))
            # axis.spines['right'].update(dict(color='#aaaaaa', linewidth=0.5))

    def _tweak_formats3(self, axis, unit):
            # formatting adjustments to ensure proper plot
            box = axis.get_position()
            axis.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            if unit and axis.is_last_col():
                y_formatter = EngFormatter(unit=unit, places=1)
                axis.yaxis.set_major_formatter(y_formatter)
            axis.set_ymargin(0.05)
            if not axis.is_last_col():
                axis.tick_params(labelright=False)
            axis.tick_params(labelbottom=False)
            axis.locator_params(axis='both', tight=True)
            # axis.locator_params(axis='y', nbins=7)
            axis.grid(axis='x')

            # axis.spines['bottom'].set_visible(False)
            # axis.spines['top'].set_visible(False)
            axis.spines['bottom'].update(dict(color='#cccccc', linewidth=0.5))
            axis.spines['top'].update(dict(color='#cccccc', linewidth=0.5))

            axis.spines['left'].set_visible(False)
            axis.spines['right'].set_visible(False)

            # axis.spines['left'].update(dict(color='#aaaaaa', linewidth=0.5))
            # axis.spines['right'].update(dict(color='#aaaaaa', linewidth=0.5))