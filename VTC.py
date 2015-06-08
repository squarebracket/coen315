import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import csv
import pandas as pd


class VTC(object):
    
    bracket_size = 1000
    stop_error = 0.001
    max_iters = 10
    
    def __init__(self, csv_file, vin='Vin', vout='Vout'):
        self.x_values = []
        self.y_values = []
        self.traces = {}
        self.annotations = []
        
        self.csvfile = csv_file
        try:
            f = open(csv_file, 'r')
            self.spamreader = csv.DictReader(f, delimiter=' ')
        except TypeError:
            print("THERE WAS AN ERROR!")
            raise
        for row in self.spamreader:
            self.x_values.append(float(row[vin]))
            self.y_values.append(float(row[vout]))
        f.close()
        
        # getting a spline's derivative is much more accurate for finding
        # the VTC's characteristic parameters
        self.spl = InterpolatedUnivariateSpline(self.x_values, self.y_values)
        self.spline_1drv = self.spl.derivative()
        self.get_characteristic_parameters()
        
        self.traces['VTC'] = {
                'x': self.x_values, 
                'y': self.y_values, 
                'label': 'Voltage Transfer Characteristic',
                'class': 'main',
        }
    
    def find_vm(self, start=None, stop=None, bracket_size=1000,
                stop_error=0.001, max_iters=10):
        if not start:
            start = self.min_x
        if not stop:
            stop = self.max_x
        
        iter_count = 0
        smallest = numpy.nan_to_num(numpy.inf)
        smallest_at = 0
        while smallest > stop_error and iter_count < max_iters:
            q = numpy.linspace(start, stop, bracket_size)
            for (x, y) in zip(q, self.spl.__call__(q)):
                if abs(y-x) < smallest:
                    smallest = abs(y-x)
                    smallest_at = x
            length = float(stop - start)
            if not float(smallest_at - (length/4)) < start:
                start = float(smallest_at - (length/4))
            if not float(smallest_at + (length/4)) > stop:
                stop = float(smallest_at + (length/4))
            iter_count += 1
        return smallest_at
        
    
    def find_inflection_point(self, **iter_params):
        if not hasattr(self, 'spline_1drv'):
            self.spline_1drv = self.spl.derivative()
        
        p = self.defaults
        p.update(iter_params)
        start, stop = p['start'], p['stop']
                
        iter_count = 0
        biggest_y = 0
        biggest_x = 0
        while iter_count < p['max_iters']:
            q = numpy.linspace(start, stop, p['bracket_size'])
            for (dx, dy) in zip(q, self.spline_1drv.__call__(q)):
                if abs(dy) > biggest_y:
                    biggest_y = abs(dy)
                    biggest_x = dx
            length = float(stop - start)
            if not float(biggest_x - (length/4)) < start:
                start = float(biggest_x - (length/4))
            if not float(biggest_x + (length/4)) > stop:
                stop = float(biggest_x + (length/4))            
            iter_count += 1
        self.inflection_point = biggest_x
        return biggest_x
            
    def find_where_derivative_is(self, value, start=None,
                                 stop=None, bracket_size=1000,
                                 stop_error=0.001, max_iters=10):
        if not hasattr(self, 'spline_1drv'):
            self.spline_1drv = self.spl.derivative()
        if not start:
            start = self.min_x
        if not stop:
            stop = self.max_x

        iter_count = 0
        closest_y = numpy.inf
        closest_x = numpy.inf
        while abs(value - closest_y) > stop_error and iter_count < max_iters:
            q = numpy.linspace(start, stop, bracket_size)
            for (dx, dy) in zip(q, self.spline_1drv.__call__(q)):
                if abs(value - dy) < abs(value - closest_y):
                    closest_y = dy
                    closest_x = dx
            length = float(stop - start)
            if not float(closest_x - (length/4)) < start:
                start = float(closest_x - (length/4))
            if not float(closest_x + (length/4)) > stop:
                stop = float(closest_x + (length/4))
            iter_count += 1
        return closest_x
    
    def get_tangent_line_at(self, x):
        y = self.spl.__call__(x)
        m = self.spl.__call__(x, 1)
        b = y - (m * x)
        return y, m, b
    
    def make_tangent_line_at(self, x):
        # make tangent line occupy 1/6 of plot
        x_extend = self.range_x / 9
        y_extend = self.range_y / 9
        y, m, b = self.get_tangent_line_at(x)
        x1 = x - x_extend
        x2 = x + x_extend
        y1 = m*x1+b
        y2 = m*x2+b
        return x1, x2, y1, y2
        
    def get_characteristic_parameters(self):
        
        self.min_x = float(min(self.x_values))
        self.max_x = float(max(self.x_values))
        self.min_y = float(min(self.y_values))
        self.max_y = float(max(self.y_values))
        self.range_x = float(self.max_x - self.min_x)
        self.range_y = float(self.max_y - self.min_y)
        
        self.defaults = {
            'bracket_size': self.bracket_size,
            'stop_error': self.stop_error,
            'max_iters': self.max_iters,
            'start': self.min_x,
            'stop': self.max_x,
        }
        
        if not hasattr(self, 'inflection_point'):
            self.find_inflection_point()
        self.voh = self.max_y
        self.vih = self.find_where_derivative_is(-1, start=self.inflection_point)
        self.vm = self.find_vm()
        self.vil = self.find_where_derivative_is(-1, stop=self.inflection_point)
        self.vol = self.min_y
        self.nml = self.vil - self.vol
        self.nmh = self.voh - self.vih
        
#         print(self.vol, self.vil, self.vm, self.vih, self.voh, self.nml, self.nmh)
    
    def plot_ly(self, filename):
        offset = (self.max_y/20)
        line_style = Line(
            color='rgb(44, 160, 44)',
            opacity=0.25,
            dash='dot',
        )
        helper_line_style = {
            'line': line_style,
            'showlegend': False,
            'mode': 'lines',
            'connectgaps': True,
        }
        
        traces = []
        traces.append(Scatter(
                x=self.x_values,
                y=self.y_values,
        ))
        
        for (point, name) in {self.vil: 'V_IL', self.vih: 'V_IH'}.items():
            x1, x2, y1, y2 = self.make_tangent_line_at(point)
            traces.append(
                Scatter(
                    x=[x1, x2], 
                    y=[y1, y2], 
                    mode='lines',
                    name='tangent at dy/dx=-1',
                    showlegend=False,
                    connectgaps=True,
                    opacity=0.5,
                    line=Line(
                        color='#AAAAAA',
                    )
                )
            )
            traces.append(
                Scatter(
                    x=[point, point],
                    y=[self.min_y, self.spl(point)],
                    **helper_line_style
                )
            )
        
        for (point, name) in dict({self.vol: 'V_{OL}', self.voh: 'V_{OH}'}).items():
            traces.append(Scatter(
                    x=[self.min_x, self.max_x], 
                    y=[point, point],
                    name=name,
                    **helper_line_style
            ))
        
        traces.append(Scatter(
                x=[0, self.vm],
                y=[0, self.vm],
                mode='lines',
                name=['V_M'],
                line=line_style,
        ))
        
        data = Data(traces)
        
        annotations = []
        annotations.append(Annotation(x=self.max_x, xanchor='left', align='left', yanchor='top', y=self.vol, text='$V_{OL}$', showarrow=False))
        annotations.append(Annotation(x=self.vil, y=self.min_y, yanchor='top', text='$V_{IL}$', showarrow=False))
        annotations.append(Annotation(x=self.vm, y=self.vm, xanchor='left', align='left', text='$V_{M}$', showarrow=False))
        annotations.append(Annotation(x=self.vih, y=self.min_y, yanchor='top', text='$V_{IH}$', showarrow=False))
        annotations.append(Annotation(x=self.max_x, xanchor='left', align='left', y=self.voh, text='$V_{OH}$', showarrow=False))
        
        layout = Layout(
            title='Voltage Transfer Characteristic',
            xaxis=XAxis(title='$V_{in} \\left(\\text{V}\\right)$', showgrid=False),
            yaxis=YAxis(title='$V_{out} \\left(\\text{V}\\right)$', showgrid=False),
            annotations=Annotations(annotations),
            showlegend=False,
            autosize=False,
            width=500,
            height=500,
            margin=Margin(
                l=50,
                r=50,
                b=50,
                t=50,
            ),
        )
        fig = Figure(data=data, layout=layout)
        plot_url = py.plot(fig, filename=filename)
        
    def matplotlib(self, filename):
        import seaborn as sns
        sns.set_style('white')

        offset = (self.voh - self.vol)/50
        self.figure = plt.figure(facecolor='white', figsize=(3.5, 3.6))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.clip_on=False
        plt.tick_params(top='off', right='off')
        plt.locator_params('both', tight=True, nbins=4)
        # set text and line formatting stuff
        # plt.rc('text', usetex=True)
        plt.rc('lines', linewidth=1)
        plt.rc('font', size=12)
        main_plot = dict(linewidth=3, zorder=20)
        tangent_lines = dict(color='grey', linewidth=1)
        marker_lines = dict(color='grey', linewidth=1, linestyle='--')
        # 
        plt.xlabel(r"$V_{in}$")
        plt.ylabel(r"$V_{out}$")
        # plot the main VTC
        plt.plot(self.x_values, self.y_values, label='VTC', **main_plot)
        plt.plot([0, 1, self.vm], [0, 1, self.vm], **marker_lines)
        ax.annotate('$V_{M}$', xy=(self.vm, self.vm), xytext=(0, -8), textcoords='offset points',
                     horizontalalignment='left', verticalalignment='middle')


        for label, point in {r'$V_{OL}$': self.vol, r'$V_{IL}$': self.vil, r'$V_{IH}$': self.vih,
                      r'$V_{OH}$': self.voh}.items():
            x1, x2, y1, y2 = self.make_tangent_line_at(point)
            ax.plot([x1, x2], [y1, y2], **tangent_lines)
            ax.axvline(x=point, **marker_lines)
            # ax.annotate(label, xy=(point, 0), xytext=(0, -8), textcoords='offset points',
            #              horizontalalignment='center', verticalalignment='top')

        # ax.axvline(x=self.vil, **marker_lines)
        # ax.axvline(x=self.vih, **marker_lines)
        ax.annotate('$V_{OL}$', xy=(self.vol, 0), xytext=(0, -8), textcoords='offset points',
                     horizontalalignment='left', verticalalignment='top')
        ax.annotate('$V_{IL}$', xy=(self.vil, 0), xytext=(0, -8), textcoords='offset points',
                     horizontalalignment='center', verticalalignment='top')
        ax.annotate('$V_{IH}$', xy=(self.vih, 0), xytext=(0, -8), textcoords='offset points',
                     horizontalalignment='center', verticalalignment='top')
        ax.annotate('$V_{OH}$', xy=(self.voh, 0), xytext=(0, -8), textcoords='offset points',
                     horizontalalignment='center', verticalalignment='top')
        ax.annotate('', xy=(0, self.voh/1.5), xycoords='data',
                    xytext=(self.vil, self.voh/1.5), textcoords='data',
                    arrowprops=dict(arrowstyle="<->", ec="k",))
        ax.annotate('$NM_L$', xy=(self.vil/2, self.voh/1.5), xytext=(0, -5), textcoords='offset points',
                    horizontalalignment='center', verticalalignment='top')
        ax.annotate('', xy=(self.vih, self.voh/1.5), xycoords='data',
                    xytext=(self.voh, self.voh/1.5), textcoords='data',
                    arrowprops=dict(arrowstyle="<->", ec="k",))
        ax.annotate('$NM_H$', xy=((self.voh-self.vih)/2+self.vih, self.voh/1.5), xytext=(0, -5), textcoords='offset points',
                    horizontalalignment='center', verticalalignment='top')


        plt.tight_layout()

        ax.locator_params(axis='both', tight=True)
        ax.set_ylim(0-0.02)
        ax.set_xlim(0)
        self.figure.savefig(filename)
        
    
    def matplotly(self, filename):
        plot_url = py.plot_mpl(self.figure)
