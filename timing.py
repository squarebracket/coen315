import GNUCapSimulationData as gc
import numpy as np

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('squarebracket', '6edn8gin4t')

for x in [1, 2, 3]:
    sim_data = None
    sim_data = gc.GNUCapSimulationData('circuit%s.csv' % x, index_col='Time',
                                       delim_whitespace=True)
    (vol, voh, total_current, total_power, avg_prop) = sim_data.get_properties(vout_node='sum', signal_node='a', psu_name='VDD')
    print('Parameters for Circuit %s' % x)
    print('VOL: %f' % vol)
    print('VOH: %f' % voh)
    print('total current: %s' % total_current)
    print('total power: %s' % total_power)
    print('average tprop: %s' % avg_prop)
    
    
    subplot_data = [
        ['V(a)', 'V(a_inv)'], 
        ['V(b)', 'V(b_inv)'], 
        ['V(cin)', 'V(cin_inv)'], 
        ['V(h)', 'V(i)'],
        ['V(j)', 'V(k)'],
        ['V(x)', 'V(y)'],
        ['V(sum)', 'V(sum_inv)'],
    ]
    f = sim_data.plot_timing_diagram(subplot_data, y1_label=r'$\mathrm{Voltage}$', 
                                     line_style={'alpha': 1},
                                     sec_line_style={'linestyle': '--', 'alpha': 1})
    f.savefig('vtd_%s.pdf' % x)
    
    subplot_data = ['I(VDD)', 'I(CH)', 'I(CI)', 'I(CJ)', 'I(CK)', 'I(CX)', 'I(CY)', 'I(CSum)', 'I(CSum_inv)']
    sim_data.df['I(PMOS)'] = sim_data.df['ID(M1P)'] + sim_data.df['ID(M5P)'] + \
    sim_data.df['ID(M9P)'] + sim_data.df['ID(M13P)']
    subplot_data = [
        ['I(VDD)',],
        # ['I(CH)', 'ID(M1P)', 'ID(M2N)',],
        # ['I(CI)', 'ID(M5P)', 'ID(M6N)',],
        # ['I(CJ)', 'ID(M9P)', 'ID(M10N)',],
        # ['I(CK)', 'ID(M13P)', 'ID(M14N)',],
        # ['I(CX)'],
        # ['I(CY)'],
        # ['I(CH)', 'ID(M1P)', 'ID(M2N)', 'ID(M3N)', 'ID(M4N)'],
        # ['I(CI)', 'ID(M5P)', 'ID(M6N)', 'ID(M7N)', 'ID(M8N)'],
        # ['I(CJ)', 'ID(M9P)', 'ID(M10N)', 'ID(M11N)', 'ID(M12N)'],
        # ['I(CK)', 'ID(M13P)', 'ID(M14N)', 'ID(M15N)', 'ID(M16N)'],
        # ['ID(M1P)', 'ID(M2N)', 'ID(M3N)', 'ID(M4N)'],
        # ['ID(M5P)', 'ID(M6N)', 'ID(M7N)', 'ID(M8N)'],
        # ['ID(M9P)', 'ID(M10N)', 'ID(M11N)', 'ID(M12N)'],
        # ['ID(M13P)', 'ID(M14N)', 'ID(M15N)', 'ID(M16N)'],
        ['ID(M1P)', 'ID(M5P)', 'ID(M9P)', 'ID(M13P)'],
        ['ID(M2N)', 'ID(M6N)', 'ID(M10N)', 'ID(M14N)'],
        ['I(CH)', 'I(CI)', 'I(CJ)', 'I(CK)'],
    ]
    f = sim_data.plot_timing_diagram(subplot_data, y1_label=r'$\mathrm{Current}$',
                                     line_style={'alpha': 0.5}, unit='A', yscale=1,
                                     sec_line_style={'alpha': 0.5}, y1_lim=(-0.0006, 0.0006), hspace=None)
    f.savefig('itd_%s.pdf' % x)

    subplot_data = [
        ('V(h)', ['I(CH)', 'ID(M1P)', 'ID(M2N)']),
        ('V(i)', ['I(CI)', 'ID(M5P)', 'ID(M6N)']),
        ('V(j)', ['I(CJ)', 'ID(M9P)', 'ID(M10N)']),
        ('V(k)', ['I(CK)', 'ID(M13P)', 'ID(M14N)']),
        ('V(x)', ['I(CX)', 'ID(M3N)', 'ID(M4N)', 'ID(M7N)', 'ID(M12N)']),
        ('V(y)', ['I(CY)', 'ID(M8N)', 'ID(M11N)', 'ID(M15N)', 'ID(M16N)']),
        ('V(sum)', ['I(CSum)', 'ID(M4N)', 'ID(M8N)'])
    ]
    if x == 3:
        subplot_data[6][1].append('ID(M17P)')
    time_slices=(
        (np.float64('5E-9'), np.float64('7E-9')),
        (np.float64('10E-9'), np.float64('12E-9')),
        (np.float64('15E-9'), np.float64('17E-9')),
        (np.float64('20E-9'), np.float64('22E-9')),
    )
    f = sim_data.plot3(subplot_data, right_line_style={'alpha': 0.75}, left_unit='V', right_unit='A', yscale=1.2,
                       left_line_style={'alpha': 0.2}, hspace=0,  time_slices=time_slices)
    f.savefig('itd3_%s.pdf' % x)
    f = sim_data.plot2(subplot_data, right_line_style={'alpha': 0.75}, left_unit='V', right_unit='A', yscale=1.2,
                       left_line_style={'alpha': 0.2}, hspace=0)
    f.savefig('itd2_%s.pdf' % x)
    
    # ax = sim_data.df[['ID(M1P)', 'ID(M5P)', 'ID(M9P)', 'ID(M13P)']].plot(kind='area', figsize=(8,1.2))
    # ax.figure.savefig('../report/test.pdf')

    subplot_data = (
        (['V(h)'], ['I(CH)', 'ID(M1P)', 'ID(M2N)', 'ID(M3N)']),
        (['V(x)', 'V(h)', 'V(i)'], ['ID(M3N)', 'ID(M4N)', 'ID(M7N)', 'ID(M12N)', 'I(CH)', 'I(CI)', 'I(CX)']),
    )

    f = sim_data.plot2(subplot_data, line_style={'alpha': 0.7}, left_unit='V', right_unit='A')
    f.savefig('current_%s.pdf' % x)
    

    f = sim_data.plot_timing_diagram(['V(x)', 'VTH(M3N)', 'VTH(M4N)', 'VTH(M7N)', 'VTH(M12N)'])
    f.savefig('a%s.pdf' % x)

    f = sim_data.plot_timing_diagram([['V(sum)', 'V(sum_inv)'], ['I(VDD)']],
                                     line_style={'alpha': 0.8}, 
                                     sec_line_style={'alpha': 0.8, 'linestyle': '--'},
                                     start_time=np.float64('4.E-9'), end_time=np.float64('9.E-9'),
                                     sharey=False)
    f.savefig('blip_%s.pdf' % x)


