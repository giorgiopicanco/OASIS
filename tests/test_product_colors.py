import unittest
import numpy as np
from pyOASIS.visualization.oasis_figures import product_style, _draw_station_mean, _plot_imports
from pyOASIS.gnss.station_statistics import station_statistics, roti_station_data
from pyOASIS.cli import build_parser

class ProductColorsTests(unittest.TestCase):
    def test_palette(self):
        self.assertEqual([product_style(s,'blue_red')[1] for s in 'GREC'], ['blue']*4)
        self.assertEqual([product_style(s,'constellation')[1] for s in 'GREC'], ['blue','red','lime','violet'])
        self.assertEqual(build_parser().parse_args(['--rinex','x','--figure-color-mode','blue_red']).product_figure_color_mode,'blue_red')
    def test_one_mean_even_with_legacy_smoothing(self):
        plt,_ = _plot_imports()
        t=np.arange(40)*30000
        data={f'G{i:02}':dict(utc_ms=t,value=np.ones(len(t))*i,elevation_deg=np.ones(len(t))*40) for i in range(1,5)}
        for mode,color in [('blue_red','red'),('constellation','black')]:
            fig,ax=plt.subplots()
            _draw_station_mean(ax,data,'VTEC',20,t,30000,smooth_minutes=15,color_mode=mode)
            self.assertEqual(len(ax.lines),1)
            self.assertEqual(ax.lines[0].get_color(),color)
            self.assertEqual(ax.lines[0].get_label(),'Station mean')
            np.testing.assert_allclose(ax.lines[0].get_ydata(),2.5)
            plt.close(fig)
    def test_mean_only_requested_products(self):
        from pyOASIS.gnss.station_statistics import PRODUCTS
        self.assertEqual(set(PRODUCTS), {'VTEC','DELTATEC','DTEC'})
        plt,_ = _plot_imports()
        t=np.arange(40)*30000
        data={f'G{i:02}':dict(utc_ms=t,window_start_utc_ms=t,value=np.ones(len(t))*i,elevation_deg=np.ones(len(t))*40) for i in range(1,5)}
        for mode in ('blue_red','constellation'):
            for product in ('LGF','ROTI','SIDX','STEC','VTEC','DELTATEC','DTEC'):
                fig,ax=plt.subplots()
                _draw_station_mean(ax,data,product,20,t,30000,color_mode=mode)
                self.assertEqual(len(ax.lines),int(product in PRODUCTS),product)
                plt.close(fig)
