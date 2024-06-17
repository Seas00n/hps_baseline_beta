import pyqtgraph as pg
import sys
import numpy as np
from plot_utils import ImpTune
pg.mkQApp()

dataset_name = 't'


imp_tuner = ImpTune(dataset=dataset_name)

if __name__ == "__main__":
    imp_tuner.show()
    pg.exec()
