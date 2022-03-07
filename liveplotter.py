#!/usr/bin/env python3


###%%%%%%%%%%%%%%%%%%%%%%

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import numpy as np
import time

class Plumeria():

    def __init__(self, title = 'Live Plot', refresh_interval = 0.0001, initial_xydata = [[0.], [0.]]):

        # instantiate the window object
        self.app = QtGui.QApplication([])
        self.window = pg.GraphicsLayoutWidget(show = True, title = "Live Plotting Window")

        self.window.resize(900,500)

        # just antialiasing
        pg.setConfigOptions(antialias = True)

        # Creates graph object
        self.graph = self.window.addPlot(title = title)
        self.curve = self.graph.plot(pen = 'y')

        self.point_count = 1
        self.plotting = False
        self.styles = {'color':'y', 'font-size':'20px'}
        self.graph.showGrid(x = True, y = True)

        self.set_data(initial_xydata)
        self.refresh_interval = refresh_interval

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        print('Plotting Object Destroyed')
        print('Ciao bella ciao bella ciao ciao ciao')


    def set_xlabel(self, label):
        self.graph.setLabel('bottom', label)

    def set_ylabel(self, label):
        self.graph.setLabel('left', label)

    def set_data(self, data):
        self.x, self.y = data
        return self

    def set_refresh_interval(self, interval):
        self.refresh_interval = interval
        return self

    def update(self):

        # set data simply changes the current dataframe to display
        self.curve.setData(self.x, self.y)
        if self.point_count == 0:
            self.graph.enableAutoRange('xy', False)  # stop auto-scaling after the first data set is plotted
        self.point_count += 1
        QtGui.QApplication.processEvents() # This command initiates a refresh
        time.sleep(self.refresh_interval)



## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        plume = Plumeria()
