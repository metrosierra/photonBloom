#!/usr/bin/env python3

from subroutines.mathematics import percentsss, starsss

###%%%%%%%%%%%%%%%%%%%%%%

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import time
from PyQt5 import QtWidgets


class Plumeria():

    def __init__(self, title = 'Live Plot', xlabel = 'X axis', ylabel = 'Y axis', refresh_interval = 0.0001, plot_no = 1):

        # instantiate the window object
        self.app = QtWidgets.QApplication([])
        self.window = pg.GraphicsLayoutWidget(show = True, title = "Live Plotting Window")
        self.window.resize(900,500)
        # just antialiasing
        pg.setConfigOptions(antialias = True)

        # Creates graph object
        self.graph = self.window.addPlot(title = title)
        self.graph.addLegend()

        self.initial_xydata = [[0.], [0.]]
        self.refresh_interval = refresh_interval

        self.xlabel = xlabel 
        self.ylabel = ylabel
        self.styling = {'font-size':'20px'}
        self.set_xlabel(self.xlabel)
        self.set_ylabel(self.ylabel)

        self.point_count = 1
        self.plotting = False

        # creating maybe multiple line plot subclass objects for the self.graph object, store in list
        self.plot_no = plot_no 
        self.curves = []
        self.data_store = []
        ### storing lineplot instances into list, with indexed data store list
        for i in range(self.plot_no):
            ### setting pen as integer makes line colour cycle through 9 hues by default
            ### check pyqtgraph documentation on styling...it's quite messy
            self.curves.append(self.graph.plot(pen = i, name = 'Channel {}!!!'.format(i+1)))
            self.data_store.append(self.initial_xydata)
        print(self.curves)
        ### style points


        self.graph.showGrid(x = True, y = True)


    def __enter__(self):
        return self
 
    @percentsss 
    @starsss
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.app.closeAllWindows()
        print('Plotting Object Destroyed')
        print('Ciao bella ciao bella ciao ciao ciao')

    def set_xlabel(self, label):
        self.graph.setLabel('bottom', label, **self.styling)

    def set_ylabel(self, label):
        self.graph.setLabel('left', label, **self.styling)

    ### please input data as list of lists => [[xdata], [ydata]]
    ### this is a local data storage, the update function then
    ### updates the curve instances
    ### the idea is self.data_store is easier to access to check
    def set_data(self, data, index):
        self.data_store[index] = data
        self.curves[index].setData(data[0], data[1])
        return self

 
    def set_refresh_interval(self, interval):
        self.refresh_interval = interval
        return self

    def update(self):
        # set data simply changes the current dataframe to display

        if self.point_count == 0:
            self.graph.enableAutoRange('xy', False)  # stop auto-scaling after the first data set is plotted

        self.point_count += 1
        QtWidgets.QApplication.processEvents() # This command initiates a refresh
        time.sleep(self.refresh_interval)

    def get_window_state(self):
        self.is_closed = self.window.isHidden()
        return self.is_closed  


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        plume = Plumeria()
