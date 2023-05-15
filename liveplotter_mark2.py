#!/usr/bin/env python3

from subroutines.mathematics import percentsss, starsss

###%%%%%%%%%%%%%%%%%%%%%%
# from PyQt5 import QtWidgets

from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import time
import sys
import numpy as np


class WorkerBee(QtCore.QThread):
    signal = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, data_func, data_kill_func, isHidden_toggle, refresh_interval, identity):
        super().__init__()

        self.isHidden = isHidden_toggle
        self.data_func = data_func
        self.data_toggle_func = data_kill_func
        self.refresh_interval = refresh_interval
        self.identity = identity 

    def run(self):

        i = 0 
        while not self.isHidden():
            i += 1
            if i%1000 == 0:
                print('1000 cycles past')
                i = 0
                # print(type(self.data_func(self.identity)), 'heyhey')
            # print(self.data_func(self.identity))
            self.signal.emit(self.data_func(self.identity))
            # time.sleep(self.refresh_interval)
            time.sleep(self.refresh_interval)

        self.data_toggle_func(self.identity)
        self.quit()
        print('WorkerBee vi saluta')


class PetalWindow(QtWidgets.QWidget):

    ### this is link to the parent class decorated functions
    closed = QtCore.pyqtSignal()

    def __init__(self, data_func, data_kill_func, title = 'Live Plot', xlabel = 'X axis', ylabel = 'Y axis', refresh_interval = 0.1, plot_no = 1, identity = 0, save_on=False):
        
        super().__init__()

        self.window = pg.GraphicsLayoutWidget(show = True, title = "Live Plotting Window")
        self.window.resize(900,500)
        # just antialiasing
        pg.setConfigOptions(antialias = True)
        # Creates graph object
        self.graph = self.window.addPlot(title = title)
        self.graph.addLegend()

        self.identity = identity
        self.save_on = save_on
        self.refresh_interval = refresh_interval
        ##################### style points #####################
        self.graph.showGrid(x = True, y = True)
        self.styling = {'font-size':'20px'}
        ########################################################
        self.initial_xydata = [[[0.], [0.]]]
        self.xlabel = xlabel 
        self.ylabel = ylabel
        self.set_xlabel(self.xlabel)
        self.set_ylabel(self.ylabel)

        # creating maybe multiple line plot subclass objects for the self.graph object, store in list
        self.plot_no = plot_no 
        self.plots = []
        self.data_store = []
        ### storing lineplot instances into list, with indexed data store list
        for i in range(self.plot_no):
            ### setting pen as integer makes line colour cycle through 9 hues by default
            ### check pyqtgraph documentation on styling...it's quite messy
            self.plots.append(self.graph.plot(pen = i, name = 'Channel {}!!!'.format(i+1)))
            self.data_store.append(self.initial_xydata)
        
        self.worker = WorkerBee(data_func, data_kill_func, self.window.isHidden, self.refresh_interval, self.identity, self.save_on)
        self.make_connection(self.worker)
        self.worker.start()
        # self.show()
        print('New Plot Window Identity is', self.identity)

    def make_connection(self, data_object):
        data_object.signal.connect(self.update)
        return self
    @QtCore.pyqtSlot(np.ndarray)
    def update(self, data):
        self.set_data(data)
        return self
    def set_xlabel(self, label):
        self.graph.setLabel('bottom', label, **self.styling)
        return self
    def set_ylabel(self, label):
        self.graph.setLabel('left', label, **self.styling)
        return self
    ### please input data as list of lists => [[xdata], [ydata]]
    ### this is a local data storage, the update function then
    ### updates the curve instances
    ### the idea is self.data_store is easier to access to check
    def set_data(self, data):
        
        if self.plot_no != 5:
            for i in range(self.plot_no):
                self.data_store[i] = data[i]
                self.plots[i].setData(np.arange(len(data[i])), data[i])
        
        else:
            for i in range(5):
                self.data_store[-i] = data[-i]
                self.plots[i].setData(np.arange(len(data[-i])), data[-i])

        return self



class PetalWindowMulti(QtWidgets.QWidget):

    ### this is link to the parent class decorated functions
    closed = QtCore.pyqtSignal()

    def __init__(self, data_func, data_kill_func, title = 'Live Plot', xlabel = 'X axis', ylabel = 'Y axis', refresh_interval = 0.1, plot_no = 1, identity = 0):
        
        super().__init__()
        self.plot_no = plot_no 

        self.window = pg.GraphicsLayoutWidget(show = True, title = "Live Plotting Window")
        self.window.resize(900,500)
        # just antialiasing
        pg.setConfigOptions(antialias = True)
        # Creates graph object
        self.graphs = []
        self.plots = []

        for i in range(self.plot_no):  
            self.graphs.append(self.window.addPlot(title = title))
            self.graphs[i].addLegend()
            self.graphs[i].showGrid(x = True, y = True)
            self.plots.append(self.graphs[i].plot(pen = i, name = 'Channel {}!!!'.format(i+1)))


        self.identity = identity
        self.refresh_interval = refresh_interval
        ##################### style points #####################
        # self.graph.showGrid(x = True, y = True)
        self.styling = {'font-size':'20px'}
        ########################################################
        self.initial_xydata = [[[0.], [0.]]]
        self.xlabel = xlabel 
        self.ylabel = ylabel
        self.set_xlabel(self.xlabel)
        self.set_ylabel(self.ylabel)

        # creating maybe multiple line plot subclass objects for the self.graph object, store in list
        self.data_store = [[]]
        ### storing lineplot instances into list, with indexed data store list
        for i in range(self.plot_no):
            ### setting pen as integer makes line colour cycle through 9 hues by default
            ### check pyqtgraph documentation on styling...it's quite messy
            self.plots.append(self.graphs[i].plot(pen = i, name = 'Channel {}!!!'.format(i+1)))
            self.data_store.append(self.initial_xydata)
        
        self.worker = WorkerBee(data_func, data_kill_func, self.window.isHidden, self.refresh_interval, self.identity)
        self.make_connection(self.worker)
        self.worker.start()
        # self.show()
        print('New Plot Window Identity is', self.identity)

    def make_connection(self, data_object):
        data_object.signal.connect(self.update)
        return self
    @QtCore.pyqtSlot(np.ndarray)
    def update(self, data):
        self.set_data(data)
        return self
    def set_xlabel(self, label):
        for i in range(self.plot_no):
            self.graphs[i].setLabel('bottom', label, **self.styling)
        return self
    def set_ylabel(self, label):
        for i in range(self.plot_no):
            self.graphs[i].setLabel('left', label, **self.styling)
        return self
    ### please input data as list of lists => [[xdata], [ydata]]
    ### this is a local data storage, the update function then
    ### updates the curve instances
    ### the idea is self.data_store is easier to access to check
    def set_data(self, data):
        
        for i in range(self.plot_no):
            self.data_store[i] = data[i]
            self.plots[i].setData(np.arange(len(data[i])), data[i])
        
        return self

### Rose is the plot GUI instance, 
### We have one rose and multiple petals (plot windows)
class RoseApp(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.w = None  # No external window yet.
        
        self.windows = []
        self.window_count = 0

        # self.new_window()
        # self.show()


    def new_window(self, data_func, data_kill_func, refresh_interval = 0.0001, title = 'Live Plot', xlabel = 'X axis', ylabel = 'Y axis', plot_no = 1, identity = 0, save_on = True):
        self.windows.append(PetalWindow(data_func, data_kill_func, title, xlabel, ylabel, refresh_interval, plot_no, identity,save_on))
        # self.windows[self.window_count].show()
        self.window_count += 1

    def new_multiwindow(self, data_func, data_kill_func, refresh_interval = 0.0001, title = 'Live Plot', xlabel = 'X axis', ylabel = 'Y axis', plot_no = 1, identity = 0):
        self.windows.append(PetalWindowMulti(data_func, data_kill_func, title, xlabel, ylabel, refresh_interval, plot_no, identity))
        # self.windows[self.window_count].show()
        self.window_count += 1

## Start Qt event loop unless running in interactive mode
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        
        app = QtGui.QApplication(sys.argv)
        rose = RoseApp()
        app.exec_()