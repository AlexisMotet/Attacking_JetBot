import sys
import torch
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph as pg
import pickle
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

def tensor_to_numpy_array(tensor):
    tensor = torch.squeeze(tensor)
    array = tensor.cpu().numpy()
    return np.transpose(array, (1, 2, 0))

def random_color():
    return tuple(np.random.randint(200, 256, size=3))

class Attribute():
    def __init__(self, name):
        self.name = name
    
    def get_attribute(self, patch_trainer) :
        return getattr(patch_trainer, self.name)
    
    def get_tuple(self, patch_trainer):
        return (self.name, self.get_attribute(patch_trainer))

def center(w):
    frame_geo = w.frameGeometry()
    screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
    centerPoint = QApplication.desktop().screenGeometry(screen).center()
    frame_geo.moveCenter(centerPoint)
    w.move(frame_geo.topLeft())
            
class PatchWidget(QWidget):
    def __init__(self, window, patch_trainer, attributes):
        super().__init__()

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        
        figure = Figure()
        canvas = FigureCanvas(figure)
        ax = figure.subplots()
        ax.set_axis_off()
        ax.imshow(tensor_to_numpy_array(patch_trainer.patch))
        canvas.draw()
        window.tabs.currentChanged.connect(lambda : canvas.setFixedSize(
            self.frameGeometry().width()//5, self.frameGeometry().width()//5))
        window.resized.connect(lambda : canvas.setFixedSize(
            self.frameGeometry().width()//5, self.frameGeometry().width()//5))
        
        vbox.addWidget(canvas)
        
        widgets = []
        
        for attr in attributes :
            name, val = attr.get_tuple(patch_trainer)
            h = QHBoxLayout()
            h.addWidget(QLabel(name))
            widget = QLineEdit()
            widget.setText(str(val))             
            widget.setReadOnly(True)
            widgets.append(widget)
            h.addWidget(widget)
            vbox.addLayout(h)

        vbox_plot = QVBoxLayout()
        l = QLabel("Success rate on test dataset : %.2f%%" % patch_trainer.test_success_rate)
        l.setStyleSheet("font-weight: bold")
        vbox_plot.addWidget(l)

        for e in patch_trainer.target_proba_train.keys() :
            plot = pg.PlotWidget(title="epoch %d" % e)
            plot.getPlotItem().setMenuEnabled(False)
            plot.getViewBox().setMouseEnabled(False, False)
            plot.getPlotItem().addLegend()
            plot.getPlotItem().getAxis("bottom").setLabel("images")
            plot.getPlotItem().getAxis("left").setLabel("% target proba")
            plot.getPlotItem().getViewBox().setYRange(0, 1)
            
            color = random_color()
            plot.plot(range(len(patch_trainer.target_proba_train[e])), 
                            patch_trainer.target_proba_train[e], 
                            pen=pg.mkPen(color = color, width = 2), 
                            name="epoch : %d" % e)
            vbox_plot.addWidget(plot)

        plot = pg.PlotWidget(title="test")
        plot.getPlotItem().setMenuEnabled(False)
        plot.getViewBox().setMouseEnabled(False, False)
        plot.getPlotItem().addLegend()
        plot.getPlotItem().getAxis("bottom").setLabel("images")
        plot.getPlotItem().getAxis("left").setLabel("% target proba")
        plot.getPlotItem().getViewBox().setYRange(0, 1)
        plot.plot(range(len(patch_trainer.target_proba_test)), 
                            patch_trainer.target_proba_test, 
                            pen=pg.mkPen(color = (255, 0, 0), width = 2))
        vbox_plot.addWidget(plot)

        hbox.addLayout(vbox, 2)
        hbox.addLayout(vbox_plot, 3)
        self.setLayout(hbox)
    
        
class MainWindow(QMainWindow):
    class ThreadAlive(Exception):
        pass
    resized = pyqtSignal()
    def __init__(self):
        super().__init__()
        
        self.attributes = (Attribute("date"),
                            Attribute("path_model"),
                            Attribute("path_dataset"),
                            Attribute("path_calibration"),
                            Attribute("n_classes"),
                            Attribute("target_class"),
                            Attribute("patch_relative_size"),
                            Attribute("lambda_tv"),
                            Attribute("lambda_print"),
                            Attribute("distort"),
                            Attribute("n_epochs"),
                            Attribute("threshold"),
                            Attribute("max_iterations"))
        
        self.setWindowTitle("Patch Viewer")
        file_menu = self.menuBar().addMenu("File")
        
        file_act = QAction("Open Patch", self)
        file_act.triggered.connect(self.open_patch)
        file_menu.addAction(file_act)
        
        self.tabs = QTabWidget()
        self.tabs.tabBar().setCursor(Qt.PointingHandCursor)
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(lambda i : self.tabs.removeTab(i))
        
        self.setCentralWidget(self.tabs)
        
        center(self)

    def resizeEvent(self, event):
        self.resized.emit()
        return super().resizeEvent(event)
    
    def open_patch(self):
        filenames, _ = QFileDialog.getOpenFileNames(filter="*.patch")
        for filename in filenames :
            patch_trainer = pickle.load(open(filename, "rb"))
            widget = PatchWidget(self, patch_trainer, self.attributes)
            self.tabs.addTab(widget, filename)

if __name__ == "__main__" :
    sys.excepthook = except_hook

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()

    app.exec()
    
    