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

def tensor_to_array(tensor):
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
        figure.subplots_adjust(bottom=0, top=1, left=0, right=1)
        canvas = FigureCanvas(figure)
        ax = figure.subplots()
        ax.set_axis_off()
        ax.imshow(tensor_to_array(patch_trainer.patch))
        
        window.tab_widget.currentChanged.connect(lambda : canvas.setFixedSize(
            self.frameGeometry().width()//8, self.frameGeometry().width()//8))
        window.resized.connect(lambda : canvas.setFixedSize(
            self.frameGeometry().width()//8, self.frameGeometry().width()//8))
        
        vbox.addWidget(canvas)
        
        widgets = []
        
        for attr in attributes :
            name, val = attr.get_tuple(patch_trainer)
            h = QHBoxLayout()
            h.addWidget(QLabel(name), 2)
            widget = QLineEdit()
            widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            widget.setText(str(val))             
            widget.setReadOnly(True)
            widgets.append(widget)
            h.addWidget(widget, 3)
            vbox.addLayout(h)

        vbox_plot = QVBoxLayout()
        if patch_trainer.validation :
            for e in patch_trainer.target_proba_test.keys() :
                color = random_color()
                plot = self.create_plot_item("test epoch %d success rate : %.2f%%" 
                                                % (e, patch_trainer.success_rate_test[e]))
                plot.plot(range(len(patch_trainer.target_proba_test[e])), 
                            patch_trainer.target_proba_test[e], 
                            pen=pg.mkPen(color = color, width = 2), 
                            name="test epoch : %d" % e)
                vbox_plot.addWidget(plot)
        if not patch_trainer.validation :
            plot = self.create_plot_item("test success rate : %.2f%%" 
                                         % patch_trainer.success_rate_test[e])
            plot.plot(range(len(patch_trainer.target_proba_test[-1])), 
                            patch_trainer.target_proba_test[-1], 
                            pen=pg.mkPen(color = color, width = 2), 
                            name="test")
            vbox_plot.addWidget(plot)


        hbox.addLayout(vbox, 1)
        hbox.addLayout(vbox_plot, 3)
        
        self.setLayout(hbox)
        
    def create_plot_item(self, title):
        plot = pg.PlotWidget(title=title)
        plot.getPlotItem().setMenuEnabled(False)
        plot.getViewBox().setMouseEnabled(False, False)
        plot.getPlotItem().addLegend()
        plot.getPlotItem().getAxis("bottom").setLabel("images")
        plot.getPlotItem().getAxis("left").setLabel("% target proba")
        plot.getPlotItem().getViewBox().setYRange(0, 1)
        return plot
        
class MainWindow(QMainWindow):
    class ThreadAlive(Exception):
        pass
    resized = pyqtSignal()
    def __init__(self):
        super().__init__()
        
        self.attributes = (Attribute("date"),
                            Attribute("path_model"),
                            Attribute("path_dataset"),
                            Attribute("n_classes"),
                            Attribute("target_class"),
                            Attribute("patch_relative_size"),
                            Attribute("distort"),
                            Attribute("n_epochs"),
                            Attribute("mode"),
                            Attribute("lambda_target"),
                            Attribute("lambda_flee"),
                            Attribute("lambda_tv"),
                            Attribute("lambda_print"),
                            Attribute("threshold"),
                            Attribute("max_iterations"))
        
        self.setWindowTitle("Patch Viewer")
        file_menu = self.menuBar().addMenu("File")
        
        file_action = QAction("Open Patch", self)
        file_action.triggered.connect(self.open_patch)
        file_menu.addAction(file_action)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(lambda i : 
            self.tab_widget.removeTab(i))
        
        self.setCentralWidget(self.tab_widget)
        
        center(self)

    def resizeEvent(self, event):
        self.resized.emit()
        return super().resizeEvent(event)
    
    def open_patch(self):
        filenames, _ = QFileDialog.getOpenFileNames(filter="*.patch")
        for filename in filenames :
            patch_trainer = pickle.load(open(filename, "rb"))
            widget = PatchWidget(self, patch_trainer, self.attributes)
            scroll_area = QScrollArea()
            scroll_area.setWidget(widget)
            scroll_area.setWidgetResizable(True)
            self.tab_widget.addTab(scroll_area, filename)

if __name__ == "__main__" :
    sys.excepthook = except_hook

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()

    app.exec()
    
    