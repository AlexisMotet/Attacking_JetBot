import sys
import utils.utils as u
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph as pg
import pickle
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import torchvision

QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

def random_color():
    return tuple(np.random.randint(200, 256, size=3))

def center(window):
    frame_geometry = window.frameGeometry()
    screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
    centerPoint = QApplication.desktop().screenGeometry(screen).center()
    frame_geometry.moveCenter(centerPoint)
    window.move(frame_geometry.topLeft())
            
class PatchWidget(QWidget):
    def __init__(self, window, patch_trainer, attributes):
        super().__init__()

        self.patch_trainer = patch_trainer

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        figure = Figure()
        figure.subplots_adjust(bottom=0, top=1, left=0, right=1)
        canvas = FigureCanvas(figure)
        ax = figure.subplots()
        ax.set_axis_off()
        ax.imshow(np.clip(u.tensor_to_array(patch_trainer.patch), 0, 1))
        window.tab_widget.currentChanged.connect(lambda : canvas.setFixedSize(
            self.frameGeometry().width()//8, self.frameGeometry().width()//8))
        window.resized.connect(lambda : canvas.setFixedSize(
            self.frameGeometry().width()//8, self.frameGeometry().width()//8))
        
        vbox.addWidget(canvas)

        button = QPushButton("Save Patch as Image")
        button.clicked.connect(self.save_patch_as_image)
        button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        vbox.addWidget(button)
        
        button_consts = QPushButton("See Constants")
        button_consts.clicked.connect(self.see_constants)
        button_consts.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        vbox.addWidget(button_consts)
        for attr in attributes :
            name, val = attr.get_tuple(patch_trainer)
            h = QHBoxLayout()
            h.addWidget(QLabel(name), 2)
            widget = QLineEdit()
            widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            widget.setText(str(val))             
            widget.setReadOnly(True)
            h.addWidget(widget, 3)
            vbox.addLayout(h)

        vbox_plot = QVBoxLayout()

        for i, e in enumerate(patch_trainer.target_proba_train.keys()) :
            if patch_trainer.n_epochs > 5 :
                if i%2 == 1 and i!=patch_trainer.n_epochs - 1:
                    continue
            color = random_color()
            plot = self.create_plot_item("Train epoch %d - "
                                            "Validation success rate : %.2f%%" % 
                                            (e, patch_trainer.success_rate_test[e]))
            plot.plot(range(len(patch_trainer.target_proba_train[e])), 
                        patch_trainer.target_proba_train[e], 
                        pen=pg.mkPen(color = color, width = 2), 
                        name="train epoch : %d" % e)
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

    def save_patch_as_image(self):
        filename = QFileDialog.getSaveFileName(filter="*.png")
        torchvision.utils.save_image(self.patch_trainer.patch, filename[0])
        
    def see_constants(self):
        message_box = QMessageBox(self)
        message_box.setWindowTitle("Constants")
        text = ""
        for key, value in self.patch_trainer.consts.items() :
            text += "%s : %s\n" % (str(key), str(value))
        message_box.setText(text)
        message_box.exec_()
        
class MainWindow(QMainWindow):
    class ThreadAlive(Exception):
        pass
    resized = pyqtSignal()
    def __init__(self):
        super().__init__()
        
        self.attributes = (u.Attribute("date"),
                            u.Attribute("target_class"),
                            u.Attribute("patch_relative_size"),
                            u.Attribute("n_epochs"))
        
        self.setWindowTitle("Patch Viewer")
        file_menu = self.menuBar().addMenu("File")
        
        file_action = QAction("Open Patch", self)
        file_action.triggered.connect(self.open_patch)
        file_menu.addAction(file_action)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(lambda i : self.tab_widget.removeTab(i))
        
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
    
