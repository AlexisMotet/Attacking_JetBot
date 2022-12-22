import sys
import patch
import torch
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph as pg
import pickle
from threading import *
import load
import calibration
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
    return tuple(np.random.randint(256, size=3))

class Attribute():
    def __init__(self, name, read_only):
        self.name = name
        self.read_only = read_only
    
    def get_attribute(self, patch_desc) :
        return getattr(patch_desc, self.name)
    
    def get_tuple(self, patch_desc):
        return (self.name, self.get_attribute(patch_desc), self.read_only)
    
class WorkerLoading(QObject) :
    start = pyqtSignal()
    finished = pyqtSignal()
    def __init__(self, path, loaded, button, patch_desc, attr, loading_function):
        super().__init__()
        self.path = path
        self.loaded = loaded
        self.button = button
        self.patch_desc = patch_desc
        self.attr = attr
        self.loading_function = loading_function
        
    def run(self):
        self.start.emit()
        self.button.setDisabled(True)
        self.button.setText("Loading...")
        self.loaded[self.path] = self.loading_function(self.path)
        self.button.setDisabled(True)
        self.button.setText("Loaded")
        res = self.loaded[self.path]
        if (type(self.attr) is tuple) :
            for (r, attr) in zip(res, self.attr) :
                setattr(self.patch_desc, attr, r)
        else :
            setattr(self.patch_desc, self.attr, res)
        self.finished.emit()
        
class Model(Attribute) :
    def load(self, path, window, button, patch):
        try :
            window.create_thread(WorkerLoading, (path, window.loaded, button, patch, ("model"), 
                                             load.load_model))        
        except MainWindow.ThreadAlive :
            window.pop_up("A thread is already working")
        
class Dataset(Attribute) :
    def load(self, path, window, button, patch):
        if patch.model is None :
            window.pop_up("Please load model before loading dataset")
        else :
            try :
                window.create_thread(WorkerLoading, (path, window.loaded, button, 
                    patch, ("train_loader", "valid_loader", "test_loader"),
                    lambda path : load.load_dataset(path)))    
            except MainWindow.ThreadAlive :
                window.pop_up("A thread is already working")
            
class Calibration(Attribute) :
    def load(self, path, window, button, patch):
        try :
            window.create_thread(WorkerLoading, (path, window.loaded, button, 
                patch, ("cam_mtx", "dist_coef"),
                calibration.distorsion.load_coef))
        except MainWindow.ThreadAlive :
                window.pop_up("A thread is already working")
    
class WorkerTraining(QObject) :
    start = pyqtSignal()
    progress = pyqtSignal()
    finished = pyqtSignal()
    def __init__(self, patch_desc, training):
        super().__init__()
        self.training = training
        self.patch_desc = patch_desc
        
    def run(self):
        self.start.emit()
        self.i = 0
        if self.training :
            self.n = self.patch_desc.n_epochs * len(self.patch_desc.train_loader)
            self.patch_desc.train(self.callback)
        else :
            self.n = len(self.patch_desc.test_loader)
            print(self.patch_desc.test)
            self.patch_desc.test(self.callback)   
                    
    def callback(self):
        if self.i < self.n :
            self.progress.emit()
        else :
            self.finished.emit()
        self.i += 1
            
def center(w):
    frame_geo = w.frameGeometry()
    screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
    centerPoint = QApplication.desktop().screenGeometry(screen).center()
    frame_geo.moveCenter(centerPoint)
    w.move(frame_geo.topLeft())
            
class PatchWidget(QWidget):
    def __init__(self, window, attributes, new):
        super().__init__()
        self.window = window
        if (new) :
            self.patch_desc = patch.PatchDesc()
        else :
            filename, _ = QFileDialog.getOpenFileName(filter="*.patch")
            self.patch_desc = pickle.load(open(filename, "rb"))
        self.create(attributes, new)
        
    def draw_patch(self):
        self.ax.imshow(tensor_to_numpy_array(self.patch_desc.patch))
        self.canvas.draw()
    
    def check_loaded(self, text, button):
        if text in self.window.loaded :
            button.setText("Loaded")
            button.setDisabled(True)
        else :
            button.setText("Load")
            button.setDisabled(False)
            
    def create(self, attributes, new) :
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots()
        self.ax.set_axis_off()
        self.window.tabs.currentChanged.connect(lambda : self.canvas.setFixedSize(
            self.frameGeometry().width()//5, self.frameGeometry().width()//8))
        self.window.resized.connect(lambda : self.canvas.setFixedSize(
            self.frameGeometry().width()//8, self.frameGeometry().width()//8))
        self.draw_patch()
        vbox.addWidget(self.canvas)
        
        self.widgets = []
        
        for attr in attributes :
            name, val, read_only = attr.get_tuple(self.patch_desc)
            h = QHBoxLayout()
            h.addWidget(QLabel(name))
            if type(val) is str :
                widget = QLineEdit()
                widget.setText(val)
            elif type(val) is int :
                widget = QSpinBox()
                widget.setMaximum(1000)
                widget.setValue(val)
            elif type(val) is float :       
                widget = QDoubleSpinBox()
                widget.setSingleStep(0.1)
                widget.setMaximum(1)
                widget.setValue(val)
            else :
                assert False               
            if not new or read_only:
                    widget.setReadOnly(True)
            else :
                widget.textChanged.connect(lambda state, w=widget: setattr(self.patch_desc, 
                                                                    name, w.text()))
            self.widgets.append(widget)
            h.addWidget(widget)
            if new :
                ld = getattr(attr, "load", None)
                if callable(ld) :
                    b = QPushButton("Load")
                    b.setCursor(QCursor(Qt.PointingHandCursor))
                    if (type(attr) is Model) :
                        self.widget_path_model = widget
                    elif (type(attr) is Dataset) :
                        self.widget_path_dataset = widget
                    elif (type(attr) is Calibration) :
                        self.widget_path_calibration = widget
                    b.clicked.connect(lambda state, v=val, ld=ld, b=b, 
                                      p=self.patch_desc : ld(v, self.window, b, p))
                    widget.textChanged.connect(lambda state, w=widget, 
                                               b=b : self.check_loaded(w.text(), b))
                    self.check_loaded(widget.text(), b)
                    h.addWidget(b) 
            vbox.addLayout(h)
        
        if new :
            h = QHBoxLayout()
            for t in ["Train", "Test"] :
                stacked_widget = QStackedWidget()
                button = QPushButton(t)
                button.setCursor(QCursor(Qt.PointingHandCursor))
                
                stacked_widget.addWidget(button)
                stacked_widget.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, 
                                                         QSizePolicy.Maximum))
                h.addWidget(stacked_widget)
                progress_bar = QProgressBar()
                progress_bar.setMinimum(0)
                progress_bar.setMaximum(100)
                stacked_widget.addWidget(progress_bar)
                if t == "Train" :
                    button.clicked.connect(lambda state : self.start(training=True))
                    self.stacked_widget_train = stacked_widget
                    self.progress_train  = progress_bar
                else :
                    button.clicked.connect(lambda state : self.start(training=False))
                    self.stacked_widget_test = stacked_widget
                    self.progress_test  = progress_bar
            vbox.addLayout(h)
        
            save_button = QPushButton("Save")
            save_button.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, 
                                                QSizePolicy.Maximum))
            save_button.setCursor(QCursor(Qt.PointingHandCursor))
            save_button.clicked.connect(self.save)
            vbox.addWidget(save_button)
        
        vbox_plot = QVBoxLayout()
        for name in ["train", "valid", "test"]:
            plot = pg.PlotWidget(title="%s success rate" % name)
            plot.getPlotItem().setMenuEnabled(False)
            plot.getViewBox().setMouseEnabled(False, False)
            plot.getPlotItem().addLegend()
            plot.getPlotItem().getAxis("bottom").setLabel("batchs")
            plot.getPlotItem().getAxis("left").setLabel("% success")
            vbox_plot.addWidget(plot)
            setattr(self, "%s_plot" % name, plot)
            
        if not new :
            for e in self.patch_desc.train_success_rate.keys() :
                color = random_color()
                self.train_plot.plot(range(len(self.patch_desc.train_success_rate[e])), 
                                self.patch_desc.train_success_rate[e], 
                                pen=pg.mkPen(color = color, width = 2), 
                                name="epoch : %d" % e)
                
                self.valid_plot.plot(range(len(self.patch_desc.valid_success_rate[e])), 
                                self.patch_desc.valid_success_rate[e], 
                                pen=pg.mkPen(color = color, width = 2), 
                                name="epoch : %d" % e)
                
            self.test_plot.plot(range(len(self.patch_desc.test_success_rate)), 
                                self.patch_desc.test_success_rate, 
                                pen=pg.mkPen(color = (255, 0, 0), width = 2))

        hbox.addLayout(vbox, 2)
        hbox.addLayout(vbox_plot, 3)
        self.setLayout(hbox)
    
    def start(self, training):
        if self.widget_path_model.text() in self.window.loaded :
            self.patch_desc.model = self.window.loaded[self.widget_path_model.text()]
        if self.widget_path_dataset.text() in self.window.loaded :
            res = self.window.loaded[self.widget_path_dataset.text()]
            self.patch_desc.train_loader, self.patch_desc.valid_loader, self.patch_desc.test_loader = res
        if self.widget_path_calibration.text() in self.window.loaded :
            res = self.window.loaded[self.widget_path_calibration.text()]
            self.patch_desc.cam_mtx, self.patch_desc.dist_coef = res
        if training :
            for name in ["model", "train_loader", "valid_loader", "cam_mtx", "dist_coef"]:
                if getattr(self.patch_desc, name) is None :
                    self.window.pop_up("please load \"%s\" before training" % name)
                    return
        else :
            for name in ["model", "test_loader", "cam_mtx", "dist_coef"]:
                if getattr(self.patch_desc, name) is None :
                    self.window.pop_up("please load \"%s\" before testing" % name)
                    return
        try :
            if training :
                worker = self.window.create_thread(WorkerTraining, (self.patch_desc, True))
                self.freeze()
                self.train_data_item = {}
                self.valid_data_item = {}
                self.epochs = []
                self.stacked_widget_train.setCurrentWidget(self.progress_train)
                worker.progress.connect(self.update_training)
            else :
                worker = self.window.create_thread(WorkerTraining, (self.patch_desc, False))
                self.stacked_widget_test.setCurrentWidget(self.progress_test)
                worker.progress.connect(self.update_test)
        except MainWindow.ThreadAlive :
                window.pop_up("A thread is already working")
        
    def update_training(self) :
        self.draw_patch()
        self.progress_train.setValue(int(100 * self.window.worker.i/float(self.window.worker.n)))
        for e in self.patch_desc.train_success_rate.keys() :
            if e not in self.epochs :
                color = random_color()
                self.train_data_item[e] = self.train_plot.plot(pen=pg.mkPen(color=color, width = 2), 
                                                               name="epoch : %d" % e)
                
                self.valid_data_item[e] = self.valid_plot.plot(pen=pg.mkPen(color=color, width = 2), 
                                                               name="epoch : %d" % e)
                self.valid_plot.setXRange(0, len(self.patch_desc.train_loader))
            self.epochs.append(e)
            self.train_data_item[e].setData(range(len(self.patch_desc.train_success_rate[e])), 
                                                      self.patch_desc.train_success_rate[e])
            self.valid_data_item[e].setData(range(len(self.patch_desc.valid_success_rate[e])), 
                                                      self.patch_desc.valid_success_rate[e])
    
    def update_test(self) :
        self.progress_test.setValue(int(100 * self.window.worker.i/float(self.window.worker.n)))
        if (self.window.worker.i == 1) :
            self.test_data_item = self.test_plot.plot(pen=pg.mkPen(color=(225, 0, 0), width = 2))
        self.test_data_item.setData(range(len(self.patch_desc.test_success_rate)), 
                                              self.patch_desc.test_success_rate)
        
    def save(self):
        filename, _ = QFileDialog.getSaveFileName(filter="*.patch")
        pickle.dump(self.patch_desc, open(filename, "wb"))
        self.window.tabs.setTabText(self.window.tabs.currentIndex(), filename)
        
    def freeze(self):
        for widget in self.widgets :
            widget.setReadOnly(True)
    

class MainWindow(QMainWindow):
    class ThreadAlive(Exception):
        pass
    resized = pyqtSignal()
    def __init__(self):
        super().__init__()
        
        self.attributes = (Attribute("date", read_only=True),
                            Model("path_model", read_only=False),
                            Dataset("path_dataset", read_only=False),
                            Attribute("path_image_folder", read_only=False),
                            Calibration("path_calibration", read_only=False),
                            Attribute("id", read_only=False),
                            Attribute("image_dim", read_only=True),
                            Attribute("patch_relative_size", read_only=False),
                            Attribute("n_epochs", read_only=False),
                            Attribute("target_class", read_only=True),
                            Attribute("threshold", read_only=False),
                            Attribute("max_iterations", read_only=False))
        
        self.loaded = {}
        
        self.thread_alive = False
        
        self.setWindowTitle("My App")
        file_menu = self.menuBar().addMenu("File")
        
        file_act = QAction("New Patch", self)
        file_act.triggered.connect(self.add_tab_patch_widget)
        file_menu.addAction(file_act)
        
        file_act = QAction("Open Patch", self)
        file_act.triggered.connect(self.open_patch)
        file_menu.addAction(file_act)
        
        self.tabs = QTabWidget()
        self.tabs.tabBar().setCursor(Qt.PointingHandCursor)
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(lambda i : self.tabs.removeTab(i))
        
        self.add_tab_patch_widget()
        self.setCentralWidget(self.tabs)
        
        center(self)
    
    def add_tab_patch_widget(self):
        self.tabs.addTab(PatchWidget(self, self.attributes, new=True), "New Patch (*)")

    def resizeEvent(self, event):
        self.resized.emit()
        return super().resizeEvent(event)
    
    def open_patch(self):
        widget = PatchWidget(self, self.attributes, new=False)
        self.tabs.addTab(widget, "patch")
        
    def set_thread_alive(self, thread_alive):
        self.thread_alive = thread_alive
    
    def create_thread(self, worker, args):
        if (self.thread_alive): 
            raise MainWindow.ThreadAlive
        self.worker = worker(*args)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start.connect(lambda x=True : self.set_thread_alive(x))
        self.worker.finished.connect(lambda x=False : self.set_thread_alive(x))
        self.thread.start()
        return self.worker
    
    def pop_up(self, text):
        self.msg_box = QMessageBox()
        self.msg_box.setIcon(QMessageBox.Critical)
        self.msg_box.setText(text)
        self.msg_box.show()
        point = QCursor().pos()
        point.setX(point.x() - self.msg_box.width()//2)
        point.setY(point.y() - self.msg_box.height())
        self.msg_box.move(point)

if __name__ == "__main__" :
    sys.excepthook = except_hook

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()

    app.exec()
    
    