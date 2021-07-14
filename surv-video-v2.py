import json , cv2, imutils
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
import pyqtgraph as pg
from random import randint
# from object_detection import get_detections




################################
### Some global variables and
### function used 
################################
## threshold for social distancing
social_distance_thresh = 5


## a matrix with distance for each
## pair of centeriods(person)
def distance_matrix(arr):
    z = np.array([complex(c[0], c[1]) for c in arr])
    m, n = np.meshgrid(z, z)
    # get the distance via the norm
    out = abs(m-n)
    return out
################################


################################# 
####nanodet model load (for person detection)
#################################
from nanodet_model_infer import Predictor
from nanodet.util import cfg, load_config, Logger
local_rank = 0  
logger = Logger(local_rank, use_tensorboard=False)
load_config(cfg, "/home/ashish/Documents/DeepLearning/nanodet/config/nanodet-m.yml")  
predictor = Predictor(cfg, "/home/ashish/Documents/DeepLearning/nanodet/pytorch-model/nanodet_m.ckpt", logger, device='cpu')
#################################


################################# 
####ssd face detection and mask classification (https://github.com/AIZOOTech/FaceMaskDetection)
#################################
from pytorch_infer import inference
id2class = {0: 'Mask', 1: 'NoMask'}

def visualize(image, output_info):
    mask_count = 0
    no_mask_count = 0
    for data in output_info:
        print("data: ", data)
        class_id, conf, xmin, ymin, xmax, ymax = data
        if class_id == 0:
            mask_count +=1
            color = (0, 255, 0)
        else:
            no_mask_count +=1
            color = (255, 0, 0)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
    return image, mask_count, no_mask_count
#################################



### video process worker
class worker(QObject):
    finished = pyqtSignal()
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_plot_signal = pyqtSignal(int, int, int, list)

    def __init__(self):
        super(QObject, self).__init__()
        self._run_flag = True
        #########
        ##nanodet
        self.meta = None
        self.res = None
        #########
        ###aizootecc
        self.output_info = None
        #########

        
    def run(self, url):
        print("url", url)
        self.cap = cv2.VideoCapture(url)
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        ###extract frame at fps
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.send_frames)
        print("running at: ", fps)
        self.timer.start(1000//fps)

        ### get detection at custom fps
        self.timer2 = QtCore.QTimer(self)
        self.timer2.timeout.connect(self.get_det)
        print("running at: ", fps)
        self.timer2.start(150)

    def send_frames(self):
        if self._run_flag:
            ret, cv_img = self.cap.read()
            self.img_np = cv_img
            if ret:
                try:
                    cv_img, person_count, person_centriods = predictor.visualize(self.res[0], self.meta, cfg.class_names, 0.5)
                    cv_img, mask_count, no_mask_count = visualize(cv_img, self.output_info)
                except Exception as e:
                    import sys
                    print("error e: ", e)
                    # sys.exit()
                self.change_pixmap_signal.emit(cv_img)
                self.update_plot_signal.emit(person_count, mask_count, no_mask_count, person_centriods)
            else:
                self._run_flag = False
                self.cap.release()
                self.finished.emit()
        # if self._run_flag==False or ret == False:

    ## get detection at custom frame rates
    def get_det(self):
        #### get person
        self.meta, self.res = predictor.inference(self.img_np)

        #### get face and mask
        img = cv2.cvtColor(self.img_np, cv2.COLOR_BGR2RGB)
        self.output_info = inference(img,
            conf_thresh=0.5,
            iou_thresh=0.5,
            target_shape=(360, 360),
            draw_result=False,
            show_result=False)
        print("output_info: ", self.output_info)
        # self.img_np = img[:, :, ::-1]

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(622, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setObjectName("tabWidget")
        self.stream = QtWidgets.QWidget()
        self.stream.setObjectName("stream")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.stream)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.screen = QtWidgets.QLabel(self.stream)
        self.screen.setFrameShape(QtWidgets.QFrame.Box)
        self.screen.setAlignment(QtCore.Qt.AlignCenter)
        self.screen.setObjectName("screen")
        self.verticalLayout.addWidget(self.screen)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")

        self.url_btn = QtWidgets.QPushButton(self.stream)
        self.url_btn.setObjectName("url_btn")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.url_btn)
        self.url_btn.clicked.connect(self.start_screen)

        self.url = QtWidgets.QLineEdit(self.stream)
        self.url.setObjectName("url")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.url)
        self.verticalLayout.addLayout(self.formLayout)

        self.op = QtWidgets.QTextEdit(self.stream)
        self.op.setObjectName("op")
        self.verticalLayout.addWidget(self.op)
        #### text shown in op field of tab1
        self.op_text = f''

        self.verticalLayout.setStretch(0, 6)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 3)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.tabWidget.addTab(self.stream, "")
        self.analysis = QtWidgets.QWidget()
        self.analysis.setObjectName("analysis")
        self.gridLayout = QtWidgets.QGridLayout(self.analysis)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.analysis)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        # self.graphicsView = QtWidgets.QGraphicsView(self.analysis)
        # self.graphicsView_2 = QtWidgets.QGraphicsView(self.analysis)
        ############################################
        ### TODO mean distance plot at the end of day for analysis
        ### for person count 
        self.graphicsView = pg.PlotWidget(self.analysis)
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayout.addWidget(self.graphicsView)
        self.graphicsView.setBackground('w')
        self.graphicsView.addLegend()

        ## For mask and non mask count
        self.graphicsView_2 = pg.PlotWidget(self.analysis)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.horizontalLayout.addWidget(self.graphicsView_2)
        self.graphicsView_2.setBackground('w')
        self.graphicsView_2.addLegend()


        # self.graphicsView.plot(hour, temperature)
        ############################################
        self.x = list(range(1000))  # 100 time points
        self.y1 = [0]*1000  # 1000 data points
        self.y2 = [0]*1000  # 1000 data points
        self.y3 = [0]*1000  # 1000 data points
        #person
        pen = pg.mkPen(color=(0, 0, 255))
        self.plot_line1 = self.graphicsView.plot(self.x, self.y1, pen=pen, name = "persons")
        self.graphicsView.setYRange(0, 20, padding=0)        

        pen = pg.mkPen(color=(0, 255, 0)) #RGB
        self.plot_line2 = self.graphicsView_2.plot(self.x, self.y2, pen=pen, name = "person with mask") # mask
        pen = pg.mkPen(color=(255, 0, 0))
        self.plot_line3 = self.graphicsView_2.plot(self.x, self.y3, pen=pen, name='person w/o mask') # no mask
        self.graphicsView_2.setYRange(0, 20, padding=0)
        ############################################


        self.horizontalLayout.setStretch(0, 5)
        self.horizontalLayout.setStretch(1, 5)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_2.setStretch(0, 6)
        self.verticalLayout_2.setStretch(1, 4)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.tabWidget.addTab(self.analysis, "")
        self.info = QtWidgets.QWidget()
        self.info.setObjectName("info")
        self.widget = QtWidgets.QWidget(self.info)
        self.widget.setGeometry(QtCore.QRect(10, 40, 571, 58))
        self.widget.setObjectName("widget")
        self.formLayout_2 = QtWidgets.QFormLayout(self.widget)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.mob_btn = QtWidgets.QPushButton(self.widget)
        self.mob_btn.setObjectName("mob_btn")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.mob_btn)
        self.mob = QtWidgets.QLineEdit(self.widget)
        self.mob.setObjectName("mob")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.mob)
        self.eid_btn = QtWidgets.QPushButton(self.widget)
        self.eid_btn.setObjectName("eid_btn")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.eid_btn)
        self.eid = QtWidgets.QLineEdit(self.widget)
        self.eid.setObjectName("eid")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.eid)
        self.tabWidget.addTab(self.info, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.screen.setText(_translate("MainWindow", "Screen"))
        self.url_btn.setText(_translate("MainWindow", "Enter"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.stream), _translate("MainWindow", "Tab 1"))
        self.label.setText(_translate("MainWindow", "Screen"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.analysis), _translate("MainWindow", "Tab 2"))
        self.mob_btn.setText(_translate("MainWindow", "Mobile No."))
        self.eid_btn.setText(_translate("MainWindow", "Email Id"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.info), _translate("MainWindow", "Page"))

    def start_screen(self):
        video_url = self.url.text()

        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = worker()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(lambda: self.worker.run(video_url))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.change_pixmap_signal.connect(self.set_screen)
        self.worker.update_plot_signal.connect(self.update_plot_data)
        # Step 6: Start the thread
        self.thread.start()
        print("end")

    ### place image to screen
    def set_screen(self, image):
        self.tmp = image
        if image.shape[0] > 461:
            image = imutils.resize(image,height=461)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # frame = get_realtime_infer(frame)
        image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        ### auto scale image to window size
        # self.screen.setScaledContents(True)
        self.screen.setPixmap(QtGui.QPixmap.fromImage(image))
        print("end set_screen")

    def update_plot_data(self, person_count, mask_count, no_mask_count, centriods):

        #############################
        ## check social distance
        dst_mtx = distance_matrix(centriods)
        social_dst = len(np.argwhere(dst_mtx>social_distance_thresh)) == 0

        ## update the tab1 o/p screen
        if social_dst==False:
            self.op.setTextColor(QColor(255, 0, 0))
            self.op_text += f'***********\nALERT: maintain social distance\n***********'


        if len(self.op_text)>1000: self.op_text = f''
        self.op_text += f'Person count: {person_count} mask count: {mask_count} non mask count: {no_mask_count}\n'


        self.op.setText(self.op_text)
        self.op.setTextColor(QColor(0, 0, 0))

        self.x = self.x[1:]  # Remove the first y element.
        self.x.append(self.x[-1] + 1)  # Add a new value 1 higher than the last.


        #############################
        self.y1 = self.y1[1:]  # Remove the first
        self.y1.append(person_count)
        self.plot_line1.setData(self.x, self.y1)

        self.y2 = self.y2[1:]  # Remove the first
        self.y2.append(mask_count)
        self.plot_line2.setData(self.x, self.y2)

        self.y3 = self.y3[1:]  # Remove the first
        self.y3.append(no_mask_count)
        self.plot_line3.setData(self.x, self.y3)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
