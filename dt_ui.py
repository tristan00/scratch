from dt import Tree
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *



class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.show()

        w = QWidget()
        hb = QHBoxLayout()

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

    app = QApplication([])
    window = MainWindow()