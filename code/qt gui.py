# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:12:50 2024

@author: Lenovo
"""
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QGridLayout, QScrollArea
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.button = QPushButton('Open Images', self)
        self.button.clicked.connect(self.openImages)

        self.scrollArea = QScrollArea(self)  # Scroll area to contain the image container
        self.imageContainer = QWidget()  # Container for the images
        self.imageLayout = QGridLayout(self.imageContainer)  # Grid layout for multiple rows and columns
        self.scrollArea.setWidget(self.imageContainer)
        self.scrollArea.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.button)
        layout.addWidget(self.scrollArea)

        self.setWindowTitle('Image Viewer')
        self.setGeometry(300, 300, 800, 600) 

    def openImages(self):
        options = QFileDialog.Options()
        initialPath = "C:\\Users\\Lenovo\\Pictures\\Camera Roll"
        fileNames, _ = QFileDialog.getOpenFileNames(self, "Open Images", initialPath, "All Files (*);;JPEG Files (*.jpeg);;PNG Files (*.png)", options=options)
        if fileNames:
            self.fileNames = fileNames  # Store filenames to reuse during resizing
            self.updateGridLayout()

    def updateGridLayout(self):
        # Clear existing widgets in the layout
        for i in reversed(range(self.imageLayout.count())):
            widgetToRemove = self.imageLayout.itemAt(i).widget()
            if widgetToRemove:
                widgetToRemove.setParent(None)

        # Add images to the grid
        num_cols = max(1, self.width() // 200)  # Calculate number of columns based on current window width
        for i, fileName in enumerate(self.fileNames):
            pixmap = QPixmap(fileName)
            label = QLabel(self)
            # Set fixed aspect ratio scaling
            label.setPixmap(pixmap.scaled(200, 200, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            row = i // num_cols
            col = i % num_cols
            self.imageLayout.addWidget(label, row, col)

    def resizeEvent(self, event):
        # Re-calculate grid layout when window size changes
        if hasattr(self, 'fileNames'):
            self.updateGridLayout()
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageViewer()
    ex.show()
    sys.exit(app.exec_())