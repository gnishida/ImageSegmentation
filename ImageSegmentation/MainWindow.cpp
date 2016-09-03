#include "MainWindow.h"
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
	ui.setupUi(this);

	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(onOpen()));
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(close()));
	connect(ui.actionSegmentation, SIGNAL(triggered()), this, SLOT(onSegmentation()));

	this->setCentralWidget(&canvas);
}

MainWindow::~MainWindow() {

}

void MainWindow::onOpen() {
	QString filename = QFileDialog::getOpenFileName(this, tr("Load Image file..."), "", tr("Image Files (*.png *.jpg *bmp)"));
	if (filename.isEmpty()) return;

	canvas.load(filename);
}

void MainWindow::onSegmentation() {
	canvas.segment();
}