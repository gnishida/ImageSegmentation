#include "Canvas.h"
#include <QPainter>
#include <QMouseEvent>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ImageSegmentation.h"
#include <iostream>

Canvas::Canvas(QWidget *parent) : QWidget(parent) {
	lineWidth = 10;
}

Canvas::~Canvas() {
}

void Canvas::load(const QString& filename) {
	image.load(filename);
	update();
}

void Canvas::segment() {
	cv::Mat src_img = QImageToCvMat(image);

	// create a mask image
	QImage maskImage(image.width(), image.height(), QImage::Format_RGB888);
	QPainter painter(&maskImage);
	painter.fillRect(0, 0, maskImage.width(), maskImage.height(), QColor(255, 255, 255));
	drawMaskLines(painter);
	cv::Mat mask = QImageToCvMat(maskImage);

	// segmentation
	cv::Mat dst_img;
	imgseg::segment(src_img, mask, dst_img);
	image = cvMatToQImage(dst_img);
	//cv::imwrite("result.png", dst_img);

	foreground_lines.clear();
	background_lines.clear();

	update();
}

void Canvas::drawMaskLines(QPainter& painter) {
	painter.setPen(QPen(QBrush(QColor(0, 0, 255)), lineWidth, Qt::SolidLine, Qt::RoundCap));
	for (auto line : foreground_lines) {
		for (int i = 0; i < line.size() - 1; ++i) {
			painter.drawLine(line[i].x(), line[i].y(), line[i + 1].x(), line[i + 1].y());
		}
	}
	painter.setPen(QPen(QBrush(QColor(255, 0, 0)), lineWidth, Qt::SolidLine, Qt::RoundCap));
	for (auto line : background_lines) {
		for (int i = 0; i < line.size() - 1; ++i) {
			painter.drawLine(line[i].x(), line[i].y(), line[i + 1].x(), line[i + 1].y());
		}
	}
}

QImage Canvas::cvMatToQImage(const cv::Mat &inMat) {
	QImage ret;
	static QVector<QRgb> sColorTable(256);

	switch (inMat.type()) {
	case CV_8UC4: // 8-bit, 4 channel
		ret = QImage(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_ARGB32);
		break;
	case CV_8UC3: // 8-bit, 3 channel
		ret = QImage(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_RGB888).rgbSwapped();
		break;
	case CV_8UC1: // 8-bit, 1 channel
		// only create our color table the first time
		if (sColorTable.isEmpty()) {
			for (int i = 0; i < 256; ++i) {
				sColorTable[i] = qRgb(i, i, i);
			}
		}

		ret = QImage(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_Indexed8);
		ret.setColorTable(sColorTable);

		break;
	default:
		std::cerr << "cv::Mat image type is not supported:" << inMat.type() << std::endl;
		break;
	}

	return ret;
}

cv::Mat Canvas::QImageToCvMat(const QImage &inImage) {
	QImage swapped;

	switch (inImage.format()) {
	case QImage::Format_ARGB32: // 8-bit, 4 channel
	case QImage::Format_ARGB32_Premultiplied: // 8-bit, 4 channel
		return cv::Mat(inImage.height(), inImage.width(), CV_8UC4, const_cast<uchar*>(inImage.bits()), static_cast<size_t>(inImage.bytesPerLine())).clone();
	case QImage::Format_RGB32: // 8-bit, 3 channel
	case QImage::Format_RGB888: // 8-bit, 3 channel
		if (inImage.format() == QImage::Format_RGB32)
			swapped = inImage.convertToFormat(QImage::Format_RGB888).rgbSwapped();
		else 
			swapped = inImage.rgbSwapped();

		return cv::Mat(swapped.height(), swapped.width(), CV_8UC3, const_cast<uchar*>(swapped.bits()), static_cast<size_t>(swapped.bytesPerLine())).clone();
	case QImage::Format_Indexed8: // 8-bit, 1 channel
		return cv::Mat(inImage.height(), inImage.width(), CV_8UC1, const_cast<uchar*>(inImage.bits()), static_cast<size_t>(inImage.bytesPerLine())).clone();
	default:
		std::cerr << "cv::Mat image type is not supported:" << inImage.format() << std::endl;
		break;
	}

	return cv::Mat();
}

void Canvas::paintEvent(QPaintEvent *event) {
	QPainter painter(this);
	painter.drawImage(0, 0, image, 0, 0);

	drawMaskLines(painter);
	/*
	painter.setPen(QPen(QBrush(QColor(0, 0, 255)), lineWidth, Qt::SolidLine, Qt::RoundCap));
	for (auto line : foreground_lines) {
		for (int i = 0; i < line.size() - 1; ++i) {
			painter.drawLine(line[i].x(), line[i].y(), line[i + 1].x(), line[i + 1].y());
		}
	}
	painter.setPen(QPen(QBrush(QColor(255, 0, 0)), lineWidth, Qt::SolidLine, Qt::RoundCap));
	for (auto line : background_lines) {
		for (int i = 0; i < line.size() - 1; ++i) {
			painter.drawLine(line[i].x(), line[i].y(), line[i + 1].x(), line[i + 1].y());
		}
	}
	*/
}

void Canvas::mousePressEvent(QMouseEvent *event) {
	lastPoint = event->pos();
	if (event->button() == Qt::LeftButton) {
		foreground_lines.resize(foreground_lines.size() + 1);
		foreground_lines.back().push_back(event->pos());
	}
	else if (event->button() == Qt::RightButton) {
		background_lines.resize(background_lines.size() + 1);
		background_lines.back().push_back(event->pos());
	}

	update();
}

void Canvas::mouseMoveEvent(QMouseEvent *event) {
	if (event->buttons() & Qt::LeftButton) {
		foreground_lines.back().push_back(event->pos());
	}
	else if (event->buttons() & Qt::RightButton) {
		background_lines.back().push_back(event->pos());
	}

	update();
}

void Canvas::mouseReleaseEvent(QMouseEvent *event) {
}