#pragma once

#include <QWidget>
#include <QImage>
#include <QPainter>
#include <opencv2/core.hpp>

class Canvas : public QWidget {
private:
	QImage image;
	QPoint lastPoint;
	std::vector<std::vector<QPoint>> foreground_lines;
	std::vector<std::vector<QPoint>> background_lines;
	int lineWidth;

public:
    Canvas(QWidget *parent = 0);
    ~Canvas();

	void load(const QString& filename);
	void segment();
	void drawMaskLines(QPainter& painter);
	QImage cvMatToQImage(const cv::Mat& inMat);
	cv::Mat QImageToCvMat(const QImage& inImage);

protected:
	void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;
	void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
	void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
	void mouseReleaseEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
};


