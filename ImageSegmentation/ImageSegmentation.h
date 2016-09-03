#pragma once

#include <opencv2/core.hpp>
#include "mrf.h"

namespace imgseg {

	cv::Vec3b discr(cv::Vec3b col);
	int Vec2Int(cv::Vec3b v);
	cv::Vec3b Int2Vec(int a);
	MRF::CostVal dCost(int pix, int i);
	MRF::CostVal fnCost(int pix1, int pix2, int i, int j);
	EnergyFunction* generate_DataFUNCTION_SmoothGENERAL_FUNCTION();
	bool segment(cv::Mat src_img, cv::Mat mask, cv::Mat& dst_img);

}


