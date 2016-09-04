/**
 * Image segmentation using MRF-2.2.
 * Use the potential functions written in "MRFs and CRFs for Vision: Models & Optimization" by Carsten Rother.
 *
 * @author	Gen Nishida
 * @date	9/3/2016
 */

#include "ImageSegmentation.h"
#include "BP-S.h"
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace imgseg {

	/** 3-channel color source image */
	cv::Mat img;

	/** 3-dimensional color space prior (1 - foreground / 0 - background) */
	cv::Mat data_prior;

	/** resolution of each dimension of color space */
	int num_res = 25;

	/**
	 * Discritize the color based on the specified resolution.
	 */
	cv::Vec3b discr(cv::Vec3b col) {
		cv::Vec3b ret;
		ret[0] = (float)col[0] / 256 * num_res;
		ret[1] = (float)col[1] / 256 * num_res;
		ret[2] = (float)col[2] / 256 * num_res;
		return ret;
	}

	/**
	 * Convert the color to a single integer value.
	 */
	int Vec2Int(cv::Vec3b v) {
		int b = (float)v[0] / 256 * num_res;
		int g = (float)v[1] / 256 * num_res;
		int r = (float)v[2] / 256 * num_res;

		return b * num_res * num_res + g * num_res + r;
	}

	/**
	 * Decode a point in the color space from a single integer value.
	 * Note that the color space is discretized by the specified resolution.
	 */
	cv::Vec3b Int2Vec(int a) {
		cv::Vec3b ret;
		ret[0] = a / num_res / num_res;
		ret[1] = (a - ret[0] * num_res * num_res) / num_res;
		ret[2] = a % num_res;

		return ret;
	}

	/**
	 * Define a data cost.
	 */
	MRF::CostVal dCost(int pix, int i) {
		int r = pix / img.cols;
		int c = pix % img.cols;
		cv::Vec3b col = discr(img.at<cv::Vec3b>(r, c));

		double p = data_prior.at<double>(col[0], col[1], col[2]);
		if (i == 0) {
			return p;
		}
		else {
			return (1 - p);
		}
	}

	/**
	 * Define a smoothness cost.
	 */
	MRF::CostVal fnCost(int pix1, int pix2, int i, int j) {
		if (pix2 < pix1) { // ensure that fnCost(pix1, pix2, i, j) == fnCost(pix2, pix1, j, i)
			int tmp;
			tmp = pix1; pix1 = pix2; pix2 = tmp;
			tmp = i; i = j; j = tmp;
		}

		int r1 = pix1 / img.cols;
		int c1 = pix1 % img.cols;
		cv::Vec3b col1 = discr(img.at<cv::Vec3b>(r1, c1));

		int r2 = pix2 / img.cols;
		int c2 = pix2 % img.cols;
		cv::Vec3b col2 = discr(img.at<cv::Vec3b>(r2, c2));

		if (i == j) {
			return 0;
		}
		else {
			double beta = 1;
			double p1 = data_prior.at<double>(col1[0], col1[1], col1[2]);
			double p2 = data_prior.at<double>(col2[0], col2[1], col2[2]);
			return exp(-beta * abs(p1 - p2));
		}
	}

	/** 
	 * Segment the image using a reference mask.
	 * The mask contains bluish color as foreground reference and redish color as background reference.
	 *
	 * @param src_img	image
	 * @param mask		mask
	 * @param dst_img	result
	 */
	bool segment(cv::Mat src_img, cv::Mat mask, cv::Mat& dst_img) {
		// copy the image to img and convert it to 3-channel color image
		if (src_img.channels() == 1) {
			cv::cvtColor(src_img, img, cv::COLOR_GRAY2BGR);
		}
		else if (src_img.channels() == 3) {
			img = src_img.clone();
		}
		else if (src_img.channels() == 4) {
			cv::cvtColor(src_img, img, cv::COLOR_BGRA2BGR);
		}

		// create histogram based on the image and mask
		std::map<int, float> hist;
		std::map<int, float> hist_cnt;
		for (int r = 0; r < mask.rows; ++r) {
			for (int c = 0; c < mask.cols; ++c) {
				cv::Vec3b m = mask.at<cv::Vec3b>(r, c);
				int col = Vec2Int(img.at<cv::Vec3b>(r, c));

				if (m[0] == 255 && m[1] == 255 && m[2] == 255) { // others
				}
				else if (m[1] < 200 && m[2] < 200) { // blue (foreground)
					if (hist.find(col) == hist.end()) {
						hist[col] = 0;
					}
					if (hist_cnt.find(col) == hist_cnt.end()) {
						hist_cnt[col] = 0;
					}
					hist[col]++;
					hist_cnt[col]++;
				}
				else if (m[0] < 200 && m[1] < 200) { // red (background)
					if (hist.find(col) == hist.end()) {
						hist[col] = 0;
					}
					if (hist_cnt.find(col) == hist_cnt.end()) {
						hist_cnt[col] = 0;
					}
					hist_cnt[col]++;
				}
			}
		}

		// normalize the histogram to [0, 1] for each point
		for (auto it = hist.begin(); it != hist.end(); ++it) {
			//std::cout << hist[it->first] << "," << hist_cnt[it->first] << std::endl;
			hist[it->first] = hist[it->first] / hist_cnt[it->first];
			//std::cout << hist[it->first] << std::endl;
		}

		// computing the prior data
		double sigma2 = num_res * num_res / 10 / 10;
		std::vector<int> sizes(3);
		for (int i = 0; i < 3; ++i) sizes[i] = num_res;
		data_prior = cv::Mat(3, sizes.data(), CV_64F, cv::Scalar(0));
		for (int i = 0; i < num_res; ++i) {
			for (int j = 0; j < num_res; ++j) {
				for (int k = 0; k < num_res; ++k) {
					double total = 0;
					double total_weight = 0;

					// weighted sum of labels
					for (auto it = hist.begin(); it != hist.end(); ++it) {
						cv::Vec3b p = Int2Vec(it->first);
						double dist2 = pow(p[0] - i, 2) + pow(p[1] - j, 2) + pow(p[2] - k, 2);
						double weight = exp(-dist2 / sigma2);
						total += it->second * weight;
						total_weight += weight;
					}

					data_prior.at<double>(i, j, k) = total / total_weight;
				}
			}
		}

		int seed = 1124285485;
		srand(seed);

		try {
			DataCost *data = new DataCost(dCost);
			SmoothnessCost *smooth = new SmoothnessCost(fnCost);
			EnergyFunction* energy = new EnergyFunction(data, smooth);

			MRF* mrf = new BPS(img.cols, img.rows, 2, energy);

			// can disable caching of values of general smoothness function:
			//mrf->dontCacheSmoothnessCosts();

			mrf->initialize();
			mrf->clearAnswer();

			MRF::EnergyVal E = mrf->totalEnergy();
			printf("Energy at the Start= %g (%g,%g)\n", (float)E, (float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());

			float tot_t = 0;
			for (int iter = 0; iter < 10; iter++) {
				float t;
				mrf->optimize(10, t);

				E = mrf->totalEnergy();
				tot_t = tot_t + t;
				printf("energy = %g (%f secs)\n", (float)E, tot_t);
			}

			// create the result image
			dst_img = img.clone();
			for (int r = 0; r < img.rows; ++r) {
				for (int c = 0; c < img.cols; ++c) {
					MRF::Label label = mrf->getLabel(r * img.cols + c);
					if (label == 1) {
						// keep the pixel color
					}
					else {
						dst_img.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
					}
				}
			}

			delete mrf;
		}
		catch (std::bad_alloc) {
			fprintf(stderr, "*** Error: not enough memory\n");
			return false;
		}

		return true;
	}

}
