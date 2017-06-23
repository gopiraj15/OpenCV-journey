#include<opencv2\opencv.hpp>
#include<cmath>
#include<iostream>

#define SQR(x) x*x

using namespace cv;
using namespace std;

struct HoughParams
{
	//minimum and Maximum size of the major axis of the ellipse
	int minMajAxisLength, maxMajAxisLength;
	//minimum and maximum angle span of the ellipse
	int Angle, angleSpan;
	//minimum Ratio between the minor axis and the major axis of the ellipse
	double AxisRatio;
	double smoothStdDev;
	bool uniformWeights;
	//number of best fit parameters
	int numBest;
	//for Randomizing the Hough Transform
	int randomize;
	//Constructor for Loading default values
	HoughParams()
	{
		minMajAxisLength = 100;
		maxMajAxisLength = 200;
		Angle = 0;
		angleSpan = 0;
		AxisRatio = 0.5;
		uniformWeights = true;
		numBest = 3;
		randomize = 2;
		smoothStdDev = 1;
	}
	//Constructor to get parameters from the default ellipse
	HoughParams(const cv::RotatedRect& ell)
	{
		double MajAxis = ell.size.width > ell.size.height ? ell.size.width : ell.size.height;
		double MinAxis = ell.size.width < ell.size.height ? ell.size.width : ell.size.height;

		minMajAxisLength = round(MajAxis * 0.75);
		maxMajAxisLength = round(MajAxis * 1.25);
		Angle = round(ell.angle);
		angleSpan = 20;
		
		AxisRatio = 0.5;
		smoothStdDev = 1;
		uniformWeights = true;
		numBest = 3;
		randomize = 2;
	}

	~HoughParams()
	{

	}
};
//This function performs full one dimensional convolution
template<typename T>
void conv(std::vector<T> const &f, std::vector<T> const &g, std::vector<T>& out)
{
	int const nf = f.size();
	int const ng = g.size();
	int const n = nf + ng - 1; //convolution length
	out.resize(n, T());
	for (auto i(0); i < n; ++i)
	{
		int const jmn = (i >= ng - 1) ? i - (ng - 1) : 0;
		int const jmx = (i <  nf - 1) ? i : nf - 1;
		for (auto j(jmn); j <= jmx; ++j)
		{
			out[i] += (f[j] * g[i - j]);
		}//end for
	}//end for
}//end conv

// Overview:
// --------
// Fits an ellipse by examining all possible major axes(all pairs of points) and
// getting the minor axis using Hough transform.The algorithm complexity depends on
// the number of valid non - zero points, therefore it is beneficial to provide as many
// restrictions in the "params" input arguments as possible if there is any prior
// knowledge about the problem.
//
// The code can be quite memory intensive.If you get out of memory errors, either 
// downsample the input image or somehow decrease the number of non - zero points in it.
// It can deal with big amount of noise but can have severe problem with occlusions(major axis
// end points need to be visible)
//
// Input arguments :
// --------
// src
//   -One - channel input image(greyscale or binary).
// params
//   -Parameters of the algorithm :
//       minMajorAxis : Minimal length of major axis accepted.
//       maxMajorAxis : Maximal length of major axis accepted.
//        Angle, AngleSpanSpan : Specification of restriction on the angle of the major axis in degrees.
//                                If rotationSpan is in(0, 90), only angles within[rotation - rotationSpan,
//                               rotation + rotationSpan] are accepted.
//       AxisRatio : Minimal aspect ratio of an ellipse(in(0, 1))
//       randomize : Subsampling of all possible point pairs.Instead of examining all N*N pairs, runs
//                  only on N*randomize pairs.If 0, randomization is turned off.
//       numBest : Top numBest to return
//       smoothStddev : In order to provide more stability of the solution, the accumulator is convolved with
//                     a gaussian kernel.This parameter specifies its standard deviation in pixels.
//
// Return value :
// --------
// Returns a matrix of best fits.Each row(there are params.numBest of them) contains six elements :
// [x0 y0 a b alpha score] being the center of the ellipse, its major and minor axis, its angle in degrees and score.
//
// Based on :
// --------
// -"A New Efficient Ellipse Detection Method" (Yonghong Xie Qiang, Qiang Ji / 2002)
// -random subsampling inspired by "Randomized Hough Transform for Ellipse Detection with Result Clustering"
// (CA Basca, M Talos, R Brad / 2005)
//
//TO DO:
//To include exit parameters when the algorithm fails
void EllipticalHoughTransform_1(cv::Mat& src, HoughParams params, cv::RotatedRect& finalEll)
{
	double eps = 0.0001;
	Mat1f bestFits = Mat::zeros(params.numBest, 6, CV_32FC1);
	params.angleSpan = std::min(params.angleSpan, 90);

	Mat1f h = getGaussianKernel(7, params.smoothStdDev);

	vector<double>H;
	for (int i = 0; i < h.rows; ++i) H.push_back(h(i, 0));
	h.release();

	vector<int> X, Y;
	
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			if (src.at<uchar>(i, j) == 0) continue;
			X.push_back(i);
			Y.push_back(j);
		}
	}

	int N = X.size();

	vector<int> idx, I, J;

	Mat1f distsSq = Mat::zeros(N, N, CV_32FC1);

	double minmajsq = pow(params.minMajAxisLength, 2);
	double maxmajsq = pow(params.maxMajAxisLength, 2);
	//computing pairwise distances between points and filter
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			double d = pow(X[j] - X[i], 2) + pow(Y[j] - Y[i], 2);
			distsSq(i, j) = d;
			if (d < minmajsq || d > maxmajsq)continue;
			if (j > i) continue;
			I.push_back(j);
			J.push_back(i);
		}
	}

	//computing pairwise angles and filter
	if (params.angleSpan > 0)
	{
		double tmp = (params.Angle - params.angleSpan)*CV_PI / 180;
		double tanLo = round(tan((params.Angle - params.angleSpan)*CV_PI / 180));
		double tanHi = round(tan((params.Angle + params.angleSpan)*CV_PI / 180));

		if (tanLo == 90 || tanLo == -90 || tanLo == 270 || tanLo == -270 || tanLo > 360 || tanLo < -360)  tanLo = numeric_limits<double>::infinity();
		if (tanHi == 90 || tanHi == -90 || tanHi == 270 || tanHi == -270 || tanHi > 360 || tanHi <-360)  tanHi = numeric_limits<double>::infinity();

		for (int i = 0; i < I.size(); ++i)
		{
			double tangents;
			if ((X[I[i]] - X[J[i]]) == 0) tangents = numeric_limits<double>::infinity();
			else tangents = (Y[I[i]] - Y[J[i]]) / (X[I[i]] - X[J[i]]);

			if (abs(tanLo) < abs(tanHi))
			{
				if (tangents > tanLo && tangents < tanHi) idx.push_back(i);
			}
			else
			{
				if (tangents > tanLo || tangents < tanHi) idx.push_back(i);
			}

		}
		vector<int> tmpI, tmpJ;
		tmpI = I; tmpJ = J;
		I.clear(); J.clear();

		for (int i = 0; i < idx.size(); ++i)
		{
			I.push_back(tmpI[idx[i]]);
			J.push_back(tmpJ[idx[i]]);
		}
		tmpI.clear(); tmpJ.clear();
		idx.clear();
	}
	
	vector<int> pairSubset;
	for (int i = 0; i < I.size(); ++i) pairSubset.push_back(i);
	
	//computing random choice and filter
	if (params.randomize > 0)
	{
		randShuffle(pairSubset);
		int num = std::min((int)pairSubset.size(), N*params.randomize);
		pairSubset.resize(num);

	}

	//check out all hypotheses
	for (int p = 0; p < pairSubset.size(); ++p)
	{
		double x1, x2, y1, y2;
		x1 = X[I[pairSubset[p]]]; y1 = Y[I[pairSubset[p]]];
		x2 = X[J[pairSubset[p]]]; y2 = Y[J[pairSubset[p]]];

		//Computing the centers
		double x0 = (x1 + x2) / 2;
		double y0 = (y1 + y2) / 2;

		//Computing the major axis
		double aSq = distsSq(I[pairSubset[p]], J[pairSubset[p]]) / 4;

		vector<double> thirdPtDistsSq, K(X.size()), cosTau, sinTauSq, fSq;
		
		//get minor ax propositions for all other points
		for (int i = 0; i < X.size(); ++i)
		{
			double tmp = pow(X[i] - x0, 2) + pow(Y[i] - y0, 2);
			thirdPtDistsSq.push_back(tmp);
			if (tmp > aSq) continue;
			K[i] = 1;
			fSq.push_back(pow(X[i] - x2, 2) + pow(Y[i] - y2, 2));
		}
		
		int num = 0;
		for (int i = 0; i < K.size(); ++i)
		{
			if (K[i] == 0) continue;
			double tmp = (aSq + thirdPtDistsSq[i] - fSq[num++])  / (2 * sqrt(aSq*thirdPtDistsSq[i]));

			tmp = std::min(1.0, std::max(-1.0, tmp));
			cosTau.push_back(tmp);
			sinTauSq.push_back(1 - SQR(tmp));
		}
		num = 0;
		vector<double> b;
		//proper bins for b
		for (int i = 0; i < K.size(); ++i)
		{
			if (K[i] == 0)continue;
			b.push_back(sqrt((aSq * thirdPtDistsSq[i] * sinTauSq[num]) / (aSq - thirdPtDistsSq[i] * pow(cosTau[num], 2) + eps)));
			num++;
		}

		thirdPtDistsSq.clear();
		fSq.clear();
		K.clear();
		sinTauSq.clear();
		cosTau.clear();

		vector<int> idxs;

		for (int i = 0; i < b.size(); ++i) idxs.push_back(ceil(b[i] + eps));
		b.clear();

		vector<double> accumarray, tmpaccum(2 + *max_element(idxs.begin(), idxs.end()));

		for (int i = 0; i < idxs.size(); i++) tmpaccum[idxs[i]]++;

		idxs.clear();

		//a bit of smoothing and finding the most busy bin
		conv(tmpaccum, H, accumarray);

		tmpaccum.clear();

		for (int i = 0; i < H.size() / 2; ++i)
		{
			accumarray.erase(accumarray.begin());
			accumarray.erase(accumarray.begin() + accumarray.size() - 1);
		}

		for (int i = 0; i < accumarray.size(); ++i)
		{
			if (i > ceil(sqrt(aSq)*params.AxisRatio)) break;
			accumarray[i] = 0;
		}

		double score = *max_element(accumarray.begin(), accumarray.end());

		int index = distance(accumarray.begin(), max_element(accumarray.begin(), accumarray.end()));

		accumarray.clear();

		//keeping only the params.numBest best hypothesis(no non - maxima suppresion)
		if (bestFits(params.numBest - 1, params.numBest - 1) < score)
		{
			bestFits(params.numBest - 1, 0) = x0;
			bestFits(params.numBest - 1, 1) = y0;
			bestFits(params.numBest - 1, 2) = sqrt(aSq);
			bestFits(params.numBest - 1, 3) = index;
			bestFits(params.numBest - 1, 4) = atan((y1 - y2) / (x1 - x2)) * 180 / CV_PI;
			bestFits(params.numBest - 1, 5) = score;

			if (params.numBest > 1)
			{
				cv::Mat one = bestFits.col(5);
				cv::Mat1i idx;
				cv::sortIdx(one, idx, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);
				cv::Mat result(bestFits.rows, bestFits.cols, bestFits.type());

				for (int i = 0; i < idx.rows; ++i)
				{
					bestFits.row(idx(i, 0)).copyTo(result.row(i));
				}

				bestFits = result.clone();
				result.release();
				idx.release();
				one.release();
			}

		}
		//cout << bestFits << endl;
		
	}
	distsSq.release();
	pairSubset.clear();
	I.clear();
	J.clear();
	X.clear();
	Y.clear();

	finalEll.center.x = round(bestFits(0, 1));
	finalEll.center.y = round(bestFits(0, 0));
	finalEll.size.height = 2 * bestFits(0, 2);
	finalEll.size.width = 2 * bestFits(0, 3);
	finalEll.angle = bestFits(0, 4);

	std::cout << bestFits << endl;

	bestFits.release();
	

}
