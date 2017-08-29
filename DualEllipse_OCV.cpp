#include<opencv2\opencv.hpp>

using namespace std;

#define NO_ERROR 0

#define NOT_A_PROPER_ELLIPSE -1
#define NO_ELLIPSES_DETECTED -2

///////////////////////////////////   Dual Conic Fitting   /////////////////////////////////////////////////
//! @fn int DualConicFitting(std::vector<cv::Point2d> roi_pixel, const cv::Mat& dx, const cv::Mat& dy, cv::Mat &conic)
/*! @brief This Function Fits the Ellipse to the contour points using Dual Conic method

In Projective Geometry, lines are dual to the points
here we are utilizing that property and we form lines based on the contour points and fit the ellipse
the duals are related by an inverse operation, finally the result is inverted to obtain back the Ellispe parameters
this function accepts the Ellipse points and gradients along the x and y axis as the input and returns the parametric matrix as output

@author Gopiraj

@param roi_pixel - pixel locations of the ellipse
@param dx, dy - x and y gradients in the image
@param Normalization - when this flag is set, the input data is Normalized

@retval conic - ellipse in the conic form [A B C D E F]

*/
template<typename T>
int DualConicFitting(const std::vector<cv::Point_<T>>& roi_pixel, const cv::Mat& dx, const cv::Mat& dy, cv::Mat& conic, const bool& Normalization)
{
	/*
	To fit ellipse using dual conic model.

	Ref :	\93Precise ellipse estimation without contour point extraction\94,
	Jean-Nicolas Ouellet & Patrick Hebert, Machine Vision and Applications(2009) 21:59-67
	2009-Ouellet-Hebert-Precise ellipse estimation without contour point extraction.pdf
	*/

	int error = NO_ERROR;

	int roi_size = roi_pixel.size();

	//a = dx at the contour points 
	//b = dy at the contour points
	//c = -(dx*x + dy*y)
	cv::Mat1d a = cv::Mat::zeros(roi_size, 1, CV_64F);
	cv::Mat1d b = cv::Mat::zeros(roi_size, 1, CV_64F);
	cv::Mat1d c = cv::Mat::zeros(roi_size, 1, CV_64F);


	for (int i = 0; i < roi_size; i++)
	{
		a(i, 0) = dx.at<float>(roi_pixel[i]);
		b(i, 0) = dy.at<float>(roi_pixel[i]);
		c(i, 0) = -(dx.at<float>(roi_pixel[i])*roi_pixel[i].x + dy.at<float>(roi_pixel[i])*roi_pixel[i].y);
	}

	cv::Mat1d H = cv::Mat::zeros(3, 3, CV_64F);

	if (Normalization)
	{
		cv::Mat1d M = cv::Mat::zeros(roi_size, 2, CV_64F);
		cv::Mat1d B = -c;
		for (int i = 0; i < roi_size; i++)
		{
			M(i, 0) = -b(i, 0);
			M(i, 1) = a(i, 0);
		}
		//finding the pseudo inverse of the non-square matrix
		// pinv(A) = A' * (A * A')^-1
		invert(M, M, cv::DECOMP_SVD);
		//Mat1d mpts = (M.t()*(M*M.t()).inv()) * B;
		cv::Mat1d mpts = M * B;

		H(0, 0) = 1; H(0, 2) = mpts(0, 0);
		H(1, 1) = 1; H(1, 2) = mpts(1, 0);
		H(2, 2) = 1;

		cv::Mat1d Lnorm = cv::Mat::zeros(roi_size, 3, CV_64F);
		for (int i = 0; i < roi_size; i++)
		{
			Lnorm(i, 0) = a(i, 0);
			Lnorm(i, 1) = b(i, 0);
			Lnorm(i, 2) = c(i, 0);
		}
		Lnorm = (H.t() * Lnorm.t()).t();

		for (int i = 0; i < roi_size; i++)
		{
			a(i, 0) = Lnorm(i, 0);
			b(i, 0) = Lnorm(i, 1);
			c(i, 0) = Lnorm(i, 2);
		}
		mpts.release();
		B.release();
		M.release();
		Lnorm.release();
	}

	std::vector<double> a2, ab, b2, ac, bc, c2;

	for (int i = 0; i < roi_size; i++)
	{
		a2.push_back(a(i, 0) * a(i, 0));
		ab.push_back(a(i, 0) * b(i, 0));
		b2.push_back(b(i, 0) * b(i, 0));
		ac.push_back(a(i, 0) * c(i, 0));
		bc.push_back(b(i, 0) * c(i, 0));
		c2.push_back(c(i, 0) * c(i, 0));
	}
	a.release();
	b.release();
	c.release();

	//Forming the A matrix and B matrix
	// AA = [sum(a2.^2),  sum(a2.*ab), sum(a2.*b2), sum(a2.*ac), sum(a2.*bc)
	//     sum(a2.*ab), sum(ab. ^ 2), sum(ab.*b2), sum(ab.*ac), sum(ab.*bc)
	// 	   sum(a2.*b2), sum(ab.*b2), sum(b2. ^ 2), sum(b2.*ac), sum(b2.*bc)
	// 	   sum(a2.*ac), sum(ab.*ac), sum(b2.*ac), sum(ac. ^ 2), sum(ac.*bc)
	// 	   sum(a2.*bc), sum(ab.*bc), sum(b2.*bc), sum(ac.*bc), sum(bc. ^ 2)];
	// 
	// BB =  [sum(-(c.^ 2).*a2)
	// 		  sum(-(c.^ 2).*ab)
	// 		  sum(-(c.^ 2).*b2)
	// 		  sum(-(c.^ 2).*ac)
	// 		  sum(-(c.^ 2).*bc)];
	//
	cv::Mat1d AA = cv::Mat::zeros(5, 5, CV_64F);
	cv::Mat1d AA_inv = cv::Mat::zeros(5, 5, CV_64F);
	cv::Mat1d BB = cv::Mat::zeros(5, 1, CV_64F);
	cv::Mat C = cv::Mat::zeros(6, 1, CV_64F);
	double BTB = 0;

	for (int i = 0; i < roi_size; ++i)
	{
		AA(0, 0) += (a2[i] * a2[i]);
		AA(0, 1) = AA(1, 0) += (a2[i] * ab[i]);
		AA(0, 2) = AA(2, 0) += (a2[i] * b2[i]);
		AA(0, 3) = AA(3, 0) += (a2[i] * ac[i]);
		AA(0, 4) = AA(4, 0) += (a2[i] * bc[i]);

		AA(1, 1) += (ab[i] * ab[i]);
		AA(1, 2) = AA(2, 1) += (ab[i] * b2[i]);
		AA(1, 3) = AA(3, 1) += (ab[i] * ac[i]);
		AA(1, 4) = AA(4, 1) += (ab[i] * bc[i]);

		AA(2, 2) += (b2[i] * b2[i]);
		AA(2, 3) = AA(3, 2) += (b2[i] * ac[i]);
		AA(2, 4) = AA(4, 2) += (b2[i] * bc[i]);

		AA(3, 3) += (ac[i] * ac[i]);
		AA(3, 4) = AA(4, 3) += (ac[i] * bc[i]);

		AA(4, 4) += (bc[i] * bc[i]);

		BB(0, 0) += (-c2[i] * a2[i]);
		BB(1, 0) += (-c2[i] * ab[i]);
		BB(2, 0) += (-c2[i] * b2[i]);
		BB(3, 0) += (-c2[i] * ac[i]);
		BB(4, 0) += (-c2[i] * bc[i]);

		BTB += (c2[i] * c2[i]);
	}

	a2.clear();
	std::vector<double>().swap(a2);
	ab.clear();
	std::vector<double>().swap(ab);
	b2.clear();
	std::vector<double>().swap(b2);
	bc.clear();
	std::vector<double>().swap(bc);
	c2.clear();
	std::vector<double>().swap(c2);
	ac.clear();
	std::vector<double>().swap(ac);

	//Solving the Least squares problem
	// X = A^-1 * B
	//
	cv::invert(AA, AA_inv, cv::DECOMP_SVD);
	cv::Mat1d sol = AA_inv * BB;

	AA_inv.release();

	//-------------------------------------error estimation--------------------------------------//
	double stdcenter[2] = { 0 };
	cv::Mat vt, w, u;

	cv::Mat R = ((sol.t()*AA*sol) - 2 * sol.t()*BB + BTB) / (roi_size - 5);

	cv::Mat1d cvar2_constantVariance = R.at<double>(0, 0)*AA.inv();

	R.release();

	double vD = cvar2_constantVariance(3, 3);
	double vDE = cvar2_constantVariance(3, 4);
	double vE = cvar2_constantVariance(4, 4);

	cv::Mat er = cv::Mat::zeros(2, 2, CV_64F);
	er.at<double>(0, 0) = cvar2_constantVariance(3, 3); er.at<double>(0, 1) = cvar2_constantVariance(3, 4);
	er.at<double>(1, 0) = cvar2_constantVariance(4, 3); er.at<double>(1, 1) = cvar2_constantVariance(4, 4);

	cv::SVDecomp(er, w, u, vt);

	stdcenter[0] = sqrt(w.at<double>(0, 0)) / 4;
	stdcenter[1] = sqrt(w.at<double>(1, 0)) / 4;

	double angleIncertitude = atan2(vt.at<double>(1, 0), vt.at<double>(0, 0));

	if (stdcenter[0] == -1 || stdcenter[0] > 0.075) return NOT_A_PROPER_ELLIPSE;

	er.release();
	w.release();
	u.release();
	vt.release();
	cvar2_constantVariance.release();
	//----------------end of error estimation---------------------//

	AA.release();
	BB.release();


	cv::Mat1d dCnorm = cv::Mat::zeros(3, 3, CV_64F);
	dCnorm(0, 0) = sol(0, 0);
	dCnorm(0, 1) = dCnorm(1, 0) = sol(1, 0) / 2;
	dCnorm(0, 2) = dCnorm(2, 0) = sol(3, 0) / 2;
	dCnorm(1, 1) = sol(2, 0);
	dCnorm(1, 2) = dCnorm(2, 1) = sol(4, 0) / 2;
	dCnorm(2, 2) = 1;

	sol.release();

	cv::Mat1d dC;

	if (Normalization == 1)   dC = H*dCnorm*H.t();
	else dC = dCnorm;

	dCnorm.release();
	H.release();

	//The DualEllispe is found by inverting the Ellipse matrix found
	C = dC.inv();
	C /= C.at<double>(2, 2);

	dC.release();

	conic = cv::Mat::zeros(6, 1, CV_64FC1);

	conic.at<double>(0, 0) = C.at<double>(0, 0);
	conic.at<double>(1, 0) = C.at<double>(0, 1) * 2;
	conic.at<double>(2, 0) = C.at<double>(1, 1);
	conic.at<double>(3, 0) = C.at<double>(0, 2) * 2;
	conic.at<double>(4, 0) = C.at<double>(2, 1) * 2;
	conic.at<double>(5, 0) = C.at<double>(2, 2);

	C.release();

	return NO_ERROR;

}

///////////////////////////////////   Dual Conic Fitting   /////////////////////////////////////////////////
//! @fn int DualConicFitting(const std::vector<cv::Point_<T>>& roi_pixel, const cv::Mat& dx, const cv::Mat& dy, cv::RotatedRect& ell, const bool& Normalization)
/*! @brief This Function Fits the Ellipse to the contour points using Dual Conic method

In Projective Geometry, lines are dual to the points
here we are utilizing that property and we form lines based on the contour points and fit the ellipse
the duals are related by an inverse operation, finally the result is inverted to obtain back the Ellispe parameters
this function accepts the Ellipse points and gradients along the x and y axis as the input and returns the parametric matrix as output

@author Gopiraj

@param roi_pixel - pixel locations of the ellipse
@param dx, dy - x and y gradients in the image
@param Normalization - when this flag is set, the input data is Normalized

@retval ell - ellipse parameters in the form of OpenCV RotatedRect

*/
template<typename T>
int DualConicFitting(const std::vector<cv::Point_<T>>& roi_pixel, const cv::Mat& dx, const cv::Mat& dy, cv::RotatedRect& ell, const bool& Normalization)
{
	if (roi_pixel.size() <= 0) return NO_ELLIPSES_DETECTED;
	cv::Mat conic;
	int err = 0;
	
	err = DualConicFitting(roi_pixel, dx, dy, conic, Normalization);

	if (err < 0)return err;

	ell = Mat2RotatedRect((cv::Mat1d)convertConicToParametric((cv::Mat1d)conic));

	if (ell.size.area <= 0 || ell.size.height <= 0 || ell.size.width <= 0) return NO_ELLIPSES_DETECTED;

	return NO_ERROR;

}
