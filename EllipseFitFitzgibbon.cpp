#include<opencv2\opencv.hpp>

using namespace std;

template <typename T> static
int sign(const T &val) { return (val > 0) - (val < 0); }

template<typename T>
cv::RotatedRect conicToParametric(cv::Mat_<T>& par)
{
	double thetarad = 0.5*atan2(par(1), par(0) - par(2));
	double cost = cos(thetarad);
	double sint = sin(thetarad);
	double sin_squared = sint*sint;
	double cos_squared = cost*cost;
	double cos_sin = sint*cost;

	double Ao = par(5);
	double Au = par(3)*cost + par(4)*sint;
	double Av = -par(3)*sint + par(4)*cost;
	double Auu = par(0)*cos_squared + par(2)*sin_squared + par(1)*cos_sin;
	double Avv = par(0)*sin_squared + par(2)*cos_squared - par(1)*cos_sin;

	//ROTATED = [Ao Au Av Auu Avv]
	double tuCentre = -Au / (2 * Auu);
	double tvCentre = -Av / (2 * Avv);
	double wCentre = Ao - Auu*tuCentre*tuCentre - Avv*tvCentre*tvCentre;

	double uCentre = tuCentre*cost - tvCentre*sint;
	double vCentre = tuCentre*sint + tvCentre*cost;

	double Ru = -wCentre / Auu;
	double Rv = -wCentre / Avv;

	Ru = sqrt(abs(Ru))*sign(Ru);
	Rv = sqrt(abs(Rv))*sign(Rv);

	cv::RotatedRect res;

	res.center.x = uCentre;
	res.center.y = vCentre;
	res.size.width = 2 * Ru;
	res.size.height = 2 * Rv;
	res.angle = thetarad * 180 / CV_PI;

	return res;
}

template<typename T>
cv::RotatedRect EllipseFitByFitzgibbon(const std::vector<cv::Point_<T>>& pts)
{
	if (pts.size() <= 5) return cv::RotatedRect();

	int length = pts.size();

	cv::Mat XY = cv::Mat(pts).reshape(1);


	//Normalizing the data
	//Calculating the centroid of the data
	double mx = mean(XY.col(0))[0];
	double my = mean(XY.col(1))[0];

	double minX, maxX, minY, maxY, sx = 0, sy = 0;

	cv::minMaxLoc(XY.col(0), &minX, &maxX);
	cv::minMaxLoc(XY.col(1), &minY, &maxY);

	sx = (maxX - minX) / 2;
	sy = (maxY - minY) / 2;

	XY.col(0) = (XY.col(0) - mx) / sx;
	XY.col(1) = (XY.col(1) - my) / sy;

	//Forming the design matrix
	cv::Mat1d D = cv::Mat::ones(length, 6, CV_64FC1);

	D.col(0) = XY.col(0).mul(XY.col(0));
	D.col(1) = XY.col(0).mul(XY.col(1));
	D.col(2) = XY.col(1).mul(XY.col(1));
	D.col(3) = XY.col(0);
	D.col(4) = XY.col(1);

	XY.release();

	//Forming the Scatter Matrix
	cv::Mat1d S = D.t() * D;

	D.release();

	//Forming the conic constraint matrix
	cv::Mat1d C = cv::Mat::zeros(6, 6, CV_64FC1);
	C(0, 2) = -2; C(1, 1) = 1; C(2, 0) = -2;

	cv::Mat1d tmpA = S(cv::Rect(0, 0, 3, 3));
	cv::Mat1d tmpB = S(cv::Range(0, 3), cv::Range(3, 6));
	cv::Mat1d tmpC = S(cv::Range(3, 6), cv::Range(3, 6));
	cv::Mat1d tmpD = C(cv::Rect(0, 0, 3, 3));
	cv::Mat1d tmpE = tmpC.inv()*tmpB.t();

	C.release();
	S.release();

	//Finding the Eigen Value using SVD
	cv::Mat u, w, vt;
	cv::SVDecomp(tmpD.inv() * (tmpA - (tmpB*tmpE)), w, u, vt);

	tmpA.release(); tmpB.release(); tmpC.release();

	cv::Point minl, maxl;
	double maxVal, minVal;

	//cv::minMaxIdx(w, &minVal, &maxVal, &minIdx, &maxIdx);
	cv::minMaxLoc(w, &minVal, &maxVal, &minl, &maxl);

	cv::Mat1d A = cv::Mat::zeros(1, 6, CV_64FC1);

	vt.row(minl.y).copyTo(A(cv::Rect(0, 0, 3, 1)));
	//A(cv::Range(0, 1), cv::Range(3, 6)) = -tmpE * A(cv::Range(0, 1), cv::Range(0, 3)).t();
	tmpA = -tmpE * A(cv::Range(0, 1), cv::Range(0, 3)).t();
	tmpA = tmpA.t();
	tmpA.copyTo(A(cv::Range(0, 1), cv::Range(3, 6)));
	A = -A;


	tmpA.release();
	w.release();
	u.release();
	vt.release();

	cout << A << endl << endl;


	//Unnormalizing the data
	double A0 = A(0)*sy*sy;
	double A1 = A(1)*sx*sy;
	double A2 = A(2)*sx*sx;
	double A3 = -2 * A(0)*sy*sy*mx - A(1)*sx*sy*my + A(3)*sx*sy*sy;
	double A4 = -A(1)*sx*sy*mx - 2 * A(2)*sx*sx*my + A(4)*sx*sx*sy;
	double A5 = A(0)*sy*sy*mx*mx + A(1)*sx*sy*mx*my + A(2)*sx*sx*my*my
		- A(3)*sx*sy*sy*mx - A(4)*sx*sx*sy*my
		+ A(5)*sx*sx*sy*sy;


	A(0) = A0; A(1) = A1; A(2) = A2; A(3) = A3; A(4) = A4; A(5) = A5;

	cv::RotatedRect ell = conicToParametric(A);

	if (ell.size.width <= 0 || ell.size.height <= 0) return cv::RotatedRect();

	A.release();

	return ell;

}
