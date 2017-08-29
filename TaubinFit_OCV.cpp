#include<opencv2\opencv.hpp>

//This Function is similar to Matlab's sign function
template <typename T>
int sign(const T &val) { return (val > 0) - (val < 0); }

//This function converts the conic ([A B C D E F]) to parametric form ([xc yc a b theta])
//The ellipse parameters are returned in the form of OpenCV RotatedRect
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
	double tuCentre = -Au / (2*Auu);
	double tvCentre = -Av / (2*Avv);
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
	res.size.width =  2 * Ru;
	res.size.height = 2 * Rv;
	res.angle = thetarad * 180 / CV_PI;

	return res;
}

//This function fits an ellipse to the given input of points
template<typename T> 
cv::RotatedRect EllipseFitByTaubin(const std::vector<cv::Point_<T>>& pts)
{
	if (pts.size() <= 5) return cv::RotatedRect();

	int length = pts.size();

	cv::Mat XY = cv::Mat(pts).reshape(1);

	//Calculating the centroid of the data
	cv::Scalar mx = mean(XY.col(0));
	cv::Scalar my = mean(XY.col(1));

	cv::Mat1d Z = cv::Mat::ones(length, 6, CV_64FC1);

	Z.col(0) = (XY.col(0) - mx[0]).mul(XY.col(0) - mx[0]); // X^2
	Z.col(1) = (XY.col(0) - mx[0]).mul(XY.col(1) - my[0]); // XY
	Z.col(2) = (XY.col(1) - my[0]).mul(XY.col(1) - my[0]); // Y^2
	Z.col(3) = XY.col(0) - mx[0]; // X
	Z.col(4) = XY.col(1) - my[0]; // Y
	
	XY.release();

	cv::Mat1d M = (Z.t()*Z) / length;

	Z.release();

	cv::Mat1d  P = M(cv::Rect(0, 0, 5, 5));

	P(0, 0) = (M(0, 0) - M(0, 5)*M(0, 5));
	P(0, 1) = P(1, 0) = M(0, 1) - M(0, 5)*M(1, 5);
	P(0, 2) = P(2, 0) = M(0, 2) - M(0, 5)*M(2, 5);
	P(1, 1) = (M(1, 1) - M(1, 5)*M(1, 5));
	P(1, 2) = P(2, 1) = M(1, 2) - M(1, 5)*M(2, 5);
	P(2, 2) = (M(2, 2) - M(2, 5) * M(2, 5));

	cv::Mat1d Q = cv::Mat::eye(5, 5, CV_64FC1);

	Q(0, 0) = 4 * M(0, 5);
	Q(0, 1) = Q(1, 0) = 2 * M(1, 5);
	Q(1, 1) = M(0, 5) + M(2, 5);
	Q(1, 2) = Q(2, 1) = 2 * M(1, 5);
	Q(2, 2) = 4 * M(2, 5);

	if (cv::determinant(Q) < 0.000000000000000001) return cv::RotatedRect();

	//Solving the generalized eigen value Problem
	//equivalent to [V, D] = eig(P,Q) 
	// [u, s, v] = svd(inv(Q)*P)
	// s ~= D
	// V ~= v
	cv::Mat1d u, w, vt;
	cv::SVDecomp(Q.inv()*P, w, u, vt);

	Q.release();
	P.release();

	
	cv::Point minl, maxl;
	double maxVal, minVal;

	cv::minMaxLoc(w, &minVal, &maxVal, &minl, &maxl);

	cv::Mat1d A = cv::Mat::zeros(1, 6, CV_64FC1);

	vt.row(minl.y).copyTo(A(cv::Rect(0, 0, 5, 1)));

	w.release();
	u.release();
	vt.release();

	A(cv::Rect(5, 0, 1, 1)) = -A(cv::Rect(0, 0, 3, 1)) * M.col(5)(cv::Rect(0, 0, 1, 3));

	M.release();

	double A4 = A(3) - (2*A(0)*mx[0]) - A(1)*my[0];
	double A5 = A(4) - (2*A(2)*my[0]) - A(1)*mx[0];
	double A6 = A(5) + A(0)*mx[0]*mx[0] + A(2)*my[0]*my[0] + A(1)*mx[0]*my[0] - A(3)*mx[0] - A(4)*my[0];

	A(3) = A4; A(4) = A5; A(5) = A6;

	A /= norm(A);

	cv::RotatedRect ell = conicToParametric(A);

	if (ell.size.width <= 0 || ell.size.height <= 0) return cv::RotatedRect();

	A.release();

	return ell;
}
