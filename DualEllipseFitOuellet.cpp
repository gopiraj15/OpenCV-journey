#include<iostream>
#include<core>
#include<vector>
#include<cmath>
#include<opencv2\opencv.hpp>

using namespace cv;
using namespace std;

const int Normalization = 1;

//This function is equivalent to the sign function from Matlab
template <typename T>
int sign(const T &val) { return (val > 0) - (val < 0); }

//This function calculates the sum of the members of the vector
double sum(vector<double> a)
{
	double s = 0;
	for (int i = 0; i < a.size(); i++)
	{
		s += a[i];
	}
	return s;
}

RotatedRect Mat2RotatedRect(Mat1f& Ell)
{
	RotatedRect el;
	el.angle = Ell(5, 0);
	el.center.x = Ell(0, 0);
	el.center.y = Ell(1, 0);
	el.size.width = Ell(2, 0) * 2;
	el.size.height = Ell(3, 0) * 2;
	return el;
}

Mat convertConicToParametric(Mat1f& par)
{
	Mat ell = Mat::zeros(5, 1, CV_64F);

	double thetarad = 0.5*atan2(par(1, 0), par(0, 0) - par(2, 0));
	double cost = cos(thetarad);
	double sint = sin(thetarad);
	double sin_squared = sint*sint;
	double cos_squared = cost*cost;
	double cos_sin = sint*cost;

	double Ao = par(5, 0);
	double Au = par(3, 0)*cost + par(4, 0)*sint;
	double Av = -par(3, 0)*sint + par(4, 0)*cost;
	double Auu = par(0, 0)*cos_squared + par(2, 0)*sin_squared + par(1, 0)*cos_sin;
	double Avv = par(0, 0)*sin_squared + par(2, 0)*cos_squared - par(1, 0)*cos_sin;

	double tuCentre = -Au / (2 * Auu);
	double tvCentre = -Av / (2.*Avv);
	double wCentre = Ao - Auu*tuCentre*tuCentre - Avv*tvCentre*tvCentre;

	double uCentre = tuCentre*cost - tvCentre*sint;
	double vCentre = tuCentre*sint + tvCentre*cost;

	double Ru = -wCentre / Auu;
	double Rv = -wCentre / Avv;

	Ru = sqrt(abs(Ru))*sign(Ru);
	Rv = sqrt(abs(Rv))*sign(Rv);


	double centrex = uCentre;
	double centrey = vCentre;
	double axea = Ru;
	double axeb = Rv;
	double angle = thetarad;

	ell.at<double>(0, 0) = centrex;
	ell.at<double>(1, 0) = centrey;
	ell.at<double>(2, 0) = axea;
	ell.at<double>(3, 0) = axeb;
	ell.at<double>(4, 0) = angle;

	return ell;
}


//This Function Fits the Ellipse to the contour points
//in Projective Geometry, lines are dual to the points
//here we are utilizing that property and we form lines based on the contour points and fit the ellipse
//the duals are related by an inverse operation, finally the result is inverted to obtain back the Ellispe parameters
//this function accepts the Ellipse points and gradients along the x and y axis as the input and returns the parametric matrix as output
Mat DualEllipseFit(vector<Point>& pts, Mat& dx, Mat& dy)
{
	Mat Ell = Mat::zeros(6, 1, CV_64F);

	Mat1f a = Mat::zeros(pts.size(), 1, CV_64F);
	Mat1f b = Mat::zeros(pts.size(), 1, CV_64F);
	Mat1f c = Mat::zeros(pts.size(), 1, CV_64F);
	//a = dx at the contour points 
	//b = dy at the contour points
	//c = -(dx*x + dy*y)
	
	double meanx = 0, meany = 0;

	for (int i = 0; i < pts.size(); i++)
	{
		a(i, 0) = dx.at<double>(pts[i]);
		b(i, 0) = dy.at<double>(pts[i]);
		c(i, 0) = -(dx.at<double>(pts[i])*pts[i].x + dy.at<double>(pts[i])*pts[i].y);
		
	}
	
	Mat1f H = Mat::zeros(3, 3, CV_64F);

	if (Normalization == 1)
	{
		Mat1f M = Mat::zeros(pts.size(), 2, CV_64F);
		Mat1f B = -c;
		for (int i = 0; i < pts.size(); i++)
		{
			M(i, 0) = -b(i, 0);
			M(i, 1) = a(i, 0);
		}
		//finding the pseudo inverse of the non-square matrix
		// pinv(A) = A' * (A * A')^-1
		Mat1f mpts = (M.t()*(M*M.t()).inv()) * B;
		
		H(0, 0) = 1; H(0, 2) = mpts(0, 0);
		H(1, 1) = 1; H(1, 2) = mpts(1, 0);
		H(2, 2) = 1;

		Mat1f Lnorm = Mat::zeros(pts.size(), 3, CV_64F);
		for (int i = 0; i < pts.size(); i++)
		{
			Lnorm(i, 0) = a(i, 0);
			Lnorm(i, 1) = b(i, 0);
			Lnorm(i, 2) = c(i, 0);
		}
		Lnorm = (H.t() * Lnorm.t()).t();

		for (int i = 0; i < pts.size(); i++)
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
	vector<double> a2, ab, b2, ac, bc, c2;
		
	for (int i = 0; i < pts.size(); i++)
	{
		a2.push_back(a(i, 0) * a(i, 0));
		ab.push_back(a(i, 0) * b(i, 0));
		b2.push_back(b(i, 0) * b(i, 0));
		ac.push_back(a(i, 0) * c(i, 0));
		bc.push_back(b(i, 0) * c(i, 0));
		c2.push_back(c(i, 0)* c(i, 0));
	}
	a.release();
	b.release();
	c.release();

	Mat1f AA = Mat::zeros(5, 5, CV_64F);
	Mat1f BB = Mat::zeros(5, 1, CV_64F);
	
	vector<double>a2_2, a2_ab, a2_b2, a2_ac, a2_bc;
	vector<double> ab_2, ab_b2, ab_ac, ab_bc;
	vector<double> b2_2, b2_ac, b2_bc;
	vector<double> ac_2, ac_bc;
	vector<double> bc_2;
		
	vector<double> c2_a2, c2_ab, c2_b2, c2_ac, c2_bc;
	for (int i = 0; i < pts.size(); i++)
	{
		a2_2.push_back(a2[i] * a2[i]);
		a2_ab.push_back(a2[i] * ab[i]);
		a2_b2.push_back(a2[i] * b2[i]);
		a2_ac.push_back(a2[i] * ac[i]);
		a2_bc.push_back(a2[i] * bc[i]);
		
		ab_2.push_back(ab[i] * ab[i]);
		ab_b2.push_back(ab[i] * b2[i]);
		ab_ac.push_back(ab[i] * ac[i]);
		ab_bc.push_back(ab[i] * bc[i]);
		
		b2_2.push_back(b2[i] * b2[i]);
		b2_ac.push_back(b2[i] * ac[i]);
		b2_bc.push_back(b2[i] * bc[i]);
		
		ac_2.push_back(ac[i] * ac[i]);
		ac_bc.push_back(ac[i] * bc[i]);
		
		bc_2.push_back(bc[i] * bc[i]);
		
		c2_a2.push_back(-c2[i] * a2[i]);
		c2_ab.push_back(-c2[i] * ab[i]);
		c2_b2.push_back(-c2[i] * b2[i]);
		c2_ac.push_back(-c2[i] * ac[i]);
		c2_bc.push_back(-c2[i] * bc[i]);
	}

	a2.clear();
	ab.clear();
	b2.clear();
	ac.clear();
	bc.clear();

	//A matrix
	AA(0, 0) = sum(a2_2);
	AA(0, 1) = AA(1, 0) = sum(a2_ab);
	AA(0, 2) = AA(2, 0) = sum(a2_b2);
	AA(0, 3) = AA(3, 0) = sum(a2_ac);
	AA(0, 4) = AA(4, 0) = sum(a2_bc);
				
	AA(1, 1) = sum(ab_2);
	AA(1, 2) = AA(2, 1) = sum(ab_b2);
	AA(1, 3) = AA(3, 1) = sum(ab_ac);
	AA(1, 4) = AA(4, 1) = sum(ab_bc);
		
	AA(2, 2) = sum(b2_2);
	AA(2, 3) = AA(3, 2) = sum(b2_ac);
	AA(2, 4) = AA(4, 2) = sum(b2_bc);
		
	AA(3, 3) = sum(ac_2);
	AA(3, 4) = AA(4, 3) = sum(ac_bc);
		
	AA(4, 4) = sum(bc_2);
	
	
	//B matrix
	BB(0, 0) = sum(c2_a2);
	BB(1, 0) = sum(c2_ab);
	BB(2, 0) = sum(c2_b2);
	BB(3, 0) = sum(c2_ac);
	BB(4, 0) = sum(c2_bc);


	//Solving the Least squares problem
	// X = A^-1 * B
	//
	//Mat1f sol = (AA.t() * (AA*AA.t()).inv()) * BB;

	Mat1f sol = AA.inv() * BB;

	AA.release();
	BB.release();
	a2_2.clear();
	a2_ab.clear();
	a2_ac.clear();
	a2_b2.clear();
	a2_bc.clear();
	ab_2.clear();
	ab_ac.clear();
	ab_b2.clear();
	ab_bc.clear();
	b2_2.clear();
	b2_ac.clear();
	b2_bc.clear();
	ac_2.clear();
	ac_bc.clear();
	bc_2.clear();
	c2_a2.clear();
	c2_ab.clear();
	c2_ac.clear();
	c2_b2.clear();
	c2_bc.clear();

	for (int i = 0; i < 5; i++)
		Ell.at<double>(i, 0) = sol(i, 0);

	Ell.at<double>(5, 0) = 1;

	Mat1f dCnorm = Mat::zeros(3, 3, CV_64F);
	dCnorm(0, 0) = Ell.at<double>(0, 0);
	dCnorm(0, 1) = dCnorm(1, 0) = Ell.at<double>(1, 0) / 2;
	dCnorm(0, 2) = dCnorm(2, 0) = Ell.at<double>(3, 0) / 2;
	dCnorm(1, 1) = Ell.at<double>(2, 0);
	dCnorm(1, 2) = dCnorm(2, 1) = Ell.at<double>(4, 0) / 2;
	dCnorm(2, 2) = Ell.at<double>(5, 0);

	Mat1f dC;

	if (Normalization == 1)   dC = H*dCnorm*H.t();
	else dC = dCnorm;

	dCnorm.release();
	H.release();

	//The DualEllispe is found by inverting the Ellipse found
	Mat1f C = dC.inv();
	C /= C(2, 2);

	dC.release();

	Ell.at<double>(0, 0) = C(0, 0);
	Ell.at<double>(1, 0) = C(1, 0) * 2;
	Ell.at<double>(2, 0) = C(1, 1);
	Ell.at<double>(3, 0) = C(2, 0) * 2;
	Ell.at<double>(4, 0) = C(1, 2) * 2;
	Ell.at<double>(5, 0) = C(2, 2);

	C.release();
	

	return Ell;
}

int main()
{
	Mat src = imread("D:/my library/test/circle.bmp", 0);
	if (!src.data) return 0;
	
	uchar *psrc = src.data;

	

	Mat1d d = Mat::zeros(1, 5, CV_64F);
	Mat1d g = Mat::zeros(1, 5, CV_64F);

	d(0, 0) = 0.2707,d(0,1) = 0.6065, d(0,2) =  0, d(0,3) =  -0.6065, d(0,4) = -0.2707;
	g(0, 0) = 0.1353, g(0, 1) = 0.6065, g(0, 2) = 1, g(0, 3) = 0.6065, g(0, 4) = 0.1353;

	Mat opx, opy;


	opx = g.t() * d;
	opy = d.t() * g;

	
	Mat dx, dy, grad;

	filter2D(src, dx, CV_64F, opx);
	filter2D(src, dy, CV_64F, opy);

	magnitude(dx, dy, grad);

	double meang = 0;

	for (int i = 3; i < src.rows - 3; i++)
	{
		for (int j = 3; j < src.cols - 3; j++)
		{
			meang += grad.at<double>(i, j);
		}
	}
	meang /= (src.rows*src.cols);

	Mat mask = grad > 3 * meang;

	Mat1i labels;

	int lab = connectedComponents(mask, labels);

	vector<vector<Point>> pts;
	vector<Point> tmp;
	//vector<double> dx, dy;
	for (int lbl = 1; lbl <= lab; lbl++)
	{
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (labels(i, j) == lbl) tmp.push_back(Point(i, j));
			}
		}
		if(tmp.size() > 0)pts.push_back(tmp);
		tmp.clear();
	}

	vector<Mat1f> Ells, EllParam;
	vector<RotatedRect> DualElls;

	for (int i = 0; i < pts.size(); i++)
	{
		Ells.push_back(DualEllipseFit(pts[0], dx, dy));
		EllParam.push_back(convertConicToParametric(Ells[i]));
		DualElls.push_back(Mat2RotatedRect(EllParam[i]));
	}

	//cout << Ell << endl;

	

	Mat res = src.clone();
	cvtColor(res, res, CV_GRAY2BGR);

	for (int i = 0; i < DualElls.size(); i++)
	{
		ellipse(res, DualElls[i], Scalar(0, 255, 0), 1);
	}


	namedWindow("Res", WINDOW_FREERATIO);
	imshow("Res", res);

	waitKey(0);
	//getchar();
	return 1;
}
