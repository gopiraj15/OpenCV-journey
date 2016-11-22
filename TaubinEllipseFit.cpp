#include<iostream>
#include<cmath>
#include<vector>
#include<Eigen/core>
#include<Eigen/Eigenvalues>
#include<ctime>


using namespace std;
using namespace Eigen;

//This structure is used to store a point
struct Point2d
{
	double x, y;//the cooridinates of the point
	//A constructor to store the point
	Point2d(double nx, double ny) :x(nx), y(ny){}
};

//This function is equivalent to the sign function from Matlab
template <typename T>
int sign(const T &val) { return (val > 0) - (val < 0); }

template <typename T>
std::vector<int> sign(const std::vector<T> &v) {
	std::vector<int> r(v.size());
	std::transform(v.begin(), v.end(), r.begin(), (int(*)(const T&))sign);
	return r;
}


//This function converts The Conic in the form [A B C D E F] 
//into an Ellipse of the form [centrex centrey axea axeb angle]
MatrixXd convertConicToParametric(const MatrixXd& par)
{
	MatrixXd ell = MatrixXd::Constant(5, 1, 0);

	double thetarad = 0.5*atan2(par(1,0), par(0,0) - par(2,0));
	double cost = cos(thetarad);
	double sint = sin(thetarad);
	double sin_squared = sint*sint;
	double cos_squared = cost*cost;
	double cos_sin = sint*cost;

	double Ao = par(5,0);
	double Au = par(3,0)*cost + par(4,0)*sint;
	double Av = -par(3,0)*sint + par(4,0)*cost;
	double Auu = par(0,0)*cos_squared + par(2,0)*sin_squared + par(1,0)*cos_sin;
	double Avv = par(0, 0)*sin_squared + par(2, 0)*cos_squared - par(1, 0)*cos_sin;

	double tuCentre = -Au / (2*Auu);
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

	ell(0, 0) = centrex;
	ell(1, 0) = centrey;
	ell(2, 0) = axea;
	ell(3, 0) = axeb;
	ell(4, 0) = angle;

	return ell;
}

///This function fits an Ellipse to the given set of points
//The resulting Conic may not be Ellipse always
MatrixXd EllipseFitbyTaubin(const vector<Point2d>& pts)
{
	MatrixXd A = MatrixXd::Constant(6, 1, 0);

	MatrixXd Xm = Eigen::MatrixXd::Constant(pts.size(), 1, 0);
	MatrixXd Ym = Eigen::MatrixXd::Constant(pts.size(), 1, 0);

	for (int i = 0; i < pts.size(); i++)
	{
		Xm(i, 0) = pts[i].x;
		Ym(i, 0) = pts[i].y;
	}

	double meanx = 0, meany = 0;
	for (int i = 0; i < pts.size(); i++)
	{
		meanx += Xm(i, 0);
		meany += Ym(i, 0);
	}
	meanx /= pts.size();
	meany /= pts.size(); 
	
	MatrixXd Zm = Eigen::MatrixXd::Constant(pts.size(), 6, 0);

	for (int i = 0; i < pts.size(); i++)
	{
		Zm(i, 0) = pow(Xm(i, 0) - meanx, 2);
		Zm(i, 1) = pow((Xm(i, 0) - meanx) * (Ym(i, 0) - meany), 1);
		Zm(i, 2) = pow(Ym(i, 0) - meany, 2);
		Zm(i, 3) = Xm(i, 0) - meanx;
		Zm(i, 4) = Ym(i, 0) - meany;
		Zm(i, 5) = 1;
	}
	MatrixXd Mm = Eigen::MatrixXd::Constant(6, 6, 0);

	Mm = (Zm.transpose() * Zm) / pts.size();

	MatrixXd Pm = Eigen::MatrixXd::Constant(5, 5, 0), Qm = Eigen::MatrixXd::Constant(5, 5, 0);

	Pm(0, 0) = Mm(0, 0) - Mm(0, 5)*Mm(0, 5);
	Pm(0, 1) = Mm(0, 1) - Mm(0, 5)*Mm(1, 5);
	Pm(0, 2) = Mm(0, 2) - Mm(0, 5)*Mm(2, 5);
	Pm(0, 3) = Mm(0, 3);
	Pm(0, 4) = Mm(0, 4);

	Pm(1, 0) = Mm(0, 1) - Mm(0, 5)*Mm(1, 5);
	Pm(1, 1) = Mm(1, 1) - Mm(1, 5)*Mm(1, 5);
	Pm(1, 2) = Mm(1, 2) - Mm(1, 5)*Mm(2, 5);
	Pm(1, 3) = Mm(1, 3);
	Pm(1, 4) = Mm(1, 4);

	Pm(2, 0) = Mm(0, 2) - Mm(0, 5)*Mm(2, 5);
	Pm(2, 1) = Mm(1, 2) - Mm(1, 5)*Mm(2, 5);
	Pm(2, 2) = Mm(2, 2) - Mm(2, 5)*Mm(2, 5);
	Pm(2, 3) = Mm(2, 3);
	Pm(2, 4) = Mm(2, 4);

	Pm(3, 0) = Mm(0, 3);
	Pm(3, 1) = Mm(1, 3);
	Pm(3, 2) = Mm(2, 3);
	Pm(3, 3) = Mm(3, 3);
	Pm(3, 4) = Mm(3, 4);

	Pm(4, 0) = Mm(0, 4);
	Pm(4, 1) = Mm(1, 4);
	Pm(4, 2) = Mm(2, 4);
	Pm(4, 3) = Mm(3, 4);
	Pm(4, 4) = Mm(4, 4);


	Qm(0, 0) = 4 * Mm(0, 5); Qm(0, 1) = 2 * Mm(1, 5);
	Qm(1, 0) = 2 * Mm(1, 5); Qm(1, 1) = Mm(0, 5) + Mm(2, 5); Qm(1, 2) = 2 * Mm(1, 5);
	Qm(2, 1) = 2 * Mm(1, 5); Qm(2, 2) = 4 * Mm(2, 5);
	Qm(3, 3) = 1;
	Qm(4, 4) = 1;

	//Generalized Eigen value problem solver from the Eigen library
	GeneralizedSelfAdjointEigenSolver<MatrixXd> EigSolver(Pm, Qm);

	EigSolver.compute(Pm, Qm);

	for (int i = 0; i < 5; i++)
	{
		A(i, 0) = EigSolver.eigenvectors()(i,0);
	}


	MatrixXd A13 = MatrixXd::Constant(3, 1, 0);
	MatrixXd M = MatrixXd::Constant(3, 1, 0);

	for (int i = 0; i < 3; i++)
	{
		A13(i, 0) = A(i, 0);
		M(i, 0) = Mm(5, i);
	}

	MatrixXd tmp = -A13.transpose() * M;

	A(5, 0) = tmp(0, 0);


	double A4 = A(3, 0) - 2 * A(0, 0)*meanx - A(1, 0)*meany;
	double A5 = A(4, 0) - 2 * A(2, 0)*meany - A(1, 0)*meanx;
	double A6 = A(5, 0) + A(0, 0)*pow(meanx, 2) + A(2, 0)*pow(meany, 2) + A(1, 0)*meanx*meany - A(3, 0)*meanx - A(4, 0)*meany;

	A(3,0) = A4;  A(4,0) = A5;  A(5,0) = A6;

	JacobiSVD<MatrixXd> svd(A);

	//The largest singular value is given as the sqrt of largest Eigen Value of the Symmetric matrix A.t() * A
	//  ||A|| = sqrt(Lambda_max (A.t()*A) )
	double normA = svd.singularValues()[0];

	A /= (-normA);

	return A;
}

int main()
{
	int Rx = 300, Ry = 200, Cx = 250, Cy = 150;
	double Rotation = 0.4;
	double val = 0;
	vector<double> t;

	t.push_back(0);
	for (int i = 0; i < 100; i++)
	{
		t.push_back(val+=0.1);
	}

	vector<Point2d> ellPoints;

	vector<double>nx, ny, x, y;
	
	for (int i = 0; i < t.size(); i++)
	{
		x.push_back(Rx * cos(t[i]));
		y.push_back(Ry * sin(t[i]));
	}

	for (int i = 0; i < t.size(); i++)
	{
		nx.push_back(x[i]*cos(Rotation) - y[i]*sin(Rotation) + Cx);
		ny.push_back(x[i] * sin(Rotation) + y[i] * cos(Rotation) + Cy);
	}

	for (int i = 0; i < t.size(); i++)
	{
		ellPoints.push_back(Point2d(nx[i], ny[i]));
	}

	clock_t start = clock();

	MatrixXd A = EllipseFitbyTaubin(ellPoints);
	MatrixXd ell = convertConicToParametric(A);

	cout << "Ellipse Parameters: " << endl << ell << endl << endl;

	cout << "Time Taken: " << (double)(clock() - start) << "ms" << endl;

	getchar();
	return 0;
}
