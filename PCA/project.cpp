////////
////////#include <iostream>
////////#include <algorithm>
////////#include<cstdlib>
////////#include <fstream>
////////#include <Eigen/Dense>
////////
////////using namespace std;
////////using namespace Eigen;
////////
////////
////////MatrixXd featurnormail(MatrixXd &X)
////////{
////////	////����ÿһά�ľ�ֵ
////////	//MatrixXd X1 = X.transpose();	
////////	//cout << "X1" << endl << X1 << endl;
////////	//MatrixXd X101 = X.transpose();
////////	//cout << "X.transpose();:" << endl << X101 << endl;
////////	//MatrixXd meanvalrow = X1.rowwise().mean();   ///������ƽ��
////////	//cout << "meanvalrow:" << endl << meanvalrow << endl;
////////	//MatrixXd meanvalcol = X1.colwise().mean();   ///������ƽ��
////////	//cout << "meanvalcol" << endl << meanvalcol << endl;
////////	////������ֵ��Ϊ0
////////	//RowVectorXd meanvecRow = meanvalrow;
////////	//X1.rowwise() -= meanvecRow;
////////	//cout << "X1.rowwise():" << endl << X1 << endl;
////////	//return X1.transpose();
////////	//����ÿһά�ľ�ֵ
////////	////MatrixXd X1 = X.transpose();
////////	////MatrixXd meanval = X1.colwise().mean(); 
////////	//////	cout << "meanval" << endl << meanval <<endl;	
////////	//////������ֵ��Ϊ0	
////////	////RowVectorXd meanvecRow = meanval;
////////	////X1.rowwise() -= meanvecRow;
////////	////return X1.transpose();
////////	MatrixXd X1 = X;
////////	//MatrixXd X1 = X.transpose();
////////	MatrixXd meanval = X1.colwise().mean();
////////	cout << "meanval" << endl << meanval << endl;
////////	//������ֵ��Ϊ0	
////////	RowVectorXd meanvecRow = meanval;
////////	X1.rowwise() -= meanvecRow;
////////	/*return X1.transpose();*/
////////	return X1.transpose();
////////}
////////
////////void ComComputeCov(MatrixXd &X, MatrixXd &C)
////////
////////{
////////	//����Э�������
////////	C = X*X.adjoint();//�൱��XT*X adjiont()�������� �൱��������ת��
////////	/*cout << "C:" << endl << C.transpose() << endl;*/
////////	C = C.array() / X.cols();//C.array()�����������ʽ
////////	cout << "X.cols():" << endl << X.cols() << endl;
////////	cout << "C:" << endl << C << endl;
////////}
////////
////////void ComputEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)
////////{
////////	//������������������ֵ SelfAdjointEigenSolver�Զ�������õ�����������������ֵ����
////////	SelfAdjointEigenSolver<MatrixXd> eig(C);
////////	vec = eig.eigenvectors();
////////	cout << "vec:"<< endl << vec << endl;
////////	val = eig.eigenvalues();
////////	cout << "val:" << endl << val << endl;
////////
////////}
////////
////////// ����ά��
////////int ComputDim(MatrixXd &val)
////////{
////////	int dim;
////////	double sum = 0;
////////	for (int i = val.rows() - 1; i >= 0; --i)
////////	{
////////		sum += val(i, 0);
////////		dim = i;
////////		if (sum / val.sum() >= 0.8)//�ﵽ����Ҫ��Ҫ��ʱ
////////			break;
////////	}
////////	cout << val.rows() - dim << endl;
////////	return val.rows() - dim;
////////}
////////
////////int main()
////////{
////////	//�ļ�����
////////	ifstream fin("F:\\PCLprincipal\\pca070808\\Project1\\project01.txt");
////////	if (fin.is_open())
////////	{
////////		cout << "����ɹ�" << endl;
////////	}
////////	ofstream fout("outpot1.txt");
////////	ofstream fout1("X1.txt");
////////
////////	//��������Ҫ����
////////	const int m = 5, n = 2, z = 2;
////////	MatrixXd X(m, n), C(z, z);
////////	MatrixXd vec, val;
////////
////////	//��ȡ����
////////	double in[z];
////////	for (int i = 0; i < m; ++i)
////////	{
////////		for (int j = 0; j < n; ++j)
////////		{
////////
////////			fin >> in[j];
////////			//cout << in[j] << endl;
////////			//cout << "in[" << j << "]" << in[j] << endl;
////////		}
////////		for (int j = 1; j <= n; ++j)
////////			X(i, j - 1) = in[j - 1];
////////	}
////////
////////	cout << endl;
////////	cout << "XΪԭʼ�����ݣ�" << endl << X << endl;
////////	cout << endl;
////////
////////
////////	// 1 ��ֵ0��׼��
////////	MatrixXd X1 = featurnormail(X);
////////	cout << "X1��ֵ0��׼�������ݣ�" << endl << X1.transpose() << endl;   ///.transpose()����Ϊ�˲鿴 N*3
////////	cout << endl;
////////
////////	// 2 ����Э����
////////	ComComputeCov(X1, C);
////////
////////	// 3 ��������ֵ����������
////////	ComputEig(C, vec, val);
////////
////////	// 4 ������ʧ�ʣ�ȷ�����͵�ά��
////////	int dim = ComputDim(val);
////////
////////	// 5������
////////	MatrixXd res001 = vec.transpose().rightCols(dim).transpose();
////////	cout << "res001:" << endl << res001 << endl;  //2*3
////////	MatrixXd res002 = vec.transpose().leftCols(1).transpose();
////////	cout << "res002:" << endl << res002 << endl;  //2*3
////////
////////
////////	MatrixXd res = res001*X1; //2*3  *  N*3
////////	cout << "res:" << endl << res.transpose() << endl;
////////	MatrixXd res2 = res002*X1; //2*3  *  N*3
////////	cout << "res2:" << endl << res2.transpose() << endl;
////////
////////	cout << "resΪ��PCA�㷨֮������ݣ�" << endl << res.transpose() << endl;
////////	cout << endl;
////////
////////	// 6������
////////	fout << "the result is" << res.rows() << "x" << res.cols() << "after pca algorithm" << endl;
////////	fout << res.transpose();
////////	fout.close();
////////	cout << "I Love n(Yaner)......" << endl;
////////
////////
////////	system("pause");
////////	return 0;
////////}
////
////
////
////#include <iostream>
////#include <algorithm>
////#include<cstdlib>
////#include <fstream>
////#include <Eigen/Dense>
////
////using namespace std;
////using namespace Eigen;
////
////
////MatrixXd featurnormail(MatrixXd &X)
////{
////	////����ÿһά�ľ�ֵ
////	//MatrixXd X1 = X.transpose();	
////	//cout << "X1" << endl << X1 << endl;
////	//MatrixXd X101 = X.transpose();
////	//cout << "X.transpose();:" << endl << X101 << endl;
////	//MatrixXd meanvalrow = X1.rowwise().mean();   ///������ƽ��
////	//cout << "meanvalrow:" << endl << meanvalrow << endl;
////	//MatrixXd meanvalcol = X1.colwise().mean();   ///������ƽ��
////	//cout << "meanvalcol" << endl << meanvalcol << endl;
////	////������ֵ��Ϊ0
////	//RowVectorXd meanvecRow = meanvalrow;
////	//X1.rowwise() -= meanvecRow;
////	//cout << "X1.rowwise():" << endl << X1 << endl;
////	//return X1.transpose();
////	//����ÿһά�ľ�ֵ
////	////MatrixXd X1 = X.transpose();
////	////MatrixXd meanval = X1.colwise().mean(); 
////	//////	cout << "meanval" << endl << meanval <<endl;	
////	//////������ֵ��Ϊ0	
////	////RowVectorXd meanvecRow = meanval;
////	////X1.rowwise() -= meanvecRow;
////	////return X1.transpose();
////	MatrixXd X1 = X;
////	//MatrixXd X1 = X.transpose();
////	MatrixXd meanval = X1.colwise().mean();
////	cout << "meanval" << endl << meanval << endl;
////	//������ֵ��Ϊ0	
////	RowVectorXd meanvecRow = meanval;
////	X1.rowwise() -= meanvecRow;
////	/*return X1.transpose();*/
////	return X1.transpose();
////}
////
////void ComComputeCov(MatrixXd &X, MatrixXd &C)
////
////{
////	//����Э�������
////	C = X*X.adjoint();//�൱��XT*X adjiont()�������� �൱��������ת��
////	/*cout << "C:" << endl << C.transpose() << endl;*/
////	C = C.array() / X.cols();//C.array()�����������ʽ
////	cout << "X.cols():" << endl << X.cols() << endl;
////	cout << "C:" << endl << C << endl;
////}
////
////void ComputEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)
////{
////	//������������������ֵ SelfAdjointEigenSolver�Զ�������õ�����������������ֵ����
////	SelfAdjointEigenSolver<MatrixXd> eig(C);
////	vec = eig.eigenvectors();
////	cout << "vec:" << endl << vec << endl;
////	val = eig.eigenvalues();
////	cout << "val:" << endl << val << endl;
////
////}
////
////// ����ά��
////int ComputDim(MatrixXd &val)
////{
////	int dim;
////	double sum = 0;
////	for (int i = val.rows() - 1; i >= 0; --i)
////	{
////		sum += val(i, 0);
////		dim = i;
////		if (sum / val.sum() >= 0.8)//�ﵽ����Ҫ��Ҫ��ʱ
////			break;
////	}
////	cout << val.rows() - dim << endl;
////	return val.rows() - dim;
////}
////
////int main()
////{
////	//�ļ�����
////	ifstream fin("F:\\PCLprincipal\\pca070808\\Project1\\testData.txt");
////	if (fin.is_open())
////	{
////		cout << "����ɹ�" << endl;
////	}
////	ofstream fout("X1X2outpot1.txt");
////	ofstream fout1("X1X2X1.txt");
////
////	//��������Ҫ����
////	const int m = 10, n = 2, z = 2;
////	MatrixXd X(m, n), C(z, z);
////	MatrixXd vec, val;
////
////	//��ȡ����
////	double in[z];
////	for (int i = 0; i < m; ++i)
////	{
////		for (int j = 0; j < n; ++j)
////		{
////
////			fin >> in[j];
////			//cout << in[j] << endl;
////			//cout << "in[" << j << "]" << in[j] << endl;
////		}
////		for (int j = 1; j <= n; ++j)
////			X(i, j - 1) = in[j - 1];
////	}
////
////	cout << endl;
////	cout << "XΪԭʼ�����ݣ�" << endl << X << endl;
////	cout << endl;
////
////
////	// 1 ��ֵ0��׼��
////	MatrixXd X1 = featurnormail(X);
////	cout << "X1��ֵ0��׼�������ݣ�" << endl << X1.transpose() << endl;   ///.transpose()����Ϊ�˲鿴 N*3
////	cout << endl;
////
////	// 2 ����Э����
////	ComComputeCov(X1, C);
////
////	// 3 ��������ֵ����������
////	ComputEig(C, vec, val);
////
////	// 4 ������ʧ�ʣ�ȷ�����͵�ά��
////	int dim = ComputDim(val);
////	//int dim = 1;
////	// 5������
////	
////	//MatrixXd res001 = vec.transpose().rightCols(dim).transpose();
////	MatrixXd res001 = vec.bottomRows(dim);
////	//MatrixXd res001 = vec.transpose().leftCols(dim).transpose();
////	cout << "res001:" << endl << res001 << endl;  //2*3
////	MatrixXd res002 = vec.transpose().leftCols(1).transpose();
////	cout << "res002:" << endl << res002 << endl;  //2*3
////
////
////	MatrixXd res = res001*X1; //2*3  *  N*3
////	cout << "res:" << endl << res.transpose() << endl;
////	MatrixXd res2 = res002*X1; //2*3  *  N*3
////	cout << "res2:" << endl << res2.transpose() << endl;
////
////	cout << "resΪ��PCA�㷨֮������ݣ�" << endl << res.transpose() << endl;
////	cout << endl;
////
////	// 6������
////	fout << "the result is" << res.rows() << "x" << res.cols() << "after pca algorithm" << endl;
////	fout << res.transpose();
////	fout.close();
////	cout << "I Love n(Yaner)......" << endl;
////
////
////	system("pause");
////	return 0;
////}
//
//
////
////#include <iostream>
////#include <algorithm>
////#include<cstdlib>
////#include <fstream>
////#include <Eigen/Dense>
////
////using namespace std;
////using namespace Eigen;
////
////
////MatrixXd featurnormail(MatrixXd &X)
////{
////	////����ÿһά�ľ�ֵ
////	//MatrixXd X1 = X.transpose();	
////	//cout << "X1" << endl << X1 << endl;
////	//MatrixXd X101 = X.transpose();
////	//cout << "X.transpose();:" << endl << X101 << endl;
////	//MatrixXd meanvalrow = X1.rowwise().mean();   ///������ƽ��
////	//cout << "meanvalrow:" << endl << meanvalrow << endl;
////	//MatrixXd meanvalcol = X1.colwise().mean();   ///������ƽ��
////	//cout << "meanvalcol" << endl << meanvalcol << endl;
////	////������ֵ��Ϊ0
////	//RowVectorXd meanvecRow = meanvalrow;
////	//X1.rowwise() -= meanvecRow;
////	//cout << "X1.rowwise():" << endl << X1 << endl;
////	//return X1.transpose();
////	//����ÿһά�ľ�ֵ
////	////MatrixXd X1 = X.transpose();
////	////MatrixXd meanval = X1.colwise().mean(); 
////	//////	cout << "meanval" << endl << meanval <<endl;	
////	//////������ֵ��Ϊ0	
////	////RowVectorXd meanvecRow = meanval;
////	////X1.rowwise() -= meanvecRow;
////	////return X1.transpose();
////	MatrixXd X1 = X;
////	//MatrixXd X1 = X.transpose();
////	MatrixXd meanval = X1.colwise().mean();
////	cout << "meanval" << endl << meanval << endl;
////	//������ֵ��Ϊ0	
////	RowVectorXd meanvecRow = meanval;
////	X1.rowwise() -= meanvecRow;
////	/*return X1.transpose();*/
////	return X1.transpose();
////}
////
////void ComComputeCov(MatrixXd &X, MatrixXd &C)
////
////{
////	//����Э�������
////	C = X*X.adjoint();//�൱��XT*X adjiont()�������� �൱��������ת��
////	/*cout << "C:" << endl << C.transpose() << endl;*/
////	C = C.array() / X.cols();//C.array()�����������ʽ
////	cout << "X.cols():" << endl << X.cols() << endl;
////	cout << "C:" << endl << C << endl;
////}
////
////void ComputEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)
////{
////	//������������������ֵ SelfAdjointEigenSolver�Զ�������õ�����������������ֵ����
////	SelfAdjointEigenSolver<MatrixXd> eig(C);
////	vec = eig.eigenvectors();
////	cout << "vec:"<< endl << vec << endl;
////	val = eig.eigenvalues();
////	cout << "val:" << endl << val << endl;
////
////}
////
////// ����ά��
////int ComputDim(MatrixXd &val)
////{
////	int dim;
////	double sum = 0;
////	for (int i = val.rows() - 1; i >= 0; --i)
////	{
////		sum += val(i, 0);
////		dim = i;
////		if (sum / val.sum() >= 0.8)//�ﵽ����Ҫ��Ҫ��ʱ
////			break;
////	}
////	cout << val.rows() - dim << endl;
////	return val.rows() - dim;
////}
////
////int main()
////{
////	//�ļ�����
////	ifstream fin("F:\\PCLprincipal\\pca070808\\Project1\\project01.txt");
////	if (fin.is_open())
////	{
////		cout << "����ɹ�" << endl;
////	}
////	ofstream fout("outpot1.txt");
////	ofstream fout1("X1.txt");
////
////	//��������Ҫ����
////	const int m = 5, n = 2, z = 2;
////	MatrixXd X(m, n), C(z, z);
////	MatrixXd vec, val;
////
////	//��ȡ����
////	double in[z];
////	for (int i = 0; i < m; ++i)
////	{
////		for (int j = 0; j < n; ++j)
////		{
////
////			fin >> in[j];
////			//cout << in[j] << endl;
////			//cout << "in[" << j << "]" << in[j] << endl;
////		}
////		for (int j = 1; j <= n; ++j)
////			X(i, j - 1) = in[j - 1];
////	}
////
////	cout << endl;
////	cout << "XΪԭʼ�����ݣ�" << endl << X << endl;
////	cout << endl;
////
////
////	// 1 ��ֵ0��׼��
////	MatrixXd X1 = featurnormail(X);
////	cout << "X1��ֵ0��׼�������ݣ�" << endl << X1.transpose() << endl;   ///.transpose()����Ϊ�˲鿴 N*3
////	cout << endl;
////
////	// 2 ����Э����
////	ComComputeCov(X1, C);
////
////	// 3 ��������ֵ����������
////	ComputEig(C, vec, val);
////
////	// 4 ������ʧ�ʣ�ȷ�����͵�ά��
////	int dim = ComputDim(val);
////
////	// 5������
////	MatrixXd res001 = vec.transpose().rightCols(dim).transpose();
////	cout << "res001:" << endl << res001 << endl;  //2*3
////	MatrixXd res002 = vec.transpose().leftCols(1).transpose();
////	cout << "res002:" << endl << res002 << endl;  //2*3
////
////
////	MatrixXd res = res001*X1; //2*3  *  N*3
////	cout << "res:" << endl << res.transpose() << endl;
////	MatrixXd res2 = res002*X1; //2*3  *  N*3
////	cout << "res2:" << endl << res2.transpose() << endl;
////
////	cout << "resΪ��PCA�㷨֮������ݣ�" << endl << res.transpose() << endl;
////	cout << endl;
////
////	// 6������
////	fout << "the result is" << res.rows() << "x" << res.cols() << "after pca algorithm" << endl;
////	fout << res.transpose();
////	fout.close();
////	cout << "I Love n(Yaner)......" << endl;
////
////
////	system("pause");
////	return 0;
////}


#include <iostream>
#include <algorithm>
#include<cstdlib>
#include <fstream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXd featurnormal(MatrixXd &X)
{
	// Eigen:��������
	// VectorXd    �ȼ���Matrix<double,Dynamic,1>;       //��������һ��
	// RowVectorXd �ȼ���Matrix<double,1,Dynamic>;       //��������һ��
	// MatrixXd    �ȼ���Matrix<double,Dynamic,Dynamic>;

	//����ÿһά�ľ�ֵ
	MatrixXd X1 = X.transpose();  //��������ͨת�á����б任��

	VectorXd meanval = X1.rowwise().mean();//colwise�������ֵ��//rowwise�������ֵ�� //X1.rowwise()������
	cout << "meanval" << endl << meanval << endl;
	
	//������ֵ��Ϊ0	
	X1.colwise() -= meanval;   //���м���ÿһ�����ֵ����
	
	return X1;
}

void ComputeCov(MatrixXd &X1, MatrixXd &C)

{
	//����Э���������������m��nά���ݼ�¼�����䰴���ų�n��m�ľ���X����C =X*XT/m
    //�˴�����m
		
	C = X1*X1.adjoint();//�൱��ԭʼ����XT*X    adjiont()�������� �൱��������ת��
	/*cout << "C:" << endl << C<< endl;*/
	C = C.array() / X1.cols();//C.array()�����������ʽ  ///��Ԫ�س���m

	cout << "C:" << endl << C << endl;
}

void ComputEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)
{
	//������������������ֵ SelfAdjointEigenSolver�Զ�������õ�����������������ֵ����
	SelfAdjointEigenSolver<MatrixXd> eig(C);
	//���������������ݶ�Ӧ������ֵ��С�������С�
	vec = eig.eigenvectors();
	val = eig.eigenvalues();
	
	//��Ӧ��������ת������������������ֵ�Ӵ�С���С������¡�
	vec.colwise().reverse();
	cout << "***************************************" << endl;
	cout << "vec.colwise().reverse():" << endl << vec.colwise().reverse() << endl;
	cout << "***************************************" << endl;
	val.colwise().reverse();
	cout << "val.colwise().reverse():" << endl << val.colwise().reverse() << endl;
	cout << "***************************************" << endl;
}

// ����ά��
int ComputDim(MatrixXd &val)
{
	int dim;
	double sum = 0;
	for (int i = val.rows() - 1; i >= 0; --i)
	{
		sum += val(i, 0);
		dim = i;
		if (sum / val.sum() >= 0.8)//�ﵽ����Ҫ��Ҫ��ʱ
			break;
	}
	//cout << val.rows() - dim << endl;
	return val.rows() - dim;
}

int main()
{
	//�ļ�����
	ifstream fin("F:\\The_Algorithms\\PCA\\m5n2.txt");
	if (fin.is_open())
	{
		cout << "����ɹ�" << endl;
	}
	ofstream fout("X1X2outpot1.txt");

	//��������Ҫ����
	const int m = 5, n = 2, z = 2;
	MatrixXd X(m, n), C(z, z);  //CΪЭ������
	MatrixXd vec, val;

	//��ȡ����
	double in[z];
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			fin >> in[j];  //��ֵ����
		}
		for (int j = 1; j <= n; ++j)
			X(i, j - 1) = in[j - 1];
	}

	cout << endl;
	cout << "XΪԭʼ�����ݣ�" << endl << X << endl;
	cout << endl;
	
	
	// 1 ��ֵ0��׼��
	MatrixXd X1 = featurnormal(X);
	cout << "X1��ֵ0��׼�������ݣ�" << endl << X1.transpose() << endl;   ///.transpose()����Ϊ�˲鿴 N*3
	cout << endl;

	// 2 ����Э����
	ComputeCov(X1, C);

	// 3 ��������ֵ����������
	ComputEig(C, vec, val);

	// 4 ������ʧ�ʣ�ȷ�����͵�ά��
	int dim = ComputDim(val);
	//int dim = 1;
	// 5������

	//��������
	//MatrixXd res001 = vec.transpose().rightCols(dim).transpose();
	//MatrixXd res001 = vec.bottomRows(dim);
	//MatrixXd res001 = vec.transpose().leftCols(dim).transpose();
	/*cout <<"X.colwise().reverse()"<<endl<< X.colwise().reverse() << endl;*/ //��ֱ��ת������ĺܣ�

	MatrixXd res001 = vec.colwise().reverse().topRows(dim);  
	//���з�תȡǰdim�� ����������ʵ����������ֵ�Ӵ�С���л�ȡ����������
	
	cout << "res001:" << endl << res001 << endl;  //2*3
	MatrixXd res = res001*X1; //2*3  *  N*3  Y=PX ��Y��Ϊ��ά֮��Ľ����

	//����鿴
	cout << "resΪ��PCA�㷨֮������ݣ�" << endl << res.transpose() << endl;
	
	// 6������
	fout << "the result is" << res.rows() << "x" << res.cols() << "after pca algorithm" << endl;
	fout << res.transpose();
	fout.close();
	
	system("pause");
	return 0;
}

