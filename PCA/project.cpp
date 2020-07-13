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
////////	////计算每一维的均值
////////	//MatrixXd X1 = X.transpose();	
////////	//cout << "X1" << endl << X1 << endl;
////////	//MatrixXd X101 = X.transpose();
////////	//cout << "X.transpose();:" << endl << X101 << endl;
////////	//MatrixXd meanvalrow = X1.rowwise().mean();   ///按行求平均
////////	//cout << "meanvalrow:" << endl << meanvalrow << endl;
////////	//MatrixXd meanvalcol = X1.colwise().mean();   ///按列求平均
////////	//cout << "meanvalcol" << endl << meanvalcol << endl;
////////	////样本均值化为0
////////	//RowVectorXd meanvecRow = meanvalrow;
////////	//X1.rowwise() -= meanvecRow;
////////	//cout << "X1.rowwise():" << endl << X1 << endl;
////////	//return X1.transpose();
////////	//计算每一维的均值
////////	////MatrixXd X1 = X.transpose();
////////	////MatrixXd meanval = X1.colwise().mean(); 
////////	//////	cout << "meanval" << endl << meanval <<endl;	
////////	//////样本均值化为0	
////////	////RowVectorXd meanvecRow = meanval;
////////	////X1.rowwise() -= meanvecRow;
////////	////return X1.transpose();
////////	MatrixXd X1 = X;
////////	//MatrixXd X1 = X.transpose();
////////	MatrixXd meanval = X1.colwise().mean();
////////	cout << "meanval" << endl << meanval << endl;
////////	//样本均值化为0	
////////	RowVectorXd meanvecRow = meanval;
////////	X1.rowwise() -= meanvecRow;
////////	/*return X1.transpose();*/
////////	return X1.transpose();
////////}
////////
////////void ComComputeCov(MatrixXd &X, MatrixXd &C)
////////
////////{
////////	//计算协方差矩阵
////////	C = X*X.adjoint();//相当于XT*X adjiont()求伴随矩阵 相当于求矩阵的转置
////////	/*cout << "C:" << endl << C.transpose() << endl;*/
////////	C = C.array() / X.cols();//C.array()矩阵的数组形式
////////	cout << "X.cols():" << endl << X.cols() << endl;
////////	cout << "C:" << endl << C << endl;
////////}
////////
////////void ComputEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)
////////{
////////	//计算特征向量和特征值 SelfAdjointEigenSolver自动将计算得到的特征向量和特征值排序
////////	SelfAdjointEigenSolver<MatrixXd> eig(C);
////////	vec = eig.eigenvectors();
////////	cout << "vec:"<< endl << vec << endl;
////////	val = eig.eigenvalues();
////////	cout << "val:" << endl << val << endl;
////////
////////}
////////
////////// 计算维度
////////int ComputDim(MatrixXd &val)
////////{
////////	int dim;
////////	double sum = 0;
////////	for (int i = val.rows() - 1; i >= 0; --i)
////////	{
////////		sum += val(i, 0);
////////		dim = i;
////////		if (sum / val.sum() >= 0.8)//达到所需要的要求时
////////			break;
////////	}
////////	cout << val.rows() - dim << endl;
////////	return val.rows() - dim;
////////}
////////
////////int main()
////////{
////////	//文件操作
////////	ifstream fin("F:\\PCLprincipal\\pca070808\\Project1\\project01.txt");
////////	if (fin.is_open())
////////	{
////////		cout << "载入成功" << endl;
////////	}
////////	ofstream fout("outpot1.txt");
////////	ofstream fout1("X1.txt");
////////
////////	//定义所需要的量
////////	const int m = 5, n = 2, z = 2;
////////	MatrixXd X(m, n), C(z, z);
////////	MatrixXd vec, val;
////////
////////	//读取数据
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
////////	cout << "X为原始的数据：" << endl << X << endl;
////////	cout << endl;
////////
////////
////////	// 1 均值0标准化
////////	MatrixXd X1 = featurnormail(X);
////////	cout << "X1均值0标准化的数据：" << endl << X1.transpose() << endl;   ///.transpose()方便为了查看 N*3
////////	cout << endl;
////////
////////	// 2 计算协方差
////////	ComComputeCov(X1, C);
////////
////////	// 3 计算特征值和特征向量
////////	ComputEig(C, vec, val);
////////
////////	// 4 计算损失率，确定降低的维数
////////	int dim = ComputDim(val);
////////
////////	// 5计算结果
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
////////	cout << "res为用PCA算法之后的数据：" << endl << res.transpose() << endl;
////////	cout << endl;
////////
////////	// 6输出结果
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
////	////计算每一维的均值
////	//MatrixXd X1 = X.transpose();	
////	//cout << "X1" << endl << X1 << endl;
////	//MatrixXd X101 = X.transpose();
////	//cout << "X.transpose();:" << endl << X101 << endl;
////	//MatrixXd meanvalrow = X1.rowwise().mean();   ///按行求平均
////	//cout << "meanvalrow:" << endl << meanvalrow << endl;
////	//MatrixXd meanvalcol = X1.colwise().mean();   ///按列求平均
////	//cout << "meanvalcol" << endl << meanvalcol << endl;
////	////样本均值化为0
////	//RowVectorXd meanvecRow = meanvalrow;
////	//X1.rowwise() -= meanvecRow;
////	//cout << "X1.rowwise():" << endl << X1 << endl;
////	//return X1.transpose();
////	//计算每一维的均值
////	////MatrixXd X1 = X.transpose();
////	////MatrixXd meanval = X1.colwise().mean(); 
////	//////	cout << "meanval" << endl << meanval <<endl;	
////	//////样本均值化为0	
////	////RowVectorXd meanvecRow = meanval;
////	////X1.rowwise() -= meanvecRow;
////	////return X1.transpose();
////	MatrixXd X1 = X;
////	//MatrixXd X1 = X.transpose();
////	MatrixXd meanval = X1.colwise().mean();
////	cout << "meanval" << endl << meanval << endl;
////	//样本均值化为0	
////	RowVectorXd meanvecRow = meanval;
////	X1.rowwise() -= meanvecRow;
////	/*return X1.transpose();*/
////	return X1.transpose();
////}
////
////void ComComputeCov(MatrixXd &X, MatrixXd &C)
////
////{
////	//计算协方差矩阵
////	C = X*X.adjoint();//相当于XT*X adjiont()求伴随矩阵 相当于求矩阵的转置
////	/*cout << "C:" << endl << C.transpose() << endl;*/
////	C = C.array() / X.cols();//C.array()矩阵的数组形式
////	cout << "X.cols():" << endl << X.cols() << endl;
////	cout << "C:" << endl << C << endl;
////}
////
////void ComputEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)
////{
////	//计算特征向量和特征值 SelfAdjointEigenSolver自动将计算得到的特征向量和特征值排序
////	SelfAdjointEigenSolver<MatrixXd> eig(C);
////	vec = eig.eigenvectors();
////	cout << "vec:" << endl << vec << endl;
////	val = eig.eigenvalues();
////	cout << "val:" << endl << val << endl;
////
////}
////
////// 计算维度
////int ComputDim(MatrixXd &val)
////{
////	int dim;
////	double sum = 0;
////	for (int i = val.rows() - 1; i >= 0; --i)
////	{
////		sum += val(i, 0);
////		dim = i;
////		if (sum / val.sum() >= 0.8)//达到所需要的要求时
////			break;
////	}
////	cout << val.rows() - dim << endl;
////	return val.rows() - dim;
////}
////
////int main()
////{
////	//文件操作
////	ifstream fin("F:\\PCLprincipal\\pca070808\\Project1\\testData.txt");
////	if (fin.is_open())
////	{
////		cout << "载入成功" << endl;
////	}
////	ofstream fout("X1X2outpot1.txt");
////	ofstream fout1("X1X2X1.txt");
////
////	//定义所需要的量
////	const int m = 10, n = 2, z = 2;
////	MatrixXd X(m, n), C(z, z);
////	MatrixXd vec, val;
////
////	//读取数据
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
////	cout << "X为原始的数据：" << endl << X << endl;
////	cout << endl;
////
////
////	// 1 均值0标准化
////	MatrixXd X1 = featurnormail(X);
////	cout << "X1均值0标准化的数据：" << endl << X1.transpose() << endl;   ///.transpose()方便为了查看 N*3
////	cout << endl;
////
////	// 2 计算协方差
////	ComComputeCov(X1, C);
////
////	// 3 计算特征值和特征向量
////	ComputEig(C, vec, val);
////
////	// 4 计算损失率，确定降低的维数
////	int dim = ComputDim(val);
////	//int dim = 1;
////	// 5计算结果
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
////	cout << "res为用PCA算法之后的数据：" << endl << res.transpose() << endl;
////	cout << endl;
////
////	// 6输出结果
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
////	////计算每一维的均值
////	//MatrixXd X1 = X.transpose();	
////	//cout << "X1" << endl << X1 << endl;
////	//MatrixXd X101 = X.transpose();
////	//cout << "X.transpose();:" << endl << X101 << endl;
////	//MatrixXd meanvalrow = X1.rowwise().mean();   ///按行求平均
////	//cout << "meanvalrow:" << endl << meanvalrow << endl;
////	//MatrixXd meanvalcol = X1.colwise().mean();   ///按列求平均
////	//cout << "meanvalcol" << endl << meanvalcol << endl;
////	////样本均值化为0
////	//RowVectorXd meanvecRow = meanvalrow;
////	//X1.rowwise() -= meanvecRow;
////	//cout << "X1.rowwise():" << endl << X1 << endl;
////	//return X1.transpose();
////	//计算每一维的均值
////	////MatrixXd X1 = X.transpose();
////	////MatrixXd meanval = X1.colwise().mean(); 
////	//////	cout << "meanval" << endl << meanval <<endl;	
////	//////样本均值化为0	
////	////RowVectorXd meanvecRow = meanval;
////	////X1.rowwise() -= meanvecRow;
////	////return X1.transpose();
////	MatrixXd X1 = X;
////	//MatrixXd X1 = X.transpose();
////	MatrixXd meanval = X1.colwise().mean();
////	cout << "meanval" << endl << meanval << endl;
////	//样本均值化为0	
////	RowVectorXd meanvecRow = meanval;
////	X1.rowwise() -= meanvecRow;
////	/*return X1.transpose();*/
////	return X1.transpose();
////}
////
////void ComComputeCov(MatrixXd &X, MatrixXd &C)
////
////{
////	//计算协方差矩阵
////	C = X*X.adjoint();//相当于XT*X adjiont()求伴随矩阵 相当于求矩阵的转置
////	/*cout << "C:" << endl << C.transpose() << endl;*/
////	C = C.array() / X.cols();//C.array()矩阵的数组形式
////	cout << "X.cols():" << endl << X.cols() << endl;
////	cout << "C:" << endl << C << endl;
////}
////
////void ComputEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)
////{
////	//计算特征向量和特征值 SelfAdjointEigenSolver自动将计算得到的特征向量和特征值排序
////	SelfAdjointEigenSolver<MatrixXd> eig(C);
////	vec = eig.eigenvectors();
////	cout << "vec:"<< endl << vec << endl;
////	val = eig.eigenvalues();
////	cout << "val:" << endl << val << endl;
////
////}
////
////// 计算维度
////int ComputDim(MatrixXd &val)
////{
////	int dim;
////	double sum = 0;
////	for (int i = val.rows() - 1; i >= 0; --i)
////	{
////		sum += val(i, 0);
////		dim = i;
////		if (sum / val.sum() >= 0.8)//达到所需要的要求时
////			break;
////	}
////	cout << val.rows() - dim << endl;
////	return val.rows() - dim;
////}
////
////int main()
////{
////	//文件操作
////	ifstream fin("F:\\PCLprincipal\\pca070808\\Project1\\project01.txt");
////	if (fin.is_open())
////	{
////		cout << "载入成功" << endl;
////	}
////	ofstream fout("outpot1.txt");
////	ofstream fout1("X1.txt");
////
////	//定义所需要的量
////	const int m = 5, n = 2, z = 2;
////	MatrixXd X(m, n), C(z, z);
////	MatrixXd vec, val;
////
////	//读取数据
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
////	cout << "X为原始的数据：" << endl << X << endl;
////	cout << endl;
////
////
////	// 1 均值0标准化
////	MatrixXd X1 = featurnormail(X);
////	cout << "X1均值0标准化的数据：" << endl << X1.transpose() << endl;   ///.transpose()方便为了查看 N*3
////	cout << endl;
////
////	// 2 计算协方差
////	ComComputeCov(X1, C);
////
////	// 3 计算特征值和特征向量
////	ComputEig(C, vec, val);
////
////	// 4 计算损失率，确定降低的维数
////	int dim = ComputDim(val);
////
////	// 5计算结果
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
////	cout << "res为用PCA算法之后的数据：" << endl << res.transpose() << endl;
////	cout << endl;
////
////	// 6输出结果
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
	// Eigen:基本功能
	// VectorXd    等价于Matrix<double,Dynamic,1>;       //列向量，一列
	// RowVectorXd 等价于Matrix<double,1,Dynamic>;       //行向量，一行
	// MatrixXd    等价于Matrix<double,Dynamic,Dynamic>;

	//计算每一维的均值
	MatrixXd X1 = X.transpose();  //先来个普通转置【行列变换】

	VectorXd meanval = X1.rowwise().mean();//colwise按列求均值。//rowwise按行求均值。 //X1.rowwise()不可用
	cout << "meanval" << endl << meanval << endl;
	
	//样本均值化为0	
	X1.colwise() -= meanval;   //按列减，每一列与均值做差
	
	return X1;
}

void ComputeCov(MatrixXd &X1, MatrixXd &C)

{
	//计算协方差矩阵设我们有m个n维数据记录，将其按列排成n乘m的矩阵X，设C =X*XT/m
    //此处除以m
		
	C = X1*X1.adjoint();//相当于原始数据XT*X    adjiont()求伴随矩阵 相当于求矩阵的转置
	/*cout << "C:" << endl << C<< endl;*/
	C = C.array() / X1.cols();//C.array()矩阵的数组形式  ///逐元素除以m

	cout << "C:" << endl << C << endl;
}

void ComputEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)
{
	//计算特征向量和特征值 SelfAdjointEigenSolver自动将计算得到的特征向量和特征值排序
	SelfAdjointEigenSolver<MatrixXd> eig(C);
	//所求特征向量依据对应的特征值从小到大排列。
	vec = eig.eigenvectors();
	val = eig.eigenvalues();
	
	//故应该依靠翻转将特征向量依据特征值从大到小排列。【如下】
	vec.colwise().reverse();
	cout << "***************************************" << endl;
	cout << "vec.colwise().reverse():" << endl << vec.colwise().reverse() << endl;
	cout << "***************************************" << endl;
	val.colwise().reverse();
	cout << "val.colwise().reverse():" << endl << val.colwise().reverse() << endl;
	cout << "***************************************" << endl;
}

// 计算维度
int ComputDim(MatrixXd &val)
{
	int dim;
	double sum = 0;
	for (int i = val.rows() - 1; i >= 0; --i)
	{
		sum += val(i, 0);
		dim = i;
		if (sum / val.sum() >= 0.8)//达到所需要的要求时
			break;
	}
	//cout << val.rows() - dim << endl;
	return val.rows() - dim;
}

int main()
{
	//文件操作
	ifstream fin("F:\\The_Algorithms\\PCA\\m5n2.txt");
	if (fin.is_open())
	{
		cout << "载入成功" << endl;
	}
	ofstream fout("X1X2outpot1.txt");

	//定义所需要的量
	const int m = 5, n = 2, z = 2;
	MatrixXd X(m, n), C(z, z);  //C为协方差阵
	MatrixXd vec, val;

	//读取数据
	double in[z];
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			fin >> in[j];  //单值读入
		}
		for (int j = 1; j <= n; ++j)
			X(i, j - 1) = in[j - 1];
	}

	cout << endl;
	cout << "X为原始的数据：" << endl << X << endl;
	cout << endl;
	
	
	// 1 均值0标准化
	MatrixXd X1 = featurnormal(X);
	cout << "X1均值0标准化的数据：" << endl << X1.transpose() << endl;   ///.transpose()方便为了查看 N*3
	cout << endl;

	// 2 计算协方差
	ComputeCov(X1, C);

	// 3 计算特征值和特征向量
	ComputEig(C, vec, val);

	// 4 计算损失率，确定降低的维数
	int dim = ComputDim(val);
	//int dim = 1;
	// 5计算结果

	//但当涉猎
	//MatrixXd res001 = vec.transpose().rightCols(dim).transpose();
	//MatrixXd res001 = vec.bottomRows(dim);
	//MatrixXd res001 = vec.transpose().leftCols(dim).transpose();
	/*cout <<"X.colwise().reverse()"<<endl<< X.colwise().reverse() << endl;*/ //垂直翻转，优秀的很，

	MatrixXd res001 = vec.colwise().reverse().topRows(dim);  
	//按列翻转取前dim行 特征向量，实现依据特征值从大到小排列获取特征向量。
	
	cout << "res001:" << endl << res001 << endl;  //2*3
	MatrixXd res = res001*X1; //2*3  *  N*3  Y=PX 此Y即为降维之后的结果。

	//方便查看
	cout << "res为用PCA算法之后的数据：" << endl << res.transpose() << endl;
	
	// 6输出结果
	fout << "the result is" << res.rows() << "x" << res.cols() << "after pca algorithm" << endl;
	fout << res.transpose();
	fout.close();
	
	system("pause");
	return 0;
}

