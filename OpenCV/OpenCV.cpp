#include "stdafx.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <imgproc.hpp>


using namespace std;
using namespace cv;
  
/***********************************************
作者：King
日期：2017.06.01
参数：src:原图
	  dst:灰度图像
功能：用于彩色图像加权平均转换为灰度图像
************************************************/
void cvtCOLOR(Mat src, Mat dst)
{
	float R, G, B;
	for (int y = 0; y < src.rows; y++)
	{
		uchar* data = dst.ptr<uchar>(y);
		for (int x = 0; x < src.cols; x++)
		{
			B = src.at<Vec3b>(y, x)[0];
			G = src.at<Vec3b>(y, x)[1];
			R = src.at<Vec3b>(y, x)[2];
			data[x] = (int)(R * 0.299 + G * 0.587 + B * 0.114);//利用公式计算灰度值（加权平均法）
		}
	}
}

/***********************************************
作者：King
日期：2017.06.01
参数：dst:灰度图像
功能：生成灰度直方图
************************************************/
void hist(Mat dst)
{
	int channels = 0;
	Mat dstHist;
	int histSize = 256;
	float midRanges[] = { 0, 256 };
	const float *ranges[] = { midRanges };
	calcHist(&dst, 1, &channels, Mat(), dstHist, 1, &histSize, ranges, true, false); //计算直方图信息
	Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3); //创建一个黑底图像
	double g_dHistMaxValue;
	minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0); //将像素的个数整合到 图像的最大范围内

	//遍历直方图得到的数据
	for (int i = 0; i < 256; i++)
	{
		int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / g_dHistMaxValue);

		line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 0, 0));
	}

	imshow("直方图", drawImage);
}

/***********************************************
作者：King
日期：2017.06.01
参数：dst:灰度图像
功能：用于直方图均衡化
************************************************/
void hist_equalize(Mat dst)
{
	Mat dst_equalize;
	equalizeHist(dst, dst_equalize); //直方图均衡化
	imshow("直方图均衡化", dst_equalize);
}

/***********************************************
作者：King
日期：2017.06.01
参数：src:原图
	  dst:灰度图
	  min:总像素灰度值的最小百分比
	  max:总像素灰度值的最大百分比
功能：用于图像拉伸变换
************************************************/
void stretch(Mat src, Mat dst, float min, float max)
{
	int low_value = 0;    //拉伸后像素的最小值
	int high_value = 0;   //拉伸后像素的最大值

	float rate = 0;          //图像的拉伸率

	float stretch_p[256], stretch_p1[256], stretch_num[256];
	//清空三个数组,初始化填充数组元素为0
	memset(stretch_p, 0, sizeof(stretch_p));
	memset(stretch_p1, 0, sizeof(stretch_p1));
	memset(stretch_num, 0, sizeof(stretch_num));
	//统计图像各个灰度级出现的次数
	uchar* srcData = (uchar*)src.data;
	uchar* dstData = (uchar*)dst.data;
	int nHeight = src.cols;
	int nWidth = src.rows;
	int i, j;
	uchar nVal = 0;
	for (i = 0;i<nHeight;i++)
	{
		for (j = 0;j<nWidth;j++)
		{
			nVal = srcData[i*nWidth + j];
			stretch_num[nVal]++;
		}
	}
	//统计各个灰度级出现的概率
	for (i = 0;i<256;i++)
	{
		stretch_p[i] = stretch_num[i] / (nHeight*nWidth);
	}
	//统计各个灰度级的概率和
	for (i = 0;i<256;i++)
	{
		for (j = 0;j <= i;j++)
		{
			stretch_p1[i] += stretch_p[j];
		}
	}
	//计算两个阈值点的值
	for (i = 0;i<256;i++)
	{
		if (stretch_p1[i] < min)     //low_value取值接近于10%的总像素的灰度值
		{
			low_value = i;
		}
		if (stretch_p1[i] > max)     //high_value取值接近于90%的总像素的灰度值
		{
			high_value = i;
			break;
		}
	}
	rate = (float)255 / (high_value - low_value + 1);
	//进行灰度拉伸
	for (i = 0;i<nHeight;i++)
	{
		for (j = 0;j<nWidth;j++)
		{
			nVal = dstData[i*nWidth + j];
			if (nVal<low_value)
			{
				dstData[i*nWidth + j] = 0;
			}
			else if (nVal>high_value)
			{
				dstData[i*nWidth + j] = 255;
			}
			else
			{
				dstData[i*nWidth + j] = (uchar)((nVal - low_value)*rate + 0.5);
				if (dstData[i*nWidth + j]>255)
				{
					dstData[i*nWidth + j] = 255;
				}
			}
		}
	}
	imshow("拉伸变换", dst);
}

/***********************************************
作者：King
日期：2017.06.07
参数：image:传入图像
功能：加入盐噪声
************************************************/
void salt(cv::Mat image, int n) {

	int i, j;
	for (int k = 0; k<n / 2; k++) {

		// rand() is the random number generator  
		i = std::rand() % image.cols; // % 整除取余数运算符,rand=1022,cols=1000,rand%cols=22  
		j = std::rand() % image.rows;

		if (image.type() == CV_8UC1) { // gray-level image  

			image.at<uchar>(j, i) = 255; //at方法需要指定Mat变量返回值类型,如uchar等  

		}
		else if (image.type() == CV_8UC3) { // color image  

			image.at<cv::Vec3b>(j, i)[0] = 255; //cv::Vec3b为opencv定义的一个3个值的向量类型  
			image.at<cv::Vec3b>(j, i)[1] = 255; //[]指定通道，B:0，G:1，R:2  
			image.at<cv::Vec3b>(j, i)[2] = 255;
		}
	}
}

/***********************************************
作者：King
日期：2017.06.07
参数：image:传入图像
功能：加入椒噪声
************************************************/
void pepper(cv::Mat image, int n) {

	int i, j;
	for (int k = 0; k<n; k++) {

		// rand() is the random number generator  
		i = std::rand() % image.cols; // % 整除取余数运算符
		j = std::rand() % image.rows;

		if (image.type() == CV_8UC1) { // gray-level image  

			image.at<uchar>(j, i) = 0; //at方法需要指定Mat变量返回值类型,如uchar等  

		}
		else if (image.type() == CV_8UC3) { // color image  

			image.at<cv::Vec3b>(j, i)[0] = 0; //cv::Vec3b为opencv定义的一个3个值的向量类型  
			image.at<cv::Vec3b>(j, i)[1] = 0; //[]指定通道，B:0，G:1，R:2 
			image.at<cv::Vec3b>(j, i)[2] = 0;
		}
	}
}

int main()
{
	Mat src = imread("fxt.jpg");

	Mat dst(src.rows, src.cols, CV_8UC1);//大小与原图相同的八位单通道图
	Mat dst1(src.rows, src.cols, CV_8UC1);//大小与原图相同的八位单通道图
	cvtCOLOR(src, dst);//加权平均
	//cvtCOLOR(src, dst1);//加权平均
	//cvtColor(src, dst1, CV_BGR2GRAY);
	dst1 = src;
	imshow("原始图", src);
	imshow("灰度图", dst);
	hist(dst);  //绘制直方图
	hist_equalize(dst); //均衡化
	stretch(src, dst, 0.9, 1);  //拉伸变换
	salt(dst1, 1000);//加入盐噪声255  
	pepper(dst1, 1000);//加入椒噪声0
	imshow("加入噪声", dst1);
	Mat FinishMedianBlur;
	Mat FinishBlur;
	Mat FinishGaussianBlur;
	Mat FinishLaplacian;
	medianBlur(dst1, FinishMedianBlur, 3);
	imshow("中值滤波", FinishMedianBlur);
	blur(dst1,FinishBlur,Size(3, 3),Point( - 1, - 1));
	imshow("均值滤波", FinishBlur);
	GaussianBlur(dst1, FinishGaussianBlur, Size(5, 5), 0, 0);
	imshow("高斯滤波", FinishGaussianBlur);
	Laplacian(dst, FinishLaplacian, CV_8U, 3, 1, 0, BORDER_DEFAULT);
	imshow("拉普拉斯算子锐化", FinishLaplacian);
	waitKey(0);
	return 0;
}