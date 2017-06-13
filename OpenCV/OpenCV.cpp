#include "stdafx.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <imgproc.hpp>


using namespace std;
using namespace cv;
  
/***********************************************
���ߣ�King
���ڣ�2017.06.01
������src:ԭͼ
	  dst:�Ҷ�ͼ��
���ܣ����ڲ�ɫͼ���Ȩƽ��ת��Ϊ�Ҷ�ͼ��
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
			data[x] = (int)(R * 0.299 + G * 0.587 + B * 0.114);//���ù�ʽ����Ҷ�ֵ����Ȩƽ������
		}
	}
}

/***********************************************
���ߣ�King
���ڣ�2017.06.01
������dst:�Ҷ�ͼ��
���ܣ����ɻҶ�ֱ��ͼ
************************************************/
void hist(Mat dst)
{
	int channels = 0;
	Mat dstHist;
	int histSize = 256;
	float midRanges[] = { 0, 256 };
	const float *ranges[] = { midRanges };
	calcHist(&dst, 1, &channels, Mat(), dstHist, 1, &histSize, ranges, true, false); //����ֱ��ͼ��Ϣ
	Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3); //����һ���ڵ�ͼ��
	double g_dHistMaxValue;
	minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0); //�����صĸ������ϵ� ͼ������Χ��

	//����ֱ��ͼ�õ�������
	for (int i = 0; i < 256; i++)
	{
		int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / g_dHistMaxValue);

		line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 0, 0));
	}

	imshow("ֱ��ͼ", drawImage);
}

/***********************************************
���ߣ�King
���ڣ�2017.06.01
������dst:�Ҷ�ͼ��
���ܣ�����ֱ��ͼ���⻯
************************************************/
void hist_equalize(Mat dst)
{
	Mat dst_equalize;
	equalizeHist(dst, dst_equalize); //ֱ��ͼ���⻯
	imshow("ֱ��ͼ���⻯", dst_equalize);
}

/***********************************************
���ߣ�King
���ڣ�2017.06.01
������src:ԭͼ
	  dst:�Ҷ�ͼ
	  min:�����ػҶ�ֵ����С�ٷֱ�
	  max:�����ػҶ�ֵ�����ٷֱ�
���ܣ�����ͼ������任
************************************************/
void stretch(Mat src, Mat dst, float min, float max)
{
	int low_value = 0;    //��������ص���Сֵ
	int high_value = 0;   //��������ص����ֵ

	float rate = 0;          //ͼ���������

	float stretch_p[256], stretch_p1[256], stretch_num[256];
	//�����������,��ʼ���������Ԫ��Ϊ0
	memset(stretch_p, 0, sizeof(stretch_p));
	memset(stretch_p1, 0, sizeof(stretch_p1));
	memset(stretch_num, 0, sizeof(stretch_num));
	//ͳ��ͼ������Ҷȼ����ֵĴ���
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
	//ͳ�Ƹ����Ҷȼ����ֵĸ���
	for (i = 0;i<256;i++)
	{
		stretch_p[i] = stretch_num[i] / (nHeight*nWidth);
	}
	//ͳ�Ƹ����Ҷȼ��ĸ��ʺ�
	for (i = 0;i<256;i++)
	{
		for (j = 0;j <= i;j++)
		{
			stretch_p1[i] += stretch_p[j];
		}
	}
	//����������ֵ���ֵ
	for (i = 0;i<256;i++)
	{
		if (stretch_p1[i] < min)     //low_valueȡֵ�ӽ���10%�������صĻҶ�ֵ
		{
			low_value = i;
		}
		if (stretch_p1[i] > max)     //high_valueȡֵ�ӽ���90%�������صĻҶ�ֵ
		{
			high_value = i;
			break;
		}
	}
	rate = (float)255 / (high_value - low_value + 1);
	//���лҶ�����
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
	imshow("����任", dst);
}

/***********************************************
���ߣ�King
���ڣ�2017.06.07
������image:����ͼ��
���ܣ�����������
************************************************/
void salt(cv::Mat image, int n) {

	int i, j;
	for (int k = 0; k<n / 2; k++) {

		// rand() is the random number generator  
		i = std::rand() % image.cols; // % ����ȡ���������,rand=1022,cols=1000,rand%cols=22  
		j = std::rand() % image.rows;

		if (image.type() == CV_8UC1) { // gray-level image  

			image.at<uchar>(j, i) = 255; //at������Ҫָ��Mat��������ֵ����,��uchar��  

		}
		else if (image.type() == CV_8UC3) { // color image  

			image.at<cv::Vec3b>(j, i)[0] = 255; //cv::Vec3bΪopencv�����һ��3��ֵ����������  
			image.at<cv::Vec3b>(j, i)[1] = 255; //[]ָ��ͨ����B:0��G:1��R:2  
			image.at<cv::Vec3b>(j, i)[2] = 255;
		}
	}
}

/***********************************************
���ߣ�King
���ڣ�2017.06.07
������image:����ͼ��
���ܣ����뽷����
************************************************/
void pepper(cv::Mat image, int n) {

	int i, j;
	for (int k = 0; k<n; k++) {

		// rand() is the random number generator  
		i = std::rand() % image.cols; // % ����ȡ���������
		j = std::rand() % image.rows;

		if (image.type() == CV_8UC1) { // gray-level image  

			image.at<uchar>(j, i) = 0; //at������Ҫָ��Mat��������ֵ����,��uchar��  

		}
		else if (image.type() == CV_8UC3) { // color image  

			image.at<cv::Vec3b>(j, i)[0] = 0; //cv::Vec3bΪopencv�����һ��3��ֵ����������  
			image.at<cv::Vec3b>(j, i)[1] = 0; //[]ָ��ͨ����B:0��G:1��R:2 
			image.at<cv::Vec3b>(j, i)[2] = 0;
		}
	}
}

int main()
{
	Mat src = imread("fxt.jpg");

	Mat dst(src.rows, src.cols, CV_8UC1);//��С��ԭͼ��ͬ�İ�λ��ͨ��ͼ
	Mat dst1(src.rows, src.cols, CV_8UC1);//��С��ԭͼ��ͬ�İ�λ��ͨ��ͼ
	cvtCOLOR(src, dst);//��Ȩƽ��
	//cvtCOLOR(src, dst1);//��Ȩƽ��
	//cvtColor(src, dst1, CV_BGR2GRAY);
	dst1 = src;
	imshow("ԭʼͼ", src);
	imshow("�Ҷ�ͼ", dst);
	hist(dst);  //����ֱ��ͼ
	hist_equalize(dst); //���⻯
	stretch(src, dst, 0.9, 1);  //����任
	salt(dst1, 1000);//����������255  
	pepper(dst1, 1000);//���뽷����0
	imshow("��������", dst1);
	Mat FinishMedianBlur;
	Mat FinishBlur;
	Mat FinishGaussianBlur;
	Mat FinishLaplacian;
	medianBlur(dst1, FinishMedianBlur, 3);
	imshow("��ֵ�˲�", FinishMedianBlur);
	blur(dst1,FinishBlur,Size(3, 3),Point( - 1, - 1));
	imshow("��ֵ�˲�", FinishBlur);
	GaussianBlur(dst1, FinishGaussianBlur, Size(5, 5), 0, 0);
	imshow("��˹�˲�", FinishGaussianBlur);
	Laplacian(dst, FinishLaplacian, CV_8U, 3, 1, 0, BORDER_DEFAULT);
	imshow("������˹������", FinishLaplacian);
	waitKey(0);
	return 0;
}