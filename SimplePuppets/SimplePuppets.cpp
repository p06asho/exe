///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2012, Tadas Baltrusaitis, all rights reserved.
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
//
//     * The software is provided under the terms of this licence stricly for
//       academic, non-commercial, not-for-profit purposes.
//     * Redistributions of source code must retain the above copyright notice, 
//       this list of conditions (licence) and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright 
//       notice, this list of conditions (licence) and the following disclaimer 
//       in the documentation and/or other materials provided with the 
//       distribution.
//     * The name of the author may not be used to endorse or promote products 
//       derived from this software without specific prior written permission.
//     * As this software depends on other libraries, the user must adhere to 
//       and keep in place any licencing terms of those libraries.
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite the following work:
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 3D
//       Constrained Local Model for Rigid and Non-Rigid Facial Tracking.
//       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.    
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////


// SimpleCLM.cpp : Defines the entry point for the console application.
//#include <glew.h>

#include <omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<glew.h>
#include <filesystem.hpp>
#include <filesystem\fstream.hpp>
#include <highgui.h>
#include <GL/freeglut.h>
#include<GL/GL.h>
#include "SimplePuppets.h"
#include <cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <cmath>

omp_lock_t writelock;

// The modules that are being used for tracking
CLMTracker::TrackerCLM clmModel;
//A few OpenCV mats:
Mat faceimg;				//face
Mat avatarWarpedHead2;		//warped face
Mat avatarS2;				//shape
String avatarfile2;			//file where the avatar is
string file = "..\\videos\\default.wmv";
string oldfile;

bool GETFACE = false;		//get a new face
bool SHOWIMAGE = false;		//show the webcam image (in the main window). Turned on when the matrix isn't empty


bool gotFace = false;
Vec6d poseEstimateCLM;
bool visi [66];
Mat shape;
Point features[66];
Point initFeatures[66];
bool gotContext;
int mainargc;
char **mainargv;
int GLWindowID;
GLuint textureID;
GLuint backgroundID;
Mat initImg;
Mat background;
vector<Point> extraPoints;
vector<Point> extraUpdated;
vector<vector<vector<int>>> triangulation;
vector<vector<int>> lips;



void use_webcam(){			//called when the 'use webcam' checkbox is ticked
	USEWEBCAM = true;
	CHANGESOURCE = true;
	resetERIExpression();
	cout << "Using Webcam. " << endl;
	resetERIExpression();
}

static void printErrorAndAbort( const std::string & error )
{
	std::cout << error << std::endl;
	abort();
}

#define FATAL_STREAM( stream ) \
	printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;
using namespace cv;

// takes in doubles for orientation for added precision, but ultimately returns a float matrix
Matx33f Euler2RotationMatrix(const Vec3d& eulerAngles)
{
	Matx33f rotationMatrix;

	double s1 = sin(eulerAngles[0]);
	double s2 = sin(eulerAngles[1]);
	double s3 = sin(eulerAngles[2]);

	double c1 = cos(eulerAngles[0]);
	double c2 = cos(eulerAngles[1]);
	double c3 = cos(eulerAngles[2]);

	rotationMatrix(0,0) = (float)(c2 * c3);
	rotationMatrix(0,1) = (float)(-c2 *s3);
	rotationMatrix(0,2) = (float)(s2);
	rotationMatrix(1,0) = (float)(c1 * s3 + c3 * s1 * s2);
	rotationMatrix(1,1) = (float)(c1 * c3 - s1 * s2 * s3);
	rotationMatrix(1,2) = (float)(-c2 * s1);
	rotationMatrix(2,0) = (float)(s1 * s3 - c1 * c3 * s2);
	rotationMatrix(2,1) = (float)(c3 * s1 + c1 * s2 * s3);
	rotationMatrix(2,2) = (float)(c1 * c2);

	return rotationMatrix;
}

void Project(Mat_<float>& dest, const Mat_<float>& mesh, Size size, double fx, double fy, double cx, double cy)
{
	dest = Mat_<float>(mesh.rows,2, 0.0);

	int NbPoints = mesh.rows;

	register float X, Y, Z;


	Mat_<float>::const_iterator mData = mesh.begin();
	Mat_<float>::iterator projected = dest.begin();

	for(int i = 0;i < NbPoints; i++)
	{
		// Get the points
		X = *(mData++);
		Y = *(mData++);
		Z = *(mData++);

		float x;
		float y;

		// if depth is 0 the projection is different
		if(Z != 0)
		{
			x = (float)((X * fx / Z) + cx);
			y = (float)((Y * fy / Z) + cy);
		}
		else
		{
			x = X;
			y = Y;
		}

		// Clamping to image size
		if( x < 0 )	
		{
			x = 0.0;
		}
		else if (x > size.width - 1)
		{
			x = size.width - 1.0f;
		}
		if( y < 0 )
		{
			y = 0.0;
		}
		else if( y > size.height - 1) 
		{
			y = size.height - 1.0f;
		}

		// Project and store in dest matrix
		(*projected++) = x;
		(*projected++) = y;
	}

}

void DrawBox(Mat image, Vec6d pose, Scalar color, int thickness, float fx, float fy, float cx, float cy)
{
	float boxVerts[] = {-1, 1, -1,
		1, 1, -1,
		1, 1, 1,
		-1, 1, 1,
		1, -1, 1,
		1, -1, -1,
		-1, -1, -1,
		-1, -1, 1};
	Mat_<float> box = Mat(8, 3, CV_32F, boxVerts).clone() * 100;


	Matx33f rot = Euler2RotationMatrix(Vec3d(pose[3], pose[4], pose[5]));
	Mat_<float> rotBox;

	Mat((Mat(rot) * box.t())).copyTo(rotBox);
	rotBox = rotBox.t();

	rotBox.col(0) = rotBox.col(0) + pose[0];
	rotBox.col(1) = rotBox.col(1) + pose[1];
	rotBox.col(2) = rotBox.col(2) + pose[2];

	// draw the lines
	Mat_<float> rotBoxProj;
	Project(rotBoxProj, rotBox, image.size(), fx, fy, cx, cy);

	Mat begin;
	Mat end;

	rotBoxProj.row(0).copyTo(begin);
	rotBoxProj.row(1).copyTo(end);
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);

	rotBoxProj.row(1).copyTo(begin);
	rotBoxProj.row(2).copyTo(end);
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);

	rotBoxProj.row(2).copyTo(begin);
	rotBoxProj.row(3).copyTo(end);
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);

	rotBoxProj.row(0).copyTo(begin);
	rotBoxProj.row(3).copyTo(end);
	//std::cout << begin <<endl;
	//std::cout << end <<endl;
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);

	rotBoxProj.row(2).copyTo(begin);
	rotBoxProj.row(4).copyTo(end);
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);

	rotBoxProj.row(1).copyTo(begin);
	rotBoxProj.row(5).copyTo(end);
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);

	rotBoxProj.row(0).copyTo(begin);
	rotBoxProj.row(6).copyTo(end);
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);

	rotBoxProj.row(3).copyTo(begin);
	rotBoxProj.row(7).copyTo(end);
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);

	rotBoxProj.row(6).copyTo(begin);
	rotBoxProj.row(5).copyTo(end);
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);

	rotBoxProj.row(5).copyTo(begin);
	rotBoxProj.row(4).copyTo(end);
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);

	rotBoxProj.row(4).copyTo(begin);
	rotBoxProj.row(7).copyTo(end);
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);

	rotBoxProj.row(7).copyTo(begin);
	rotBoxProj.row(6).copyTo(end);
	cv::line(image, Point((int)begin.at<float>(0), (int)begin.at<float>(1)), Point((int)end.at<float>(0), (int)end.at<float>(1)), color, thickness);


}

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 1; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

GLuint matToTexture(cv::Mat &mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter)
{
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mat.cols, mat.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, mat.ptr());
	return textureID;
}

double Distance(Point a, Point b)
{
	double x = a.x - b.x;
	x *= x;
	double y= a.y - b.y;
	y *= y;
	return sqrt(x+y);
}

void getTriangulation(Mat img)
{
	Rect r(0,0,img.cols,img.rows);
	Subdiv2D subdiv(r);
	for (int i = 0; i < 66; i++)
	{
		subdiv.insert(features[i]);
	}
	for (int i = 0; i < extraPoints.size(); i++)
	{
		subdiv.insert(extraPoints[i]);
	}
	vector<Vec6f> triangles;
	subdiv.getTriangleList(triangles);
	for(int i = 0; i < triangles.size(); i++)
	{
		Vec6f t = triangles[i];
		vector<vector<int>> current;
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 66; k++)
			{
				if (t[2*j] == features[k].x && t[2*j+1] == features[k].y)
				{
					vector<int> point;
					point.push_back(0);
					point.push_back(k);
					current.push_back(point);
				}
			}
			for (int k = 0; k < extraPoints.size(); k++)
			{
				if (t[2*j] == extraPoints[k].x && t[2*j+1] == extraPoints[k].y)
				{
					vector<int> point;
					point.push_back(1);
					point.push_back(k);
					current.push_back(point);
				}
			}
		}
		if (current.size() == 3)
		{
			triangulation.push_back(current);
		}
	}
}

Point rotatePoint(Point a, Point b, double angle)//rotate b around a through angle
{
	double s = sin(angle);
	double c = cos(angle);
	Point output(b.x-a.x, b.y-a.y);
	output.x = cvRound(c*output.x - s*output.y);
	output.y = cvRound(s*output.x + c*output.y);
	output.x += a.x;
	output.y += a.y;
	return output;
}

double getTheta(Point a, Point b)//Point a should be the top of the nose bridge, point b the bottom
{
	int xDif = a.x - b.x;
	int yDif = a.y - b.y;
	double ratio;
	if (xDif == 0 || yDif == 0)
	{
		ratio = 0.0;
	}
	else
	{
		ratio = xDif/yDif;
	}
	double angle = atan(ratio);
	return angle;
}

Point getCentre(Point f[])
{
	int xsum = 0;
	int ysum = 0;
	for (int i = 0; i < 66; i++)
	{
		xsum += f[i].x;
		ysum += f[i].y;
	}
	xsum = xsum/66;
	ysum = ysum/66;
	return Point(xsum, ysum);
}

void doInpainting(Mat img)
{
	Mat mask(img.rows, img.cols, CV_8U, cvScalar(0));
	Mat inpainted = img.clone();
	vector<Point> allPoints;
	vector<Point> hull;
	for (int i = 0; i < 66; i++)
	{
		allPoints.push_back(initFeatures[i]);
	}
	for (int i = 0; i < extraPoints.size(); i++)
	{
		allPoints.push_back(extraPoints[i]);
	}
	convexHull(allPoints, hull);
	fillConvexPoly(mask, hull, hull.size(), 255);
	imshow("mask", mask);
	inpaint(img, mask, inpainted, 5, INPAINT_TELEA);
	imshow("inpainted", inpainted);
}

void updateExtraPoints()
{
	if (extraUpdated.size() != extraPoints.size())
	{
		extraUpdated = extraPoints;
	}
	//first, the points above the eyebrows (the first 10)
	//get a scaling factor using the nose
	double scaleFactor = Distance(features[27], features[30])/Distance(initFeatures[27], initFeatures[30]);
	//and a rotation using the points at the side of the face
	double initAngle = getTheta(initFeatures[1], initFeatures[0]);
	double currentAngle = getTheta(features[1], features[0]);
	double rotation = currentAngle - initAngle;
	for (int i = 0; i < 10; i++)//for each point
	{
		int newDistance = cvRound(scaleFactor*(extraPoints[i].y - initFeatures[17+i].y));
		Point pos = rotatePoint(Point(0,0), Point(0, newDistance), rotation);
		extraUpdated[i] = Point(features[17+i].x+pos.x, features[17+i].y+pos.y);
	}
	//now the side points
	//these are not directly along an axis from a feature - they need proper scaling
	double xScaleFactor = Distance(features[0], features[17])/Distance(initFeatures[0], initFeatures[17]);
	int newX = cvRound(xScaleFactor*(extraPoints[10].x-initFeatures[17].x));
//	int newY = cvRound(scaleFactor*(extraPoints[10].y-initFeatures[17].y));
	Point pos = rotatePoint(Point(0,0), Point(newX, 0), rotation);
	extraUpdated[10] = Point(features[17].x+pos.x, features[17].y+pos.y);

	newX = cvRound(xScaleFactor*(extraPoints[12].x-initFeatures[17].x));
	int newY = cvRound(scaleFactor*(extraPoints[12].y-initFeatures[17].y));
	pos = rotatePoint(Point(0,0), Point(newX, newY), rotation);
	extraUpdated[12] = Point(features[17].x+pos.x, features[17].y+pos.y);

	xScaleFactor = Distance(features[16], features[26])/Distance(initFeatures[16], initFeatures[26]);
	newX = cvRound(xScaleFactor*(extraPoints[11].x-initFeatures[26].x));
//	newY = cvRound(scaleFactor*(extraPoints[11].y-initFeatures[26].y));
	pos = rotatePoint(Point(0,0), Point(newX, 0), rotation);
	extraUpdated[11] = Point(features[26].x+pos.x, features[26].y+pos.y);

	newX = cvRound(xScaleFactor*(extraPoints[13].x-initFeatures[26].x));//likewise
	newY = cvRound(scaleFactor*(extraPoints[13].y-initFeatures[26].y));
	pos = rotatePoint(Point(0,0), Point(newX, newY), rotation);
	extraUpdated[13] = Point(features[26].x+pos.x, features[26].y+pos.y);

	Mat extras(480, 640, CV_32F);
	for (int i = 0; i < extraUpdated.size(); i++)
	{
		circle(extras, extraUpdated[i], 1, Scalar(255,255,255));
	}
	imshow("extras", extras);
	for (int i = 0; i < extraUpdated.size(); i++)
	{
		circle(extras, extraUpdated[i], 1, Scalar(0,0,0));
	}
}

void getLips()
{
	vector<int> tri;
	tri.push_back(48);
	tri.push_back(49);
	tri.push_back(60);
	lips.push_back(tri);
	tri[0] = 50;
	lips.push_back(tri);
	tri[1] = 61;
	lips.push_back(tri);
	tri[2] = 51;
	lips.push_back(tri);
	tri[0] = 62;
	lips.push_back(tri);
	tri[1] = 52;
	lips.push_back(tri);
	tri[2] = 53;
	lips.push_back(tri);
	tri[1] = 54;
	lips.push_back(tri);
	tri[0] = 48;
	tri[1] = 65;
	tri[2] = 59;
	lips.push_back(tri);
	tri[0] = 58;
	lips.push_back(tri);
	tri[2] = 64;
	lips.push_back(tri);
	tri[1] = 57;
	lips.push_back(tri);
	tri[0] = 63;
	lips.push_back(tri);
	tri[2] = 56;
	lips.push_back(tri);
	tri[1] = 55;
	lips.push_back(tri);
	tri[2] = 54;
	lips.push_back(tri);
}

void doTransformation()
{
	//Don't get new triangulations every time. Use the original set of triangles, and apply it like a texture
	if (!gotContext)
	{
		background = initImg.clone();
		glutInitWindowSize(640, 480);
		glutInitWindowPosition(0,0);
		GLWindowID = glutCreateWindow("Output");
		glutSetWindow(GLWindowID);
		gotContext = true;
		Mat flipped;
		flip(initImg, flipped, 0);
		textureID = matToTexture(flipped, GL_NEAREST, GL_NEAREST, GL_CLAMP);
		getLips();
		//flip(initImg, background, 0);
		//fillConvexPoly(background, perimeter, 27, Scalar(0));
		//backgroundID = matToTexture(background, GL_NEAREST, GL_NEAREST, GL_CLAMP);
	}
	
	updateExtraPoints();
	//Rendering stuff here. Remember to swap buffers each time. 
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
	/*glBindTexture(GL_TEXTURE_2D, backgroundID);
	glBegin(GL_QUADS);
		glTexCoord2f(0.0, 0.0); glVertex2d(-1.0, -1.0);
		glTexCoord2f(0.0, 1.0); glVertex2d(-1.0, 1.0);
		glTexCoord2f(1.0, 1.0); glVertex2d(1.0, 1.0);
		glTexCoord2f(1.0, 0.0); glVertex2d(1.0, -1.0);
	glEnd();
	glDrawArrays(GL_QUADS, 0, 4);*/
	glBindTexture(GL_TEXTURE_2D, textureID);
	for (int i = 0; i < triangulation.size(); i++)
	{
		bool draw = true;
		bool bottom = false;
		bool top = false;
		for (int a = 0; a < 3; a++)//check whether the triangle covers the area between the lips
		{
			if (triangulation[i][a][1] >= 60)
			{
				draw = false;
				break;
			}
			if (triangulation[i][a][1] > 48 && triangulation[i][a][1] < 52)
			{
				top = true;
			}
			if (triangulation[i][a][1] > 52 && triangulation[i][a][1] < 60)
			{
				bottom = true;
			}
			if (top && bottom)
			{
				draw = false;
			}
		}
		if (draw)
		{
			glBegin(GL_TRIANGLES);
				for (int j = 0; j < 3; j++)
				{
					if (triangulation[i][j][0] == 0)
					{
						glTexCoord2f(initFeatures[triangulation[i][j][1]].x/640.0, 
							1.0 - initFeatures[triangulation[i][j][1]].y/480.0);
						glVertex2d((features[triangulation[i][j][1]].x/320.0) - 1.0, 
							1.0 - (features[triangulation[i][j][1]].y/240.0));
					}
					else
					{
						glTexCoord2f(extraPoints[triangulation[i][j][1]].x/640.0,
							1.0 - extraPoints[triangulation[i][j][1]].y/480.0);
						glVertex2d((extraUpdated[triangulation[i][j][1]].x/320.0) - 1.0,
							1.0 - (extraUpdated[triangulation[i][j][1]].y/240.0));
					}
				}
			glEnd();
		}
	}
	for (int i = 0; i < lips.size(); i++)
	{
		glBegin(GL_TRIANGLES);
			for (int j = 0; j < 3; j++)
			{
				glTexCoord2f(initFeatures[lips[i][j]].x/640.0, 
					1.0 - initFeatures[lips[i][j]].y/480.0);
				glVertex2d((features[lips[i][j]].x/320.0) - 1.0, 
					1.0 - (features[lips[i][j]].y/240.0));
			}
		glEnd();
	}
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glutSwapBuffers();
}

void getExtraPoints(Mat edges)
{
	//first, find the points to include
	for (int i = 17; i <= 26; i++)//above the eyebrows
	{
		for (int j = 0; j <= features[i].y; j++)//moving down from the top of the image
		{
			if (edges.at<uchar>(Point(features[i].x, j)) != 0)//if we have an edge, record it and stop looking
			{
				extraPoints.push_back(Point(features[i].x, j));
				break;
			}
		}
	}
	//then to the side
	for (int i = 0; i <= features[17].x; i++)//from the left end inwards
	{
		if (edges.at<uchar>(Point(i, features[17].y)) != 0)//again, if we have an edge, record it and stop
		{
			extraPoints.push_back(Point(i, features[17].y));
			break;
		}
	}
	for (int i = edges.cols - 1; i >= features[26].x; i--)//this time from the right inwards
	{
		if (edges.at<uchar>(Point(i, features[26].y)) != 0)//record and stop on the first edge
		{
			extraPoints.push_back(Point(i, features[26].y));
			break;
		}
	}
	//and, to make it cleaner, two "halfway" points
	int xPos = cvRound((extraPoints[extraPoints.size() - 2].x + features[17].x) / 2);//halfway between the outer eyebrow point and the side point
	for (int i = 0; i <= features[17].y; i++)//from the top
	{
		if (edges.at<uchar>(Point(xPos, i)) != 0)//if we find an edge, record it and stop looking
		{
			extraPoints.push_back(Point(xPos, i));
			break;
		}
	}
	xPos = cvRound((extraPoints[extraPoints.size() - 2].x + features[26].x) / 2);
	for (int i = 0; i <= features[26].y; i++)
	{
		if (edges.at<uchar>(Point(xPos, i)) != 0)
		{
			extraPoints.push_back(Point(xPos, i));
			break;
		}
	}
	for (int i = 0; i < 3; i++)//this may not be a good idea - it might cause bad behaviour with triangles and hulls
	{							//Yes. You can add logic to ignore the Point(-1,-1) bits. 
		for (int j = 0; j < features[i].x; j++)
		{
			if (edges.at<uchar>(Point(j, features[i].y)) != 0)
			{
				extraPoints.push_back(Point(j, features[i].y));
				break;
			}
		}
		if (extraPoints.size() < 14 + i + 1)
		{
			extraPoints.push_back(Point(-1,-1));
		}
	}
	for (int i = 16; i > 13; i--)
	{
		for (int j = edges.cols; j >= features[i].x; j--)
		{
			if (edges.at<uchar>(Point(j, features[i].y)) != 0)
			{
				extraPoints.push_back(Point(j, features[i].y));
				break;
			}
		}
		if (extraPoints.size() < 17 + i + 1)
		{
			extraPoints.push_back(Point(-1,-1));
		}
	}
}

void extractFace(Mat img)
{
	for (int i = 0; i < 66; i++)
	{
		initFeatures[i] = features[i];
	}
	//get some edges so we can try to include the hair
	Mat gray;
	Mat edges;
	cvtColor(img, gray, CV_BGR2GRAY);
	double thresh = threshold(gray, edges, 0, 255, THRESH_BINARY+THRESH_OTSU);
	Canny(gray, edges, thresh/2.0, thresh);
	namedWindow("edges", 1);
	imshow("edges", edges);
	gotFace = true;//Don't do it every frame
	getExtraPoints(edges);
	doInpainting(img);
	getTriangulation(img);
	Mat triangles = img.clone();
	for (int i = 0; i < triangulation.size(); i++)
	{
		line(triangles, triangulation[i][0][0] == 0 ? features[triangulation[i][0][1]] : extraPoints[triangulation[i][0][1]], triangulation[i][1][0] == 0 ? features[triangulation[i][1][1]] : extraPoints[triangulation[i][1][1]], Scalar(255, 0, 0));
		line(triangles, triangulation[i][1][0] == 0 ? features[triangulation[i][1][1]] : extraPoints[triangulation[i][1][1]], triangulation[i][2][0] == 0 ? features[triangulation[i][2][1]] : extraPoints[triangulation[i][2][1]], Scalar(255, 0, 0));
		line(triangles, triangulation[i][2][0] == 0 ? features[triangulation[i][2][1]] : extraPoints[triangulation[i][2][1]], triangulation[i][0][0] == 0 ? features[triangulation[i][0][1]] : extraPoints[triangulation[i][0][1]], Scalar(255, 0, 0));
	}
	namedWindow("triangles", 1);
	imshow("triangles", triangles);
	imwrite("Z:\\triangles.jpg", triangles);
}

void doFaceTracking(int argc, char **argv){


	bool done = false;

	while(!done )
	{
		cout << "Not done yet!" << endl;
		cout << USEWEBCAM << ", " << NEWFILE << endl;


		vector<string> arguments = get_arguments(argc, argv);

		// Some initial parameters that can be overriden from command line	
		vector<string> files, dDirs, outposes, outvideos, outfeatures;

		// By default try webcam
		int device = 0;

		// cx and cy aren't always half dimx or half dimy, so need to be able to override it (start with unit vals and init them if none specified)
		float fx = 500, fy = 500, cx = 0, cy = 0;
		int dimx = 0, dimy = 0;

		bool useCLMTracker = true;

		CLMWrapper::CLMParameters clmParams(arguments);

		clmParams.wSizeCurrent = clmParams.wSizeInit;

		PoseDetectorHaar::PoseDetectorHaarParameters haarParams;

#if OS_UNIX
		haarParams.ClassifierLocation = "/usr/share/OpenCV-2.3.1/haarcascades/haarcascade_frontalface_alt.xml";
#else
		haarParams.ClassifierLocation = "..\\lib\\3rdParty\\OpenCV\\classifiers\\haarcascade_frontalface_alt.xml";
#endif

		// Get the input output file parameters
		CLMWrapper::get_video_input_output_params(files, dDirs, outposes, outvideos, outfeatures, arguments);
		// Get camera parameters
		CLMWrapper::get_camera_params(fx, fy, cx, cy, dimx, dimy, arguments);    

		// Face detector initialisation
		CascadeClassifier classifier(haarParams.ClassifierLocation);

		int f_n = -1;

		// We might specify multiple video files as arguments
		if(files.size() > 0)
		{
			f_n++;			
			file = files[f_n];
		}

		if(NEWFILE){
			file = inputfile;
		}

		bool readDepth = !dDirs.empty();

		if(USEWEBCAM)
		{
			INFO_STREAM( "Attempting to capture from device: " << device );
			vCap = VideoCapture( device );
			if( !vCap.isOpened() ) 
			{
				USEWEBCAM = false;
				CHANGESOURCE = true;
				resetERIExpression();
			}
		}

		// Do some grabbing
		if( !USEWEBCAM)
		{

			if(file.size() > 0 )
			{
				INFO_STREAM( "Attempting to read from file: " << file );
				vCap = VideoCapture( file );
			}
			else 
			{
				INFO_STREAM("No file specified. Please use webcam or load file manually");
				USEWEBCAM = 1;
			}
		}
		vCap = VideoCapture("Z:\\Documents\\Project\\init.wmv");
		if (!vCap.isOpened())
		{
			printf("Failed to open file");
		}

		Mat img;
		vCap.read(img);
		//vCap >> img;


		// If no dimensions defined, do not do any resizing
		if(dimx == 0 || dimy == 0)
		{
			dimx = img.cols;
			dimy = img.rows;
		}

		// If optical centers are not defined just use center of image
		if(cx == 0 || cy == 0)
		{
			cx = dimx / 2.0f;
			cy = dimy / 2.0f;
		}

		//for constant-size input:
		//dimx = 200;
		//dimy = 200;
		//cx = 100;
		//cy = 100;



		int frameProc = 0;

		// faces in a row detected
		facesInRow = 0;

		// saving the videos
		VideoWriter writerFace;
		if(!outvideos.empty())
		{
			writerFace = VideoWriter(outvideos[f_n], CV_FOURCC('D','I','V','X'), 30, img.size(), true);		
		}

		// Variables useful for the tracking itself
		bool success = false;
		trackingInitialised = false;

		// For measuring the timings
		int64 t1,t0 = cv::getTickCount();
		double fps = 10;

		Mat disp;
		Mat rgbimg;

		CHANGESOURCE = false;


		//todo: fix bug with crash when selecting video file to play under webcam mode (disable video select button?)
		//also occasionally opencv error when changing between different sizes of video input/webcam owing to shape going outside boundries. 


		gotContext = false;
		while(!img.empty() && !CHANGESOURCE && !done)						//This is where stuff happens once the file's open.
		{		
			//for constant-size input:
			//resize(img, img, Size( dimx, dimy));

			Mat_<float> depth;
			Mat_<uchar> gray;
			cvtColor(img, gray, CV_BGR2GRAY);
			cvtColor(img, rgbimg, CV_BGR2RGB);

			if(GRAYSCALE)
			{
				cvtColor(gray, rgbimg, CV_GRAY2RGB);
			}

			parsecolour(rgbimg);			//this sends the rgb image to the PAW loop

			writeToFile = 0;

			if(GETFACE)
			{
				GETFACE = false;
				writeToFile = !writeToFile;
				PAWREADAGAIN = true;
			}

			// Don't resize if it's unneeded
			Mat_<uchar> img_scaled;		
			if(dimx != gray.cols || dimy != gray.rows)
			{
				resize( gray, img_scaled, Size( dimx, dimy ) , 0, 0, INTER_NEAREST );
				resize(img, disp, Size( dimx, dimy), 0, 0, INTER_NEAREST );
			}
			else
			{
				img_scaled = gray;
				disp = img.clone();
			}

			disp.copyTo(faceimg);

			//namedWindow("colour",1);

			// Get depth image
			if(readDepth)
			{
				char* dst = new char[100];
				std::stringstream sstream;
				//sstream << dDir << "\\depth%06d.png";
				sstream << dDirs[f_n] << "\\depth%05d.png";
				sprintf(dst, sstream.str().c_str(), frameProc + 1);
				Mat_<short> dImg = imread(string(dst), -1);
				if(!dImg.empty())
				{
					if(dimx != dImg.cols || dimy != dImg.rows)
					{
						Mat_<short> dImgT;
						resize(dImg, dImgT, Size( dimx, dimy), 0, 0, INTER_NEAREST );
						dImgT.convertTo(depth, CV_32F);
					}
					else
					{
						dImg.convertTo(depth, CV_32F);
					}
				}
				else
				{
					WARN_STREAM( "Can't find depth image" );
				}
			}

			Vec6d poseEstimateHaar;
			Matx66d poseEstimateHaarUncertainty;

			Rect faceRegion;

			// The start place where CLM should start a search (or if it fails, can use the frame detection)
			if(!trackingInitialised || (!success && ( frameProc  % 5 == 0)))
			{
				// The tracker can return multiple head pose observation
				vector<Vec6d> poseEstimatesInitialiser;
				vector<Matx66d> covariancesInitialiser;			
				vector<Rect> regionsInitialiser;

				bool initSuccess = PoseDetectorHaar::InitialisePosesHaar(img_scaled, depth, poseEstimatesInitialiser, covariancesInitialiser, regionsInitialiser, classifier, fx, fy, cx, cy, haarParams);

				if(initSuccess)
				{
					if(poseEstimatesInitialiser.size() > 1)
					{
						cout << "ambiguous detection ";
						// keep the closest one (this is a hack for the experiment)
						double best = 10000;
						int bestIndex = -1;
						for( size_t i = 0; i < poseEstimatesInitialiser.size(); ++i)
						{
							cout << poseEstimatesInitialiser[i][2] << " ";
							if(poseEstimatesInitialiser[i][2] < best  && poseEstimatesInitialiser[i][2] > 200)
							{
								bestIndex = i;
								best = poseEstimatesInitialiser[i][2];
							}									
						}
						if(bestIndex != -1)
						{
							cout << endl << "Choosing " << poseEstimatesInitialiser[bestIndex][2] << regionsInitialiser[bestIndex].x << " " << regionsInitialiser[bestIndex].y <<  " " << regionsInitialiser[bestIndex].width << " " <<  regionsInitialiser[bestIndex].height << endl;
							faceRegion = regionsInitialiser[bestIndex];
						}
						else
						{
							initSuccess = false;
						}
					}
					else
					{	
						faceRegion = regionsInitialiser[0];
					}				

					facesInRow++;
				}
			}

			// If condition for tracking is met initialise the trackers
			if(!trackingInitialised && facesInRow >= 1)
			{			
				trackingInitialised = CLMWrapper::InitialiseCLM(img_scaled, depth, clmModel, poseEstimateHaar, faceRegion, fx, fy, cx, cy, clmParams);		
				facesInRow = 0;
			}		

			// opencv detector is needed here, if tracking failed reinitialise using it
			if(trackingInitialised)
			{
				success = CLMWrapper::TrackCLM(img_scaled, depth, clmModel, vector<Vec6d>(), vector<Matx66d>(), faceRegion, fx, fy, cx, cy, clmParams);								
			}			
			if(success)
			{			
				clmParams.wSizeCurrent = clmParams.wSizeSmall;
			}
			else
			{
				clmParams.wSizeCurrent = clmParams.wSizeInit;
			}

			poseEstimateCLM = CLMWrapper::GetPoseCLM(clmModel, fx, fy, cx, cy, clmParams);

			shape = clmModel._shape;
			//Use poseEstimateCLM for box
			int n = shape.rows / 2;
			int idx = clmModel._clm.GetViewIdx();
			for (int i = 0; i < n; ++i)
			{
				features[i] = Point((int)shape.at<double>(i), (int)shape.at<double>(i +n));
				cv::circle(disp, features[i], 1, Scalar(0,0,255), 2);
				if (clmModel._clm._visi[0][idx].at<int>(i)) visi[i] = true;
				else visi[i] = false;
			}
			DrawBox(disp, poseEstimateCLM, Scalar(0,0,255), 3, fx, fy, cx, cy);

			if(frameProc % 10 == 0)
			{      
				t1 = cv::getTickCount();
				fps = 10.0 / (double(t1-t0)/cv::getTickFrequency()); 
				t0 = t1;
			}

			frameProc++;
			Mat disprgb;			
			imshow("colour", disp);

			cvtColor(disp, disprgb, CV_RGB2BGR);
			resize(disprgb,disprgb,Size(500,400),0,0,INTER_NEAREST );


			char fpsC[255];
			_itoa((int)fps, fpsC, 10);
			string fpsSt("FPS:");
			fpsSt += fpsC;
			cv::putText(disprgb, fpsSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0));

			opencvImage = disprgb;

			if(disprgb.empty())
			{
				SHOWIMAGE = false;
			}
			else
			{
				SHOWIMAGE = true;
			}

			if(!depth.empty())
			{
				imshow("depth", depth/2000.0);
			}

			vCap >> img;

			if(!outvideos.empty())
			{		
				writerFace << disp;
			}

			// detect key presses
			char c = cv::waitKey(1);

			// key detections

			GRAYSCALE = false;

			if(quitmain==1){
				cout << "Quit." << endl;
				return;
			}

			// restart the tracker
			if(c == 'r')
			{
				trackingInitialised = false;
				facesInRow = 0;
			}
			if (!gotFace)
			{
				initImg = img.clone();
				extractFace(img);
				vCap.release();
				vCap = VideoCapture(device);
			}
			if (gotFace)
			{
				doTransformation();//fiddle with it
			}
		}	

		trackingInitialised = false;
		facesInRow = 0;
	}
}

int main (int argc, char **argv)
{
	omp_init_lock(&writelock);	
	glutInit(&argc, argv);
	doFaceTracking(argc, argv);
	omp_destroy_lock(&writelock);
	return 0;
}