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
Mat background;

bool GETFACE = false;		//get a new face
bool SHOWIMAGE = false;		//show the webcam image (in the main window). Turned on when the matrix isn't empty


bool gotFace = false;
Vec6d poseEstimateCLM;
bool visi [66];
Mat shape;
Point features[66];
Mat face;
Mat warpedFace;
vector<Vec6f> triangles;
bool gotContext;
//GLFWwindow* window;
int mainargc;
char **mainargv;
int GLWindowID;
GLuint textureID;



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

// Move all of this to OpenGL?
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
	// Generate a number for our textureID's unique handle
	GLuint textureID;
	glGenTextures(1, &textureID);
 
	// Bind to our texture handle
	glBindTexture(GL_TEXTURE_2D, textureID);
 
	// Catch silly-mistake texture interpolation method for magnification
	if (magFilter == GL_LINEAR_MIPMAP_LINEAR  ||
	    magFilter == GL_LINEAR_MIPMAP_NEAREST ||
	    magFilter == GL_NEAREST_MIPMAP_LINEAR ||
	    magFilter == GL_NEAREST_MIPMAP_NEAREST)
	{
		cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << endl;
		magFilter = GL_LINEAR;
	}
 
	// Set texture interpolation methods for minification and magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
 
	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);
 
	// Set incoming texture format to:
	// GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
	// GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
	// Work out other mappings as required ( there's a list in comments in main() )
	GLenum inputColourFormat = GL_BGR;
	if (mat.channels() == 1)
	{
		inputColourFormat = GL_LUMINANCE;
	}
 
	// Create the texture
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
	             0,                 // Pyramid level (for mip-mapping) - 0 is the top level
	             GL_RGB,            // Internal colour format to convert to
	             mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
	             mat.rows,          // Image height i.e. 480 for Kinect in standard mode
	             0,                 // Border width in pixels (can either be 1 or 0)
	             inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
	             GL_UNSIGNED_BYTE,  // Image data type
	             mat.ptr());        // The actual image data itself
 
	// If we're using mipmaps then generate them. Note: This requires OpenGL 3.0 or higher
	if (minFilter == GL_LINEAR_MIPMAP_LINEAR  ||
	    minFilter == GL_LINEAR_MIPMAP_NEAREST ||
	    minFilter == GL_NEAREST_MIPMAP_LINEAR ||
	    minFilter == GL_NEAREST_MIPMAP_NEAREST)
	{
		glGenerateMipmap(GL_TEXTURE_2D);
	}
 
	return textureID;
}

void doTransformation(Mat img, int argc, char **argv)
{
	//Don't get new triangulations every time. Use the original set of triangles, and apply it like a texture
	warpedFace = Mat(img.rows, img.cols, img.type());	
	if (!gotContext)
	{
		//glfwInit();
		//window = glfwCreateWindow(640, 480, "Output", NULL, NULL);
		//glfwMakeContextCurrent(window);
		//glutInit(&argc, argv);
		glutInitWindowSize(640, 480);
		glutInitWindowPosition(0,0);
		GLWindowID = glutCreateWindow("Output");
		glutSetWindow(GLWindowID);
		gotContext = true;
		textureID = matToTexture(img, GL_NEAREST, GL_NEAREST, GL_CLAMP);
	}
	//Rendering stuff here. Remember to swap buffers each time. 
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glBegin(GL_QUADS);
		glTexCoord2f(0.0, 0.0);
		glTexCoord2f(0.0, 1.0);
		glTexCoord2f(1.0, 0.0);
		glTexCoord2f(1.0, 1.0);
	glEnd();
	glutSwapBuffers();
}

void extractFace(Mat img)
{
	//Get a (Delaunay) triangulation from the feature points
	Rect rect(0,0,img.cols, img.rows);
	cv::Subdiv2D subdiv(rect);
	for (int i = 0; i < 66; i++)
	{
		subdiv.insert(features[i]);
	}
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);
	Mat tri = img.clone();
	for (int i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		bool draw = true;
		for (int i = 0; i < 3; i++)
		{
			if (pt[i].x < 0 || pt[i].x >= tri.cols || pt[i].y < 0 || pt[i].y >= tri.rows)
			{
				draw = false;
			}
		}
		if (draw)
		{
			cv::line(tri, pt[0], pt[1], 255);
			cv::line(tri, pt[1], pt[2], 255);
			cv::line(tri, pt[2], pt[0], 255);
			triangles.push_back(t);
		}
	}
	namedWindow("triangulated", 1);
	imshow("triangulated", tri);

	//Use matrix transformations to get a texture in direct, 'face-on' orientation. 
	//In this orientation, the face should be roughly symmetric - get distances between specific points in different axes and stretch/compress parts of the image accordingly. 
	//Then rotate the whole thing in 2d to be 'right side up'


	gotFace = true;//Don't do it every frame
}

void extractFaceOld(Mat img)
{
	if (gotFace) return;//don't need to grab it once we already have it
	else
	{
		//Get one!
		//join the dots
		Mat ccut(img.rows, img.cols, img.type());
		face = Mat(img.rows, img.cols, img.type());
		for (int i = 0; i < 16; i++) //perimeter
		{
			cv::line(ccut, features[i], features[i+1], 255, 2);
		}
		cv::line(ccut, features[16], features[26], 255, 2);
		for (int i = 26; i > 17; i--) //eyebrows
		{
			cv::line(ccut, features[i], features[i-1], 255, 2);
		}
		cv::line(ccut, features[17], features[0], 255, 2);

		namedWindow("cutter", 1);
		imshow("cutter", ccut);		
		
		//scanline: problem here
		int firsts [480];
		int lasts [480];
		for (int i = 0; i < 480; i++)
		{
			firsts[i] = 0;
			lasts[i] = 0;
		}
		for (int i = 0; i < img.rows; i++)//for each row
		{
			bool found = false;
			for (int j = 0; j < img.cols; j++)//along the row
			{				
				if (!found && (ccut.at<int>(i,j) == 255))
				{
					firsts[i] = j;//point to include from
					found = true; //we're no longer looking for a first!
				}
				if (found && (ccut.at<int>(i,j) == 255))
				{
					lasts[i] = j;//point to include until
				}
			}
		}
		for (int i = 0; i < ccut.rows; i++)
		{
			if (!(lasts[i] == 0))
			{
				for (int j = firsts[i]; j <= lasts[i]; j++)
				{
					face.at<double>(i,j) = img.at<double>(i,j);
				}
			}
		}

		gotFace = true;//Don't repeat the work
		namedWindow("face", 1);
		imshow("face", face);
	}
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
				cout << "Not using Webcam. " << endl;
				//FATAL_STREAM( "Failed to open video source" );
			}
		}

		// Do some grabbing
		if( !USEWEBCAM)
		{

			if(file.size() > 0 ){
				INFO_STREAM( "Attempting to read from file: " << file );
				vCap = VideoCapture( file );
			}
			else {
				INFO_STREAM("No file specified. Please use webcam or load file manually");
				USEWEBCAM = 1;

			}
		}



		Mat img;
		vCap >> img;


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



		while(!img.empty() && !CHANGESOURCE && !done)						//This is where stuff happens once the file's open.
		{		
			//for constant-size input:
			//resize(img, img, Size( dimx, dimy));

			Mat_<float> depth;
			Mat_<uchar> gray;
			cvtColor(img, gray, CV_BGR2GRAY);
			cvtColor(img, rgbimg, CV_BGR2RGB);

			if(GRAYSCALE){
				cvtColor(gray, rgbimg, CV_GRAY2RGB);
			}

			parsecolour(rgbimg);			//this sends the rgb image to the PAW loop

			writeToFile = 0;

			if(GETFACE){
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

			// Changes for no reinit version
			//success = true;
			//clmParams.wSizeCurrent = clmParams.wSizeInit;

			poseEstimateCLM = CLMWrapper::GetPoseCLM(clmModel, fx, fy, cx, cy, clmParams);

		/*	if(success)			
			{
				int idx = clmModel._clm.GetViewIdx(); 	

				// drawing the facial features on the face if tracking is successful
				clmModel._clm._pdm.Draw(disp, clmModel._shape, clmModel._clm._triangulations[idx], clmModel._clm._visi[0][idx]);


				//cout << clmModel._clm.shape << endl;
				//cv::imshow("other", clmModel._shape);

				DrawBox(disp, poseEstimateCLM, Scalar(255,0,0), 3, fx, fy, cx, cy);			
			}
			else if(!clmModel._clm._pglobl.empty())
			{			
				int idx = clmModel._clm.GetViewIdx(); 	

				// draw the facial features
				clmModel._clm._pdm.Draw(disp, clmModel._shape, clmModel._clm._triangulations[idx], clmModel._clm._visi[0][idx]);

				// if tracking fails draw a red outline
				DrawBox(disp, poseEstimateCLM, Scalar(0,0,255), 3, fx, fy, cx, cy);	
			}*/

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

			if(disprgb.empty()){
				SHOWIMAGE = false;
			}
			else{
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
				extractFace(img);//get the part of the image we want to transform
			}
			else
			{
				doTransformation(img, argc, argv);//fiddle with it
			}
//			namedWindow("face", 1);
//			imshow("face", face);
		}	

		trackingInitialised = false;
		facesInRow = 0;

		// break out of the loop if done with all the files
		if(f_n == files.size() -1)
		{
		//	if( waitKey( 5000 ) >= 0 )
		//    			done = true;
		}
	}
	/*glfwDestroyWindow(window);
	glfwTerminate();*/
}

int main (int argc, char **argv)
{
	omp_init_lock(&writelock);	
	glutInit(&argc, argv);
	doFaceTracking(argc, argv);
	omp_destroy_lock(&writelock);
	return 0;
}