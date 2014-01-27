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

#include <CLMTracker.h>
#include <PoseDetectorHaar.h>

#include <fstream>
#include <sstream>

#include <cv.h>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

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
Matx33f Euler2RotMat(const Vec3d& eulerAngles)
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

// Need to move this all to opengl
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


	Matx33f rot = Euler2RotMat(Vec3d(pose[3], pose[4], pose[5]));
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

int main (int argc, char **argv)
{

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
    haarParams.ClassifierLocation = "classifiers/haarcascade_frontalface_alt.xml";
	#else
		haarParams.ClassifierLocation = "classifiers/haarcascade_frontalface_alt.xml";
	#endif
		
	// Get the input output file parameters
	CLMWrapper::get_video_input_output_params(files, dDirs, outposes, outvideos, outfeatures, arguments);
	// Get camera parameters
	CLMWrapper::get_camera_params(fx, fy, cx, cy, dimx, dimy, arguments);    
	
	// The modules that are being used for tracking
	CLMTracker::TrackerCLM clmModel;	
	
	// Face detector initialisation
	CascadeClassifier classifier(haarParams.ClassifierLocation);
	if(classifier.empty())
	{
		string err = "Could not open a face detector at: " + haarParams.ClassifierLocation;
		FATAL_STREAM( err );
	}

	bool done = false;
	
	int f_n = -1;

	while(!done)
	{
		string file;

		// We might specify multiple video files as arguments
		if(files.size() > 0)
		{
			f_n++;			
		    file = files[f_n];
		}

		bool readDepth = !dDirs.empty();	

		// Do some grabbing
		VideoCapture vCap;
		if( file.size() > 0 )
		{
			INFO_STREAM( "Attempting to read from file: " << file );
			vCap = VideoCapture( file );
		}
		else
		{
			INFO_STREAM( "Attempting to capture from device: " << device );
			vCap = VideoCapture( device );
		}

		if( !vCap.isOpened() ) FATAL_STREAM( "Failed to open video source" );
		else INFO_STREAM( "Device or file opened");

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
	
		// Creating output files
		std::ofstream posesFile;
		if(!outposes.empty())
		{
			posesFile.open (outposes[f_n]);
		}
	
		std::ofstream featuresFile;		
		if(!outfeatures.empty())
		{
			featuresFile.open(outfeatures[f_n]);
		}
	
		int frameProc = 0;

		// faces in a row detected
		int facesInRow = 0;

		// saving the videos
		VideoWriter writerFace;
		if(!outvideos.empty())
		{
			writerFace = VideoWriter(outvideos[f_n], CV_FOURCC('D','I','V','X'), 30, img.size(), true);		
		}

		// Variables useful for the tracking itself
		bool success = false;
		bool trackingInitialised = false;
	
		// For measuring the timings
		int64 t1,t0 = cv::getTickCount();
		double fps = 10;

		Mat disp;

		INFO_STREAM( "Starting tracking");
		while(!img.empty())
		{		

			Mat_<float> depth;
			Mat_<uchar> gray;
			cvtColor(img, gray, CV_BGR2GRAY);
		
			// Don't resize if it's unneeded
			Mat_<uchar> img_scaled;		
			if(dimx != gray.cols || dimy != gray.rows)
			{
				resize( gray, img_scaled, Size( dimx, dimy ) );
				resize(img, disp, Size( dimx, dimy));
			}
			else
			{
				img_scaled = gray;
				disp = img.clone();
			}
		
			namedWindow("colour",1);

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
						resize(dImg, dImgT, Size( dimx, dimy));
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
			if(!trackingInitialised || (!success && ( frameProc  % 2 == 0)))
			{
				INFO_STREAM( "Attempting to initialise a face");
				// The tracker can return multiple head pose observation
				vector<Vec6d> poseEstimatesInitialiser;
				vector<Matx66d> covariancesInitialiser;			
				vector<Rect> regionsInitialiser;

				bool initSuccess = PoseDetectorHaar::InitialisePosesHaar(img_scaled, depth, poseEstimatesInitialiser, covariancesInitialiser, regionsInitialiser, classifier, fx, fy, cx, cy, haarParams);
					
				if(initSuccess)
				{
					INFO_STREAM( "Face(s) detected");
					if(poseEstimatesInitialiser.size() > 1)
					{
						cout << "ambiguous detection: " << endl;
						// keep the closest one (this is a hack for the experiment)
						double best = 10000;
						int bestIndex = -1;
						for( size_t i = 0; i < poseEstimatesInitialiser.size(); ++i)
						{
							cout << regionsInitialiser[i].x << " " << regionsInitialiser[i].y <<  " " << regionsInitialiser[i].width << " " <<  regionsInitialiser[i].height << endl;
							if(poseEstimatesInitialiser[i][2] < best  && poseEstimatesInitialiser[i][2] > 100)
							{
								bestIndex = i;
								best = poseEstimatesInitialiser[i][2];
							}									
						}
						if(bestIndex != -1)
						{
							cout << "Choosing bbox:" << regionsInitialiser[bestIndex].x << " " << regionsInitialiser[bestIndex].y <<  " " << regionsInitialiser[bestIndex].width << " " <<  regionsInitialiser[bestIndex].height << endl;
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
				INFO_STREAM( "Initialising CLM");
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

			Vec6d poseEstimateCLM = CLMWrapper::GetPoseCLM(clmModel, fx, fy, cx, cy, clmParams);

			if(!outfeatures.empty())
			{
				featuresFile << frameProc + 1 << " " << success;
				for (int i = 0; i < 66 * 2; ++ i)
				{
					featuresFile << " " << clmModel._shape.at<double>(i) << endl;
				}
				featuresFile << endl;
			}

			if(!outposes.empty())
			{
				posesFile << frameProc + 1 << " " << (float)frameProc * 1000/30 << " " << 1 << " " << poseEstimateCLM[0] << " " << poseEstimateCLM[1] << " " << poseEstimateCLM[2] << " " << poseEstimateCLM[3] << " " << poseEstimateCLM[4] << " " << poseEstimateCLM[5] << endl;
			}										
	
			if(success)			
			{
				int idx = clmModel._clm.GetViewIdx(); 	

				// drawing the facial features on the face if tracking is successful
				clmModel._clm._pdm.Draw(disp, clmModel._shape, clmModel._clm._triangulations[idx], clmModel._clm._visi[0][idx]);

				DrawBox(disp, poseEstimateCLM, Scalar(255,0,0), 3, fx, fy, cx, cy);			
			}
			else if(!clmModel._clm._pglobl.empty())
			{			
				int idx = clmModel._clm.GetViewIdx(); 	
			
				// draw the facial features
				clmModel._clm._pdm.Draw(disp, clmModel._shape, clmModel._clm._triangulations[idx], clmModel._clm._visi[0][idx]);

				// if tracking fails draw a red outline
				DrawBox(disp, poseEstimateCLM, Scalar(0,0,255), 3, fx, fy, cx, cy);	
			}
			if(frameProc % 10 == 0)
			{      
				t1 = cv::getTickCount();
				fps = 10.0 / (double(t1-t0)/cv::getTickFrequency()); 
				t0 = t1;
			}

			char fpsC[255];
			sprintf(fpsC, "%d", (int)fps);
			string fpsSt("FPS:");
			fpsSt += fpsC;
			cv::putText(disp, fpsSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0));
		
			frameProc++;
						
			imshow("colour", disp);
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
			char c = cv::waitKey(2);

			// key detections

			// restart the tracker
			if(c == 'r')
			{
				trackingInitialised = false;
				facesInRow = 0;
			}
			// quit the application
			else if(c=='q')
			{
				return(0);
			}


		}
		
		trackingInitialised = false;
		facesInRow = 0;

		posesFile.close();

		// break out of the loop if done with all the files
		if(f_n == files.size() -1)
		{
			done = true;
		}
	}

	return 0;
}

