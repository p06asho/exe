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

#include <PoseDetectorHaar.h>
#include <PoseDetectorHaarParameters.h>
#include <CLMTracker.h>

#include <fstream>

#include <cxcore.h>
#include <highgui.h>

#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

using namespace std;
using namespace cv;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 1; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

void convert_to_grayscale(const Mat& in, Mat& out)
{
	if(in.channels() == 3)
	{
		// Make sure it's in a correct format
		if(in.depth() != CV_8U)
		{
			if(in.depth() == CV_16U)
			{
				Mat tmp = in / 256;
				tmp.convertTo(tmp, CV_8U);
				cvtColor(tmp, out, CV_BGR2GRAY);
			}
		}
		else
		{
			cvtColor(in, out, CV_BGR2GRAY);
		}
	}
	else
	{
		if(in.depth() == CV_16U)
		{
			Mat tmp = in / 256;
			out = tmp.clone();
		}
		else
		{
			out = in.clone();
		}
	}
}

void write_out_landmarks(const string& outfeatures, const TrackerCLM& clm_model)
{

	std::ofstream featuresFile;
	featuresFile.open(outfeatures);		

	if(featuresFile.is_open())
	{	
		int n = clm_model._clm._visi[0][0].rows;
		featuresFile << "version: 1" << endl;
		featuresFile << "npoints: " << n << endl;
		featuresFile << "{" << endl;
		
		for (int i = 0; i < n; ++ i)
		{
			// Use matlab format, so + 1
			featuresFile << clm_model._shape.at<double>(i) + 2 << " " << clm_model._shape.at<double>(i+n) + 2 << endl;
		}
		featuresFile << "}" << endl;			
		featuresFile.close();
	}
}

void create_display_image(const Mat& orig, Mat& disp, TrackerCLM& clm_model)
{
	
	// preparing the visualisation image
	disp = orig.clone();		

	// Creating a display image
	int idx = clm_model._clm.GetViewIdx();
			
	Mat xs = clm_model._shape(Rect(0, 0, 1, clm_model._shape.rows/2));
	Mat ys = clm_model._shape(Rect(0, clm_model._shape.rows/2, 1, clm_model._shape.rows/2));
	double minX, maxX, minY, maxY;

	cv::minMaxLoc(xs, &minX, &maxX);
	cv::minMaxLoc(ys, &minY, &maxY);

	double width = maxX - minX;
	double height = maxY - minY;

	int minCropX = max((int)(minX-width/3.0),0);
	int minCropY = max((int)(minY-height/3.0),0);

	int widthCrop = min((int)(width*5.0/3.0), disp.cols - minCropX - 1);
	int heightCrop = min((int)(height*5.0/3.0), disp.rows - minCropY - 1);

	double scaling = 350.0/widthCrop;
	
	// first crop the image
	disp = disp(Rect((int)(minCropX), (int)(minCropY), (int)(widthCrop), (int)(heightCrop)));
		
	// now scale it
	cv::resize(disp.clone(), disp, Size(), scaling, scaling);

	// Make the adjustments to points
	xs = (xs - minCropX)*scaling;
	ys = (ys - minCropY)*scaling;

	Mat shape = clm_model._shape.clone();

	xs.copyTo(shape(Rect(0, 0, 1, clm_model._shape.rows/2)));
	ys.copyTo(shape(Rect(0, clm_model._shape.rows/2, 1, clm_model._shape.rows/2)));

	clm_model._clm._pdm.Draw(disp, shape, clm_model._clm._triangulations[0], clm_model._clm._visi[0][idx]);
						
}

int main (int argc, char **argv)
{

	//Convert arguments to more convenient vector form
	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line
	vector<string> files, dFiles, outimages, outfeaturess;

	// these can be overriden using -cx, -cy, -fx, -fy, -dimx, -dimy flags
	float fx = 500, fy = 500, cx = 0, cy = 0;
	int dimx = 0, dimy = 0;
	
	// initial translation, rotation and scale (can be specified by the user)
	vector<Vec6d> initial_poses;

	bool useCLMTracker = true;

	// Get camera parameters
	CLMWrapper::get_camera_params(fx, fy, cx, cy, dimx, dimy, arguments);
	CLMWrapper::get_image_input_output_params(files, dFiles, outfeaturess, outimages, initial_poses, arguments);	
	CLMWrapper::CLMParameters clmParams(arguments);	

	// The modules that are being used for tracking
	CLMTracker::TrackerCLM clmModel;

	cout << "Loading the model" << endl;
	clmModel.Read(clmParams.defaultModelLoc, clmParams.override_pdm_loc);
	cout << "Model loaded" << endl;

	PoseDetectorHaar::PoseDetectorHaarParameters haarParams;

	haarParams.ClassifierLocation = "classifiers/haarcascade_frontalface_alt2.xml";

	CascadeClassifier classifier(haarParams.ClassifierLocation);

	bool visualise = true;

	//clmParams.multi_view = true;

	// Do some image loading
	for(size_t i = 0; i < files.size(); i++)
	{
		string file = files.at(i);

		// Loading image
		Mat img = imread(file, -1);
		
		// Loading depth file if exists
		Mat dTemp;		
		Mat_<float> dImg;

		if(dFiles.size() > 0)
		{
			string dFile = dFiles.at(i);
			dTemp = imread(dFile, -1);
			dTemp.convertTo(dImg, CV_32F);
		}

		if(dimx != 0)
		{
			cv::resize(img.clone(), img, cv::Size(dimx, dimy));
			if(!dImg.empty())
			{
				cv::resize(dImg.clone(), dImg, cv::Size(dimx, dimy));
			}
		}

		bool trackingInitialised = false;
	
		// Making sure the image is in uchar grayscale
		Mat_<uchar> gray;		
		convert_to_grayscale(img, gray);
		
			
		// Face detection initialisation
		Vec6d poseEstimateHaar;
		Matx66d poseEstimateHaarUncertainty;

		vector<Vec6d> poseEstimatesInitialiser;
		vector<Matx66d> covariancesInitialiser;	
		vector<Rect> faceRegions;
	
		bool initSuccess = false;

		// if no pose defined we just use OpenCV
		if(initial_poses.empty())
		{
			initSuccess = PoseDetectorHaar::InitialisePosesHaar(gray, dImg, poseEstimatesInitialiser, covariancesInitialiser, faceRegions, classifier, fx, fy, cx, cy, haarParams);

			if(initSuccess)
			{
				if(poseEstimatesInitialiser.size() > 1)
				{
					cout << "Multiple faces detected" << endl;
				}
			}
		
			if(initSuccess)
			{
				// perform landmark detection for every face detected
				for(int r=0; r < faceRegions.size(); ++r)
				{
					if(!clmParams.multi_view)
					{
						Vec6d pose(0.0);
						bool success = CLMWrapper::InitialiseCLM(gray, dImg, clmModel, pose, faceRegions[r], fx, fy, cx, cy, clmParams);
					}
					else
					{
						vector<Vec3d> orientations;

						// Have multiple hypotheses, check which one is best and keep it
						orientations.push_back(Vec3d(0,0,0));
						orientations.push_back(Vec3d(0,0.5236,0));
						orientations.push_back(Vec3d(0,-0.5236,0));
						orientations.push_back(Vec3d(0.5236,0,0));
						orientations.push_back(Vec3d(-0.5236,0,0));
				
						double best_lhood;
						Mat best_shape;

						for (size_t p = 0; p<orientations.size(); ++p)
						{

							Vec6d pose;		

							pose(1) = orientations[p][0];
							pose(2) = orientations[p][1];
							pose(3) = orientations[p][2];
					
							CLMWrapper::InitialiseCLM(gray, dImg, clmModel, pose, faceRegions[r], fx, fy, cx, cy, clmParams);

							double curr_lhood = clmModel._clm._likelihood;

							if(p==0 || curr_lhood > best_lhood)
							{
								best_lhood = curr_lhood;
								best_shape = clmModel._shape.clone();
							}
						}

						clmModel._shape = best_shape;
						clmModel._clm._likelihood = best_lhood;
					}

					// Writing out the detected landmarks
					if(!outfeaturess.empty())
					{
						char name[100];
						sprintf(name, "_det_%d", r);

						boost::filesystem::path slash("/");
						std::string preferredSlash = slash.make_preferred().string();

						// append detection number
						boost::filesystem::path out_feat_path(outfeaturess.at(i));
						boost::filesystem::path dir = out_feat_path.parent_path();
						boost::filesystem::path fname = out_feat_path.filename().replace_extension("");
						boost::filesystem::path ext = out_feat_path.extension();
						string outfeatures = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();
						write_out_landmarks(outfeatures, clmModel);
					}

					// displaying detected stuff
					Mat disp;
					create_display_image(img, disp, clmModel);

					if(visualise)
					{
						// For debug purposes show final likelihood
						//std::ostringstream s;
						//s << clmModel._clm._likelihood;
						//string lhoodSt("Lhood:");
						//lhoodSt += s.str();
						//cv::putText(disp, lhoodSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0));
						imshow("colour", disp);
						cv::waitKey(10);
					}

					if(!outimages.empty())
					{
						string outimage = outimages.at(i);
						if(!outimage.empty())
						{
							char name[100];
							sprintf(name, "_det_%d", r);

							boost::filesystem::path slash("/");
							std::string preferredSlash = slash.make_preferred().string();

							// append detection number
							boost::filesystem::path out_feat_path(outimage);
							boost::filesystem::path dir = out_feat_path.parent_path();
							boost::filesystem::path fname = out_feat_path.filename().replace_extension("");
							boost::filesystem::path ext = out_feat_path.extension();
							outimage = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();

							imwrite(outimage, disp);	
						}
					}


				}
			}
			else
			{
				cout << "No faces detected" << endl;
				continue;
			}
		}
		else
		{
			initSuccess = true;

			// if pose defined we don't need the conventional initialisation			
			if(!clmParams.multi_view)
			{
				clmModel._clm._pglobl = Mat(initial_poses[i]);			

				Vec6d pose = CLMWrapper::GetPoseCLM(clmModel, dImg, fx, fy, cx, cy, clmParams);

				bool success = CLMWrapper::InitialiseCLM(gray, dImg, clmModel, pose, Rect(), fx, fy, cx, cy, clmParams);
			}
			else
			{
				vector<Vec3d> orientations;

				// Have multiple hypotheses, check which one is best and keep it
				orientations.push_back(Vec3d(0,0,0));
				orientations.push_back(Vec3d(0,0.5236,0));
				orientations.push_back(Vec3d(0,-0.5236,0));
				orientations.push_back(Vec3d(0.5236,0,0));
				orientations.push_back(Vec3d(-0.5236,0,0));
				
				double best_lhood;
				Mat best_shape;

				for (size_t p = 0; p<orientations.size(); ++p)
				{

					clmModel._clm._pglobl = Mat(initial_poses[i]);		

					clmModel._clm._pglobl.at<double>(1) = orientations[p][0];
					clmModel._clm._pglobl.at<double>(2) = orientations[p][1];
					clmModel._clm._pglobl.at<double>(3) = orientations[p][2];
					
					Vec6d pose = CLMWrapper::GetPoseCLM(clmModel, dImg, fx, fy, cx, cy, clmParams);

					CLMWrapper::InitialiseCLM(gray, dImg, clmModel, pose, Rect(), fx, fy, cx, cy, clmParams);

					double curr_lhood = clmModel._clm._likelihood;

					if(p==0 || curr_lhood > best_lhood)
					{
						best_lhood = curr_lhood;
						best_shape = clmModel._shape.clone();
					}
				}

				clmModel._shape = best_shape;

			}

			// Writing out the detected landmarks
			if(!outfeaturess.empty())
			{
				string outfeatures = outfeaturess.at(i);
				write_out_landmarks(outfeatures, clmModel);
			}

			// displaying detected stuff
			Mat disp;
			create_display_image(img, disp, clmModel);

			if(visualise)
			{
				imshow("colour", disp);
				cv::waitKey(5);
			}

			if(!outimages.empty())
			{
				string outimage = outimages.at(i);
				if(!outimage.empty())
				{
					imwrite(outimage, disp);	
				}
			}


		}				

		// Reset the parameters if more images are coming
		if(files.size() > 0)
		{
			clmModel._clm._plocal.setTo(0);
		}

	}
	return 0;
}

