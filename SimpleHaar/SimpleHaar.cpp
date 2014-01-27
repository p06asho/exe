#include <iostream>
#include <conio.h>

#include <PoseDetectorHaar.h>
#include <PoseDetectorHaarParameters.h>

#include <iostream>
#include <fstream>

#include <cxcore.h>
#include <highgui.h>

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

void get_image_input_output_params(vector<string> &input_image_files, vector<string> &output_detection_files, bool &verbose, vector<string> &arguments)
{
	bool* valid = new bool[arguments.size()];

	for(size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-f") == 0) 
		{                    
			input_image_files.push_back(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
		else if (arguments[i].compare("-of") == 0)
		{
			output_detection_files.push_back(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;
			i++;
		} 
		else if (arguments[i].compare("-v") == 0)
		{
			verbose = true;
			valid[i] = false;
		} 
		else if (arguments[i].compare("-help") == 0)
		{
			cout << "Input output files are defined as: -f <infile (can have multiple ones)> -of <outdetections(can have multiple ones)>\n"; // Inform the user of how to use the program				
		}
	}

	// remove used up arguments
	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}

}
// -f "file.avi/file.png" -of -outputFolder/outputFile
int main (int argc, char **argv)
{
	// Some initial parameters that can be overriden from command line
	vector<string> inFiles, outFiles;

	// by default use the fast version of parameters
	PoseDetectorHaar::PoseDetectorHaarParameters haarParams;

	haarParams.ClassifierLocation = "classifiers/haarcascade_frontalface_alt.xml";
	bool verbose = false;
	
	//Convert arguments to more convenient vector form
	vector<string> arguments = get_arguments(argc, argv);

	get_image_input_output_params(inFiles, outFiles, verbose, arguments);

	std::vector<string> classifierLocations;

	classifierLocations.push_back(haarParams.ClassifierLocation);

	CascadeClassifier classifier(haarParams.ClassifierLocation);

	for( int src = 0; src < inFiles.size(); ++ src)
	{
		std::ofstream detectedRegion(outFiles.at(src));

		// Try opening as an image first
		Mat image = imread(inFiles.at(src), - 1);

		// type 0 is frontal 1 is left profile 2 is right profile
		int type = 0;

		if(!image.empty())
		{		
			Mat_<uchar> gray;
			if(image.channels() == 3)
			{
				cvtColor(image, gray, CV_BGR2GRAY);
			}
			else
			{
				gray = image;
			}			

			for(int c = 0; c < classifierLocations.size(); ++c)
			{

				if( c == 3)
					cv::flip(gray.clone(), gray, 1);

				classifier.load(classifierLocations.at(c));

				std::vector<Rect> regions;
				classifier.detectMultiScale(gray, regions, 1.2,2,CV_HAAR_DO_CANNY_PRUNING, Size(40, 40));


				if(regions.size() > 0)
				{
					Rect biggestRegion = regions.front();

					// write out the biggest region
					for(size_t i = 1; i < regions.size(); ++i)
					{
						if(regions[i].width > biggestRegion.width)
						{
							biggestRegion = regions[i];
						}
					}
					if(verbose)
					{
						cv::rectangle(gray, biggestRegion, Scalar(1.0,0.0,0.0), 2);
					}
					//detectedRegion << biggestRegion.x << " " << biggestRegion.y << " " << biggestRegion.width << " " << biggestRegion.height << " " << type << endl;
					detectedRegion << biggestRegion.x << " " << biggestRegion.y << " " << biggestRegion.width << " " << biggestRegion.height << endl;
					break;
					//cout << "detected" << endl;
				}
				if(verbose)
				{
					imshow("img", gray);
					cv::waitKey(10);
				}
			}			
			detectedRegion.close();
		}
		else
		{
			// Try to open the video
			VideoCapture vCap(inFiles.at(src));
			int frame = 0;
			if(vCap.isOpened())	
			{
				Mat img;
				vCap >> img;

				while(!img.empty())
				{		

					Mat_<uchar> gray;
					cvtColor(img, gray, CV_BGR2GRAY);

					std::vector<Rect> regions;

					classifier.detectMultiScale(gray, regions, 1.2,2,CV_HAAR_DO_CANNY_PRUNING, Size(40, 40)); 

					if(regions.size() > 0)
					{
						Rect biggestRegion = regions.front();

						// write out the biggest region
						for(int i = 1; i < regions.size(); ++i)
						{
							if(regions[i].width > biggestRegion.width)
							{
								biggestRegion = regions[i];
							}
						}
						detectedRegion << frame << " " << biggestRegion.x << " " << biggestRegion.y << " " << biggestRegion.width << " " << biggestRegion.height << endl;
						if(verbose)
						{
							cv::rectangle(img, biggestRegion, Scalar(1.0,0.0,0.0), 2);
						}
					}
					else
					{
						detectedRegion << frame << " " << 0 << " " << 0 << " " << 0 << " " << 0 << endl;
					}

					if(verbose)
					{
						imshow("img", img);
						cv::waitKey(10);
					}

					vCap >> img;
					frame++;
				}

				detectedRegion.close();
			}
			else
			{
				cout << "Failed to open image or video: " << inFiles.at(src) << endl;
			}
		}
	}
	return 0;
}

