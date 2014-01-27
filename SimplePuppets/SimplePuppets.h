
#ifndef __SIMPLEPUPPETS_h_
#define __SIMPLEPUPPETS_h_



#include <CLMTracker.h>
#include <PoseDetectorHaar.h>
#include <Avatar.h>
#include <PAW.h>
#include <CLM.h>
#include <fstream>
#include <sstream>

#include <cv.h>

  int mindreadervideo = -1;

  void readFromStock(int c);

bool writeToFile = 0;
bool ERIon = 0;
bool quitmain = 0;
string choiceavatar = "0";
bool GRAYSCALE = false;

int option, oldoption;

int facesInRow = 0;
bool trackingInitialised;

string inputfile;
bool NEWFILE = false;
VideoCapture vCap;

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl


IplImage opencvImage;
bool USEWEBCAM = true;
bool CHANGESOURCE = false;
bool PAWREADAGAIN = false;
bool PAWREADNEXTTIME = false;



void use_webcam();
static void printErrorAndAbort( const std::string & error );
Matx33f Euler2RotationMatrix(const Vec3d& eulerAngles);
void Project(Mat_<float>& dest, const Mat_<float>& mesh, Size size, double fx, double fy, double cx, double cy);
void DrawBox(Mat image, Vec6d pose, Scalar color, int thickness, float fx, float fy, float cx, float cy);
vector<string> get_arguments(int argc, char **argv);




void doFaceTracking(int argc, char **argv);
void startGTK(int argc, char **argv);
int main (int argc, char **argv);
void Puppets();




#endif
