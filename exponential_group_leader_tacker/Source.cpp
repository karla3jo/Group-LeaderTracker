////////////////////////////////////////////////////////////////////////////////////////
////		    GROUP LEADER DETECTOR AND TRACKER by KARLA TREJO	            ////
////										    ////
////	National Insitute of Informatics - Universitat Politècnica de Catalunya	    ////
////				July, 2016. Tokyo, Japan.			    ////
////										    ////
////////////////////////////////////////////////////////////////////////////////////////

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/mat.hpp"

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <iostream>
#include <vector>
#include <Windows.h>
#include <cmath>
#include <sstream>
#include <fstream>
#include <algorithm> 

#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_transforms.h>

using namespace cv;
using namespace std;
using namespace dlib;

// Simple 2D vector class
struct Vec2D
{
	int min, max;
};

// 2D axially-aligned bounding box.
struct Box2D
{
	Vec2D x, y;
}a, b, c;

bool BoxesIntersect(const Box2D &a, const Box2D &b)
{
	if (a.x.max < b.x.min) return false; // a is left of b
	if (a.x.min > b.x.max) return false; // a is right of b
	if (a.y.max < b.y.min) return false; // a is above b
	if (a.y.min > b.y.max) return false; // a is below b
	return true; // boxes overlap
}


int main(int argc, char* argv[])
{
	if (argc < 2){
		std::cerr << "Usage: " << argv[0] << " 'Video_Name.avi'" << std::endl;
		return 1;
	}

	cv::Mat img;			//Image
	cv::Mat cpy_img;		//Copy of Image
	cv::Mat croppedImage;		//Cropped Image to optimize calculations of Optical Flow
	int counter_img = 0;		//Number of frames
	int distances[9][15] = { 0 };	//Distances matrix, array accumulating Delta values of every detection in the frames (Rows:DiffXY,CoordXY & MemberFlag, Cols:Detections)
	int matrix[5][15] = { 0 };	//Copy of Distances Matrix to detect unchaged values (False Positives detections) that could be eliminated
	int mat_i = 0;			//Columns-pointer for Distances Matrix 
	int value_x = 0;		//Current X value, left coordinate of the bounding box provided by OpenCV's People Detector
	int compare_x = 0;		//Difference between value_x and a X value stored in Distances Matrix from previous detections
	int save_loc_2 = 0;		//Crucial location (ID) of a X value
	int ii = 0;			//Row-pointer for accesing arrays of information
	int jj = 0;			//Column-pointer for accessing arrays of information
	int center_x = 0;		//Center X coordinate of leader's location (info required by Dlib's tracker)
	int center_y = 0;		//Center Y coordinate of leader's location (info required by Dlib's tracker)
	int leader_width = 0;		//Width of leader's bounding box (info required for Dlib Tracker) and a constant value for ROI's sizes and Optical Flow inputs
	int leader_height = 0;		//Height of leader's bounding box (info required for Dlib Tracker) and a constant value for ROI's sizes and Optical Flow inputs
	int d = 0;			//First detection comparisson flag
	int rem = 0;			//Remainder of the division that triggers several procedures every 25 frames
	int flag_track = 0;		//Flag to trigger the tracker's update over the frames
	int rightc[15] = { 0 };		//Vector storing the right corner coordinate from the detection bounding boxes
	int m = 0;			//Multipurpose variable for counting tasks
	int thresh = 8;			//Thresholding to filter outrageous sizes coming from possible erroneous coordinates of bounding boxes
	float opt_flows[2][1] = { 0 };	//Array computing the sum of optical flows for later average every 25 frames
	cv::Mat flow, cflow, prevgray;	//Optical flow image arrays
	cv::Mat prevframe, currframe, nextframe; //Motion detection image arrays
	int flag_member = 0;		//Flag for setting Membership Values 
	int id_member = 0;		//Saves position in the array for the current detection
	int unch = 0;			//Flag to compare Distances Matrix and its copy
	int flag_forced = 0;		//Flag to know if only the leader is detected in the scene, but the group members are not spoted yet by the People Detector algorithm
	int flag_true_member = 0;	//Flag to consider if a detection with no motion could be considered as an static group member or as a non-member
	int exp = 0;
	int index = 0;
	int mem_index = 0;
	int mem_index2 = 0;
	int conta_tag = 0;
	long rectang_props[8][15] = { 0 };

	// d1 and d2 for calculating the differences
	// result, the result of and operation, calculated on d1 and d2
	// number_of_changes, the amount of changes in the result matrix.
	// color, the color for drawing the Movement rectangle when something has changed.
	cv::Mat d1, d2, motion;
	cv::Mat result;
	int number_of_changes = 0;
	int	number_of_sequence = 0;
	cv::Rect rectan;
	cv::Scalar mean_, color(0, 255, 255); //yellow

	// Corners of the "Movement rectangle", 
	// when all 0 means no motion is detected on the scene
	int le = 0;
	int to = 0;
	int ri = 0;
	int bo = 0;

	//Ground truth
	/*stringstream ss;
	string name = "frm_";
	////string name_motion = "motion_";
	string type = ".jpg";
	int muestra = 0;
	int c_muestra = 0;
	////int d_muestra = 0;*/

	//Initialization of the Dlib tracker algorithm
	dlib::correlation_tracker tracker;

	//Video Input
	cv::VideoCapture cap(argv[1]);

	if (!cap.isOpened())
		return -1;

	//Calling HOG descriptor for people detection
	cv::HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	cv::namedWindow("Group Leader Tracker", 1);

	for (;;)
	{
		cap >> img;

		if (img.empty())
			break;

		counter_img = counter_img + 1;
		std::printf("FRAME=%d\n", counter_img);

		if (!img.data)
			continue;

		img.copyTo(cpy_img);

		rem = counter_img % 25;

		//Preparing images for Motion Detection algorithm
		if (counter_img == 1)
		{
			img.copyTo(prevframe);
			cv::cvtColor(prevframe, prevframe, CV_RGB2GRAY);
		}

		if (counter_img == 2)
		{
			img.copyTo(currframe);
			cv::cvtColor(currframe, currframe, CV_RGB2GRAY);
		}

		if (counter_img == 3)
		{
			img.copyTo(nextframe);
			cv::cvtColor(nextframe, nextframe, CV_RGB2GRAY);
		}

		fflush(stdout);
		std::vector<Rect> found, found_filtered;

		//double t = (double)getTickCount();
		// run the detector with default parameters. to get a higher hit-rate
		// (and more false alarms, respectively), decrease the hitThreshold and
		// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).

		//DEFAULT SETTING
		//hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
		hog.detectMultiScale(img, found, -0.2, Size(12, 12), Size(16, 16), 1.05, 2);

		//t = (double)getTickCount() - t;
		//printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());

		size_t i, j;
		for (i = 0; i < found.size(); i++)
		{
			cv::Rect r = found[i];
			for (j = 0; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}

		// Detect motion in window
		int x_start = 10, x_stop = currframe.cols - 11;
		int y_start = 10, y_stop = currframe.rows - 11;

		// If more than 'there_is_motion' pixels are changed, we say there is motion
		int there_is_motion = 5;

		// Maximum deviation of the image, the higher the value, the more motion is allowed
		int max_deviation = 20;

		// Erode kernel
		cv::Mat kernel_ero = getStructuringElement(MORPH_RECT, Size(2, 2));

		for (i = 0; i < found_filtered.size(); i++)
		{
			cv::Rect r = found_filtered[i];
			// the HOG detector returns slightly larger rectangles than the real objects.
			// so we slightly shrink the rectangles to get a nicer output.
			r.x += cvRound(r.width*0.1);
			std::printf("r_x=%d\n", r.x);
			r.width = cvRound(r.width*0.7);
			r.y += cvRound(r.height*0.07);
			std::printf("r_y=%d\n", r.y);
			r.height = cvRound(r.height*0.8);

			if (counter_img == 1)
			{
				distances[2][i] = r.x;
				distances[3][i] = r.y;

				rectang_props[0][i] = r.x;
				rectang_props[1][i] = r.y;
				rectang_props[2][i] = r.width;
				rectang_props[3][i] = r.height;
				rectang_props[4][i] = r.tl().x;
				rectang_props[5][i] = r.tl().y;
				rectang_props[6][i] = r.br().x - 1;
				rectang_props[7][i] = r.br().y - 1;
			}

			if (distances[0][0] == 0 && distances[1][0] == 0 && distances[2][0] == 0 && distances[3][0] == 0 && distances[4][0] == 0)
			{
				distances[2][i] = r.x;
				distances[3][i] = r.y;

				rectang_props[0][i] = r.x;
				rectang_props[1][i] = r.y;
				rectang_props[2][i] = r.width;
				rectang_props[3][i] = r.height;
				rectang_props[4][i] = r.tl().x;
				rectang_props[5][i] = r.tl().y;
				rectang_props[6][i] = r.br().x - 1;
				rectang_props[7][i] = r.br().y - 1;
			}

			mat_i = 0;

			if (counter_img > 1) //After the first image, we need to identify the group leader
			{
				while (distances[2][mat_i] != 0)
				{
					if (d == 0) //first detections comparisson case?
					{
						value_x = r.x; //we intialize by assigning the first X value of the first detection on the current image

						compare_x = abs(value_x - distances[2][mat_i]); //and saving the value of the difference between the first X coord of previous image
						save_loc_2 = mat_i;

						mat_i = mat_i + 1; //we move to the next X coordinate on the previous image to keep comparing
						d = d + 1;         //flag for first detection comparisson performed
					}
					else //no first detections comparisson?
					{
						if (abs(value_x - distances[2][mat_i]) < compare_x) //compare if the current difference value is lower than other X values of
						{														   //the detections made on previous image
							compare_x = abs(value_x - distances[2][mat_i]); //if TRUE then we assign the difference
							save_loc_2 = mat_i;										   //and save the ID location to later point at it and calculate the difference on Y
						}

						mat_i = mat_i + 1; //we move to the next X coordinate on the previous image to keep comparing
					}
				}

				d = 0; //clear first detection comparisson flag

				if (compare_x < thresh)
				{
					if ((abs(r.y - distances[3][save_loc_2])) < thresh)
					{
						//when all the comparissons are finished for that detection we will have the information we want stored on the variables
						//then, we just then stored them on the distances vector
						distances[0][save_loc_2] = distances[0][save_loc_2] + compare_x; //the X distance difference between this same detection on the previous image respect to the current image
						distances[1][save_loc_2] = distances[1][save_loc_2] + abs((r.y) - (distances[3][save_loc_2])); //the same for Y distance
						distances[2][save_loc_2] = value_x; //and the current X coordinate to which this info belongs, to use later as a pointer for DeltaC
						distances[3][save_loc_2] = r.y; //also, the Y coordinate

						rectang_props[0][save_loc_2] = r.x;
						rectang_props[1][save_loc_2] = r.y;
						rectang_props[2][save_loc_2] = r.width;
						rectang_props[3][save_loc_2] = r.height;
						rectang_props[4][save_loc_2] = r.tl().x;
						rectang_props[5][save_loc_2] = r.tl().y;
						rectang_props[6][save_loc_2] = r.br().x - 1;
						rectang_props[7][save_loc_2] = r.br().y - 1;
					}
					else
					{
						distances[2][save_loc_2] = value_x;
						distances[3][save_loc_2] = r.y;

						rectang_props[0][save_loc_2] = r.x;
						rectang_props[1][save_loc_2] = r.y;
						rectang_props[2][save_loc_2] = r.width;
						rectang_props[3][save_loc_2] = r.height;
						rectang_props[4][save_loc_2] = r.tl().x;
						rectang_props[5][save_loc_2] = r.tl().y;
						rectang_props[6][save_loc_2] = r.br().x - 1;
						rectang_props[7][save_loc_2] = r.br().y - 1;
					}
				}
				else
				{
					if ((compare_x > 40))
					{
						jj = 0;
						while (distances[2][jj] != 0)
						{
							jj = jj + 1;
						}

						distances[2][jj] = value_x;
						distances[3][jj] = r.y;

						rectang_props[0][jj] = r.x;
						rectang_props[1][jj] = r.y;
						rectang_props[2][jj] = r.width;
						rectang_props[3][jj] = r.height;
						rectang_props[4][jj] = r.tl().x;
						rectang_props[5][jj] = r.tl().y;
						rectang_props[6][jj] = r.br().x - 1;
						rectang_props[7][jj] = r.br().y - 1;
					}
					else
					{
						distances[2][save_loc_2] = value_x;
						distances[3][save_loc_2] = r.y;

						rectang_props[0][save_loc_2] = r.x;
						rectang_props[1][save_loc_2] = r.y;
						rectang_props[2][save_loc_2] = r.width;
						rectang_props[3][save_loc_2] = r.height;
						rectang_props[4][save_loc_2] = r.tl().x;
						rectang_props[5][save_loc_2] = r.tl().y;
						rectang_props[6][save_loc_2] = r.br().x - 1;
						rectang_props[7][save_loc_2] = r.br().y - 1;
					}
				}
			}
		}

		if (counter_img > 3)
		{
			prevframe = currframe;
			currframe = nextframe;
			nextframe = cpy_img;
			result = nextframe;
			cv::cvtColor(nextframe, nextframe, CV_RGB2GRAY);

			// Calc differences between the images and do AND-operation
			// threshold image, low differences are ignored (ex. contrast change due to sunlight)
			absdiff(prevframe, nextframe, d1);
			absdiff(nextframe, currframe, d2);
			bitwise_and(d1, d2, motion);
			threshold(motion, motion, 35, 255, CV_THRESH_BINARY);
			erode(motion, motion, kernel_ero);

			//To obtain images of the calc differences between images 
			/*ss << name_motion << (d_muestra) << type;
			string filename = ss.str();
			ss.str("");
			imwrite(filename, d1);
			d_muestra = d_muestra + 1;

			ss << name_motion << (d_muestra) << type;
			filename = ss.str();
			ss.str("");
			imwrite(filename, d2);
			d_muestra = d_muestra + 1;

			ss << name_motion << (d_muestra) << type;
			filename = ss.str();
			ss.str("");
			imwrite(filename, motion);
			d_muestra = d_muestra + 1;*/

			cv::Scalar mean, stddev;
			meanStdDev(motion, mean, stddev);
			// if not to much changes then the motion is real (neglect agressive snow, temporary sunlight)
			if (stddev[0] < max_deviation)
			{
				int number_of_changes = 0;
				int min_x = motion.cols, max_x = 0;
				int min_y = motion.rows, max_y = 0;
				// loop over image and detect changes
				for (int j = y_start; j < y_stop; j += 2){ // height
					for (int i = x_start; i < x_stop; i += 2){ // width
						// check if at pixel (j,i) intensity is equal to 255
						// this means that the pixel is different in the sequence
						// of images (prev_frame, current_frame, next_frame)
						if (static_cast<int>(motion.at<uchar>(j, i)) == 255)
						{
							number_of_changes++;
							if (min_x>i) min_x = i;
							if (max_x<i) max_x = i;
							if (min_y>j) min_y = j;
							if (max_y<j) max_y = j;
						}
					}
				}
				if (number_of_changes){
					//check if not out of bounds
					if (min_x - 10 > 0) min_x -= 10;
					if (min_y - 10 > 0) min_y -= 10;
					if (max_x + 10 < result.cols - 1) max_x += 10;
					if (max_y + 10 < result.rows - 1) max_y += 10;

					// draw rectangle round the changed pixel
					Point x(min_x, min_y);
					le = min_x;
					to = min_y;

					Point y(max_x, max_y);
					ri = max_x;
					bo = max_y;

					a.x.min = min_x;
					a.y.min = min_y;
					a.x.max = max_x;
					a.y.max = max_y;

					Rect rectan(x, y);
					Mat cropped = result(rectan);
					cv::rectangle(result, rectan, color, 2);
				}
			}


			// If a lot of changes happened, we assume something changed.
			if (number_of_changes >= there_is_motion)
			{
				if (number_of_sequence>0){
					cv::rectangle(cpy_img, rectan, color, 2);
				}
				number_of_sequence++;
			}
			else
			{
				number_of_sequence = 0;
			}
		}

		for (i = 0; i < found_filtered.size(); i++)
		{
			std::printf("i=%d\n", i);

			cv::Rect r = found_filtered[i];

			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.7);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);

			std::printf("r_x0=%d\n", r.x);

			char text[255];
			std::sprintf(text, "%d,%d", r.x, r.y);

			b.x.min = r.x;
			b.y.min = r.y;
			b.y.max = cvRound(r.y + r.height);
			b.x.max = cvRound(r.x + r.width);

			if (le != 0 && to != 0 && ri != 0 && bo != 0 && counter_img < 51)
			{
				cv::rectangle(cpy_img, r.tl(), r.br(), cv::Scalar(204, 0, 204), 3);
				cv::putText(cpy_img, "Detected", cvPoint(r.x, cvRound(r.y + r.height + 13)),
					FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(204, 0, 204), 1, CV_AA);
				cv::putText(cpy_img, text, cvPoint(r.x, cvRound(r.y - 10)),
					FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(204, 0, 204), 1, CV_AA);
				std::printf("DETECTED\n");
			}

			if (le == 0 && to == 0 && ri == 0 && bo == 0) //If there's no Movement rectangle but there is a Detection bounding box, people is just marked as "Detected"
			{
				cv::rectangle(cpy_img, r.tl(), r.br(), cv::Scalar(204, 0, 204), 3);
				cv::putText(cpy_img, "Detected", cvPoint(r.x, cvRound(r.y + r.height + 13)),
					FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(204, 0, 204), 1, CV_AA);
				cv::putText(cpy_img, text, cvPoint(r.x, cvRound(r.y - 10)),
					FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(204, 0, 204), 1, CV_AA);
				std::printf("DETECTED\n");
			}
			else
			{
				if (BoxesIntersect(a, b) == true) //If Movement rectangle and Detection bounding box intersect...
				{
					flag_member = flag_member + 1;
					m = 0;

					std::printf("r_x1=%d\n", r.x);
					std::printf("flag_member=%d\n", flag_member);

					while (distances[2][m] != 0)
					{
						//Locate the corresponding detection. If this is the first person moving significantly than others on the scene (Flag member = 1), 
						//point it as the leader (Group Leader = 3), else, it is considered as a follower (Group Member = 1)
						if (r.x == distances[2][m] && flag_member != 1)
						{
							if ((distances[4][m] == 0) || (distances[4][m] != 0 && distances[8][m] > 0)) //|| distances[4][m] == 4)
							{
								distances[6][m] = distances[6][m] + 1;
								std::printf("MyD intersect (M)=%d\n", distances[6][m]);
							}
							id_member = m;
						}

						if (r.x == distances[2][m] && flag_member == 1)
						{
							if ((distances[4][m] == 0) || (distances[4][m] != 0 && distances[8][m] > 0)) //|| distances[4][m] == 4)
							{
								distances[5][m] = distances[5][m] + 1;
								std::printf("MyD intersect (L)=%d\n", distances[5][m]);
							}
							id_member = m;
						}

						m = m + 1;
					}

					flag_member = 0;
				}
				else //If Movement rectangle and Detection bounding box DO NOT intersect...
				{
					m = 0;
					std::printf("r_x2=%d\n", r.x);

					while (distances[2][m] != 0)
					{
						if (r.x == distances[2][m]) //but there is a corresponding previous detection for that bounding box
						{
							//If it has NO Category tag or there is Potential GM, check if this rectangle intersects with any other Detection bounding box
							if ((distances[4][m] == 0) || (distances[4][m] != 0 && distances[8][m] > 0))
							{
								for (ii = 0; ii < found_filtered.size(); ii++)
								{
									cv::Rect r = found_filtered[ii];

									r.x += cvRound(r.width*0.1);
									r.width = cvRound(r.width*0.7);
									r.y += cvRound(r.height*0.07);
									r.height = cvRound(r.height*0.8);

									c.x.min = r.x;
									c.y.min = r.y;
									c.y.max = cvRound(r.y + r.height);
									c.x.max = cvRound(r.x + r.width);

									//If it does, it means this detection is a "True Group Member" but it was complex to spot it in the first place
									if (r.x != distances[2][m] && BoxesIntersect(b, c) == true)
									{
										flag_true_member = 1;
									}	
								}

								int myints[15] = { 0 };
								int r = 0;

								for (jj = 0; jj < 15; jj++)
								{
									myints[jj] = distances[6][jj];
								}

								std::vector<int> myvector(myints, myints + 15);

								auto it = find_if(myvector.begin(), myvector.end(), [](int i){return i > 0; });

								if (it != myvector.end())
								{
									auto pos = it - myvector.begin();
									int result = abs(distances[2][pos] - distances[2][m]);
									std::printf("pos=%d\n", pos);
									std::printf("it=%d\n", m);
									std::printf("result=%d\n", result);

									//If this detection do not intersect with any other but the bounding boxes are close enough, it is considered a true group member as well
									if (result < 80)
									{
										flag_true_member = 1;
									}
										
								}
								else
								{
									int n = 0;

									while (distances[2][n] != 0)
									{
										for (ii = 0; ii < found_filtered.size(); ii++)
										{
											cv::Rect r = found_filtered[ii];

											r.x += cvRound(r.width*0.1);
											r.width = cvRound(r.width*0.7);
											r.y += cvRound(r.height*0.07);
											r.height = cvRound(r.height*0.8);

											if (distances[2][n] == r.x)
											{
												rightc[n] = cvRound(r.x + r.width);
											}
										}

										n = n + 1;
									}

									n = 0;

									//Check if the left and right corners of Detection bounding boxes are close enough (5 pixels apart as maximum). If this is TRUE and the detection has
									//NO tag assigned yet, means we have a Potential Group Member = 4.
									while (distances[2][n] != 0)
									{
										if ((m != n && (abs(distances[2][m] - rightc[n]) < 5)) || (m != n && (abs(rightc[m] - distances[2][n]) < 5)))
										{
											flag_true_member = 1;
										}

										n = n + 1;
									}
								}

								//If the "True Group Member" flag was activated, tag the detection as Group Member = 1, else, this detection is a Non-Member = 2
								if (flag_true_member == 1)
								{
									distances[6][m] = distances[6][m] + 1;
									std::printf("MyD DONT intersect (M)=%d\n", distances[6][m]);
								}
								else
								{
									distances[7][m] = distances[7][m] + 1;
									std::printf("MyD DONT intersect (N)=%d\n", distances[7][m]);
								}

								flag_true_member = 0;
							}

							id_member = m;
						}

						m = m + 1;
					}

				}

			}		
		
		}

		//Printing Detection bounding boxes with their corresponding Roles given by Category tags (1 & 2)
		exp = counter_img % 51;

		if (exp == 0)
		{
			int lm = 0;

			while (distances[2][lm] != 0)
			{
				std::vector<int> exp_tags{ distances[6][lm], distances[7][lm], distances[8][lm] };
				std::vector<int>::iterator exp_results;
				int max_e = *max_element(exp_tags.begin(), exp_tags.end());
				exp_results = std::max_element(exp_tags.begin(), exp_tags.end());
				index = std::distance(exp_tags.begin(), exp_results);

				std::printf("index=%d\n", index);
				std::printf("lm=%d\n", lm);
				std::printf("max_e=%d\n", max_e);

				if (max_e == 0)
				{
					index = 2;
				}

				if (index == 0)
				{
					distances[4][lm] = 1; //Group member
				}

				if (index == 1)
				{
					distances[4][lm] = 2; //Non-member
				}

				if (index == 2)
				{
					distances[4][lm] = 4; //Potential
				}

				lm = lm + 1;
			}

			int g_members[15] = { 0 };

			for (jj = 0; jj < 15; jj++)
			{
				g_members[jj] = distances[5][jj];
			}

			std::vector<int> mem_vector(g_members, g_members + 15);
			std::vector<int>::iterator mem_results;
			mem_results = std::max_element(mem_vector.begin(), mem_vector.end());
			mem_index = std::distance(mem_vector.begin(), mem_results);

			distances[4][mem_index] = 3; //Group leader
			std::printf("mem_index=%d\n", mem_index);

			int h_members[15] = { 0 };

			for (jj = 0; jj < 15; jj++)
			{
				h_members[jj] = distances[6][jj];
			}

			std::vector<int> mem_vector2(h_members, h_members + 15);
			std::vector<int>::iterator mem_results2;
			mem_results2 = std::max_element(mem_vector2.begin(), mem_vector2.end());
			mem_index2 = std::distance(mem_vector2.begin(), mem_results2);

			std::printf("mem_index2=%d\n", mem_index2);

			if (distances[6][mem_index2] > distances[5][mem_index])
			{
				distances[4][mem_index] = 1;
				distances[4][mem_index2] = 3;
			}

			std::printf("EXPO TAGS\n");
			for (int ii = 4; ii < 9; ii++) {
				for (int jj = 0; jj < 15; jj++) {
					std::printf("%d ", distances[ii][jj]);
				}
				std::printf("\n");
			}

			std::printf("RECTANGLE MTRX\n");
			for (int ii = 0; ii < 8; ii++) {
				for (int jj = 0; jj < 15; jj++) {
					std::printf("%d ", rectang_props[ii][jj]);
				}
				std::printf("\n");
			}

		}

		if (counter_img > 51)
		{
			int lm = 0;

			while (distances[2][lm] != 0)
			{
				Point x_tl(rectang_props[4][lm], rectang_props[5][lm]);
				Point y_br(rectang_props[6][lm], rectang_props[7][lm]);

				Rect rectan_2(x_tl, y_br);

				char text_roles[255];
				std::sprintf(text_roles, "%d,%d", rectang_props[0][lm], rectang_props[1][lm]);

				if (distances[4][lm] == 1)
				{
					cv::rectangle(cpy_img, rectan_2, cv::Scalar(0, 255, 0), 3);
					cv::putText(cpy_img, "Group Member", cvPoint(rectang_props[0][lm], cvRound(rectang_props[1][lm] + rectang_props[3][lm] + 13)),
						FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(0, 255, 0), 1, CV_AA);
					cv::putText(cpy_img, text_roles, cvPoint(rectang_props[0][lm], cvRound(rectang_props[1][lm] - 10)),
						FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(0, 255, 0), 1, CV_AA);
					std::printf("GROUP MEMBER\n");
				}

				if (distances[4][lm] == 2)
				{
					cv::rectangle(cpy_img, rectan_2, cv::Scalar(0, 0, 255), 3);
					cv::putText(cpy_img, "NON Member", cvPoint(rectang_props[0][lm], cvRound(rectang_props[1][lm] + rectang_props[3][lm] + 13)),
						FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(0, 0, 255), 1, CV_AA);
					cv::putText(cpy_img, text_roles, cvPoint(rectang_props[0][lm], cvRound(rectang_props[1][lm] - 10)),
						FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(0, 0, 255), 1, CV_AA);
					std::printf("NON MEMBER\n");
				}

				if (distances[4][lm] == 4)
				{
					cv::rectangle(cpy_img, rectan_2, cv::Scalar(204, 0, 204), 3);
					cv::putText(cpy_img, "Detected", cvPoint(rectang_props[0][lm], cvRound(rectang_props[1][lm] + rectang_props[3][lm] + 13)),
						FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(204, 0, 204), 1, CV_AA);
					cv::putText(cpy_img, text_roles, cvPoint(rectang_props[0][lm], cvRound(rectang_props[1][lm] - 10)),
						FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(204, 0, 204), 1, CV_AA);
					std::printf("DETECTED\n");
				}

				//Initializing Leader Tracker using the Detection bounding box as ROI
				if (distances[4][lm] == 3 && flag_track == 0)
				{
					int new_center_x = rectang_props[0][lm] + cvRound(rectang_props[2][lm] / 2);
					int new_center_y = rectang_props[1][lm] + cvRound(rectang_props[3][lm] / 2);

					leader_width = rectang_props[2][lm];
					leader_height = rectang_props[3][lm];

					cv::Rect myROI_prev(rectang_props[0][lm], rectang_props[1][lm], leader_width, leader_height);

					if (myROI_prev.x < 0)
						myROI_prev.x = 1;

					if (myROI_prev.y < 0)
						myROI_prev.y = 1;

					if (myROI_prev.x + myROI_prev.width > cpy_img.cols)
					{
						int diff_w = (myROI_prev.x + myROI_prev.width) - cpy_img.cols;
						myROI_prev.x = myROI_prev.x - diff_w;
					}

					if (myROI_prev.y + myROI_prev.height > cpy_img.rows)
					{
						int diff_h = (myROI_prev.y + myROI_prev.height) - cpy_img.rows;
						myROI_prev.y = myROI_prev.y - diff_h;
					}

					img(myROI_prev).copyTo(prevgray);

					cv::cvtColor(prevgray, prevgray, COLOR_BGR2GRAY);

					dlib::array2d<dlib::rgb_pixel> cimg;
					dlib::assign_image(cimg, dlib::cv_image<dlib::bgr_pixel>(img));

					tracker.start_track(cimg, dlib::centered_rect(dlib::point(new_center_x, new_center_y), leader_width, leader_height)); //Leader tracking starts!

					flag_track = 1;
				}

				lm = lm + 1;
			}
		}

		if (exp == 0)
		{
			conta_tag = conta_tag + 1;

			//Cleaning matrix
			for (int ii = 5; ii < 8; ii++) {
				for (int jj = 0; jj < 15; jj++) {
					distances[ii][jj] = 0;
				}
			}
		}

		//Every 25 frames, we check for any false-positives to erase from the database. False-positives are spoted by identifying NO CHANGES on their coordinates over
		//25 frames. A detection that remains static is very likely to be an object, and so, a false-positive.
		m = 0;
		int g = 0;

		if (rem == 0)
		{
			unch = unch + 1;

			while (distances[0][m] != 0)
			{
				m = m + 1;
			}

			if (distances[2][m] != 0)
			{
				g = m + 1;

				while (distances[2][g] != 0)
				{
					g = g + 1;
				}

				distances[0][m] = distances[0][g - 1];
				distances[1][m] = distances[1][g - 1];
				distances[2][m] = distances[2][g - 1];
				distances[3][m] = distances[3][g - 1];

				distances[0][g - 1] = 0;
				distances[1][g - 1] = 0;
				distances[2][g - 1] = 0;
				distances[3][g - 1] = 0;
			}

			if (unch == 1)
			{
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 15; j++) {
						matrix[i][j] = distances[i][j];
					}
				}
			}
			else
			{
				m = 0;
				while (distances[2][m] != 0){
					if (matrix[0][m] == distances[0][m] && matrix[1][m] == distances[1][m])
					{
						g = m + 1;

						while (distances[2][g] != 0)
						{
							g = g + 1;
						}

						distances[0][m] = distances[0][g - 1];
						distances[1][m] = distances[1][g - 1];
						distances[2][m] = distances[2][g - 1];
						distances[3][m] = distances[3][g - 1];

						distances[0][g - 1] = 0;
						distances[1][g - 1] = 0;
						distances[2][g - 1] = 0;
						distances[3][g - 1] = 0;
					}

					m = m + 1;
				}
			}

			m = 0;

			//Identifying and storing the right corners of all Detection bounding boxes
			if (number_of_sequence == 0 && distances[2][m] != 0)
			{
				while (distances[2][m] != 0)
				{
					for (i = 0; i < found_filtered.size(); i++)
					{
						cv::Rect r = found_filtered[i];

						r.x += cvRound(r.width*0.1);
						r.width = cvRound(r.width*0.7);
						r.y += cvRound(r.height*0.07);
						r.height = cvRound(r.height*0.8);

						if (distances[2][m] == r.x)
						{
							rightc[m] = cvRound(r.x + r.width);
						}
					}

					m = m + 1;
				}

				m = 0;

				//Check if the left and right corners of Detection bounding boxes are close enough (100 pixels apart as maximum). If this is TRUE and the detection has
				//NO tag assigned yet, means we have a Potential Group Member = 4.
				while (distances[2][m] != 0)
				{
					if ((abs(distances[2][m] - rightc[m + 1]) < 150 && distances[4][m] == 0) || (abs(rightc[m] - distances[2][m + 1]) < 150 && distances[4][m] == 0))
					{
						distances[8][m] = distances[8][m] + 1;
						std::printf("RC closeness (P)=%d\n", distances[8][m]);
					}

					m = m + 1;
				}

			}

			if (found_filtered.size() == 1) //If there is only ONE detection, we have a Stationary State (Hard Case), flag_forced = 1.
			{
				flag_forced = 1;
			}

		} //end of "Every 25 frames" activity

		std::printf("detections=%d\n", found_filtered.size());

		//If there is NO detections, there is NO one to track
		if ((found_filtered.size() == 0))
		{
			flag_track = 0;
		}

		//If there is a leader to track, then we obtain all the necessary information. Convert the leader bounding box into a ROI and calculate the optical flow
		//from the center area (the leader's torso) on every image. Store this optical flow data. Print the corresponding leader box and tag.
		if (flag_track == 1)
		{
			dlib::array2d<dlib::rgb_pixel> cimg;
			dlib::assign_image(cimg, dlib::cv_image<dlib::bgr_pixel>(img));

			tracker.update(cimg);
			dlib::drectangle rect = tracker.get_position();

			int left = cvRound(rect.left());
			int top = cvRound(rect.top());
			int right = cvRound(rect.left() + rect.width() - 1);
			int bottom = cvRound(rect.top() + rect.height() - 1);

			cv::Rect myROI(left, top, leader_width, leader_height);

			if (myROI.x < 0)
				myROI.x = 1;

			if (myROI.y < 0)
				myROI.y = 1;

			if (myROI.x + myROI.width > cpy_img.cols)
			{
				int  diff_w = (myROI.x + myROI.width) - cpy_img.cols;
				myROI.x = myROI.x - diff_w;
			}

			if (myROI.y + myROI.height > cpy_img.rows)
			{
				int diff_h = (myROI.y + myROI.height) - cpy_img.rows;
				myROI.y = myROI.y - diff_h;
			}

			img(myROI).copyTo(croppedImage);

			cv::cvtColor(croppedImage, croppedImage, COLOR_BGR2GRAY);

			int center_roi_x = left + cvRound(rect.width() / 2);
			int center_roi_y = top + cvRound(rect.height() / 2);

			cv::calcOpticalFlowFarneback(prevgray, croppedImage, flow, 0.4, 1, 12, 2, 8, 1.2, 0);

			flow.copyTo(cflow);

			int center_x = cvRound(leader_width / 2);
			int center_y = cvRound(leader_height / 2);

			const cv::Point2f& fxy = cflow.at<cv::Point2f>(center_y, center_x) * 10;

			opt_flows[0][0] = opt_flows[0][0] + fxy.x;
			opt_flows[1][0] = opt_flows[1][0] + fxy.y;

			croppedImage.copyTo(prevgray);

			cv::rectangle(cpy_img,
				cvPoint(left, top),
				cvPoint(right, bottom),
				CV_RGB(0, 0, 255), 3, 8, 0);

			char text[255];
			std::sprintf(text, "%d,%d", left, top);

			cv::putText(cpy_img, "Group Leader", cvPoint(left, cvRound(top + rect.height() + 13)),
				FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(255, 0, 0), 1, CV_AA);
			cv::putText(cpy_img, text, cvPoint(left, cvRound(top - 10)),
				FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(255, 0, 0), 1, CV_AA);

			//Every 25 frames, an average from the optical flows obtained throughout that time frame is computed. The result is an arrow with magnitude and direction
			//that let us know the actual direction the leader is facing on the scene. 
			if (rem == 0)
			{
				int avr_x = cvRound((opt_flows[0][0] / 25) * 10);
				int avr_y = cvRound((opt_flows[1][0] / 25) * 10);

				cv::arrowedLine(cpy_img, cv::Point(center_roi_x, center_roi_y), cv::Point(cvRound(center_roi_x + avr_x), cvRound(center_roi_y + avr_y)), Scalar(255, 0, 0), 1, 8, 0, 0.1);
			}
		}

		//Reset
		mat_i = 0;

		//For Ground Truth comparisson
		/*muestra = counter_img % 5;

		if (muestra == 0)
		{
		c_muestra = c_muestra + 1;
		ss << name << (c_muestra) << type;
		string filename = ss.str();
		ss.str("");
		imwrite(filename, cpy_img);
		}*/

		std::printf("Distances Matrix\n");
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 15; j++) {
				std::printf("%d ", distances[i][j]);
			}
			std::printf("\n");
		}

		cv::imshow("Group Leader Tracker", cpy_img);

		if (cv::waitKey(20) >= 0)
			break;
	}

	return 0;
}
