///////////////////////////////////////////////////////////////////////////
//
//	Hand Writing digits Recognition
//		#1 classification
//		#2 training
//		#3 testing
//
///////////////////////////////////////////////////////////////////////////

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>
#include<math.h>

// dimension variables ////////////////////////////////////////////////////////

const int UNIT = 20;
const int MIN = 10;

// classification	   ////////////////////////////////////////////////////////

int classification()
{

	std::cout << "This program will recognize hand writing digits." << std::endl;

	cv::Mat img_init;									//// input MNIST image
	cv::Mat img_gray_scale;								////
	cv::Mat img_gray_scale_cpoy;						////

	cv::Mat classification;								////
	cv::Mat img_float;
	cv::Mat img_float_flat;						////
	cv::Mat img_float_flat_total;					////

	std::vector<std::vector<cv::Point>> output_contours;////
	std::vector<cv::Vec4i> output_hierarchy;			////for find contour

	//std::cout << std::endl << " digits: " << std::endl
	//	<< "'0', '1', '2', '3', '4', '5', '6', '7', '8', '9' "
	//	<< std::endl << std::endl;

	///	load and process image	/////////////////////////////////////////////////

	img_init = cv::imread("digits_training_dataset.png");//// load dataset image

	if (img_init.empty())
	{
		std::cout << "Error: Image loading failed." << std::endl;

		return -1;
	}
	else
	{
		std::cout << "Image loading successful." << std::endl;

	}

	cv::cvtColor(img_init, img_gray_scale, CV_BGR2GRAY);

	img_gray_scale_cpoy = img_gray_scale.clone();

	//	contour hand writing figures	//////////////////////////////////////////////////

	cv::findContours(img_gray_scale_cpoy,
		output_contours,
		output_hierarchy,
		cv::RETR_EXTERNAL,
		cv::CHAIN_APPROX_SIMPLE);

	//	classification of the contours	////////////////////////////////////////////////

	for (int i = 0; i < output_contours.size(); i++)
	{
		if (cv::contourArea(output_contours[i]) > MIN)					// avoid noise dots
		{
			cv::Rect boundary = cv::boundingRect(output_contours[i]);	// extract data from contour[i]

			//std::cout << boundary.y / 100;					// contour[i] y digit

			classification.push_back(boundary.y / 100);					// append

																		// To check whether contours work well, IMAGES show by the end.
			cv::rectangle(img_init, boundary, cv::Scalar(225, 225, 225), 1);

			cv::Mat ROI = img_init(boundary);
			cv::Mat ROI_resized;
			cv::resize(ROI, ROI_resized, cv::Size(UNIT, UNIT));
			ROI_resized.convertTo(img_float, CV_32FC1);

			img_float_flat = img_float.reshape(1, 1);
			img_float_flat_total.push_back(img_float_flat);

		}	// end if
	}	//	end for

		// save classification and img_float_flattened	////////////////////////////////////

	cv::FileStorage classification_file_storage("classification.xml", cv::FileStorage::WRITE);
	classification_file_storage << "classification" << classification;
	classification_file_storage.release();

	cv::FileStorage img_float_flat_total_file_storage("images.xml", cv::FileStorage::WRITE);
	img_float_flat_total_file_storage << "images" << img_float_flat_total;
	img_float_flat_total_file_storage.release();

	std::cout << std::endl;
	std::cout << "Classification finished.";
	std::cout << std::endl;
	//	Image show to check whether contours cover items properly 


	return 0;
}

//	training		//////////////////////////////////////////////////////////////////

int training()
{
	//	loading classification.xml and images.xml	//////////////////////////////////
	cv::Mat classification;
	cv::Mat img_float_flat_total;

	cv::FileStorage classification_file_storage("../classification.xml", cv::FileStorage::READ);
	classification_file_storage["classification"] >> classification;
	classification_file_storage.release();

	cv::FileStorage img_float_flat_total_file_storage("images.xml", cv::FileStorage::READ);
	img_float_flat_total_file_storage["images"] >> img_float_flat_total;
	img_float_flat_total_file_storage.release();

	// trainning	///////////////////////////////////////////////////////////////
	cv::Ptr<cv::ml::KNearest> k_nearest(cv::ml::KNearest::create());

	k_nearest->train(img_float_flat_total, cv::ml::ROW_SAMPLE, classification);
	k_nearest->save("k_nearest.yml");
	std::cout << std::endl;


	std::cout << std::endl;
	std::cout << "Training finished.";
	std::cout << std::endl;


	return 0;
}

int testing() 
{
	cv::Mat testing_init;
	cv::Mat testing_grayscale;
	cv::Mat testing_blur;
	cv::Mat testing_thresh;
	cv::Mat testing_thresh_copy;

	cv::Mat classification;
	cv::Mat hirarchy;

	std::cout << std::endl;
	std::cout << "Testing finished.";
	std::cout << std::endl;
	 
	return 0;
}

int main()
{
	int option_key=0;
	std::cout
		<< std::endl
		<< "///////////////////////////////////////////////////////////////////////////" << std::endl
		<< "//" << std::endl
		<< "//	Hand Writing digits Recognition " << std::endl
		<< "//		1 classification " << std::endl
		<< "//		2 training " << std::endl
		<< "//		3 testing " << std::endl
		<<"//		4 exit"		<<std::endl
		<< "// " << std::endl
		<< "///////////////////////////////////////////////////////////////////////////" << std::endl
		<< " Please input an option (between 1 to 3): ";
		

	while(std::cin >> option_key)
	{
		if (option_key == 1)
			classification();
		else if (option_key == 2)
			training();
		else if (option_key == 3)
			training();
		else if (option_key == 4)
			return 0;

		std::cout
			<< std::endl
			<< "///////////////////////////////////////////////////////////////////////////" << std::endl
			<< "//" << std::endl
			<< "//	Hand Writing digits Recognition " << std::endl
			<< "//		1 classification " << std::endl
			<< "//		2 training " << std::endl
			<< "//		3 testing " << std::endl
			<< "//		4 exit" << std::endl
			<< "// " << std::endl
			<< "///////////////////////////////////////////////////////////////////////////" << std::endl
			<< " Please input an option (between 1 to 3): ";
	}

}