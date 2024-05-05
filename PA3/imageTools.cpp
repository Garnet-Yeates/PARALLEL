#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

extern "C" int* imageToMat(char* filename,int* dims){
    Mat image;
    image=imread(filename,CV_LOAD_IMAGE_COLOR);

    Mat gray_image;
    cvtColor(image,gray_image,CV_BGR2GRAY);

    if(! image.data ) // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        int* temp;
        return temp;
    }
    int width=gray_image.size().width;
    int height=gray_image.size().height;
    dims[0]=height;
    dims[1]=width;

    cout << "Height: " << dims[0] << endl;
    cout << "Width : " << dims[1] << endl;
    //namedWindow( "Original Image", WINDOW_AUTOSIZE ); // Create a window for display.
    //imshow( "Original Image", gray_image ); // Show our image inside it.
 
 
    //allocate 2d array
    int *matrix;
    matrix=(int*) malloc(height*width*sizeof(*matrix));
    for(size_t i=0;i<height;i++){
        for(size_t j=0;j<width;j++){
            int intensity=gray_image.at<uchar>(i,j);
            if(intensity>255){
                intensity=255;
            }
            if(intensity<0){
                intensity=0;
            }
            matrix[i*width+j]=intensity;
        }
    }

    return matrix;
}


extern "C" void matToImage(char* filename, int* mat, int* dims){
    int height=dims[0];
    int width=dims[1];
    Mat image(height, width, CV_8UC1, Scalar(0,0,0));

    for(size_t i=0;i<height;i++){
        for(size_t j=0;j<width;j++){
            image.at<uchar>(i,j) = (int)mat[i*width+j];
        }
    } 
    //namedWindow( "Processed Image", WINDOW_AUTOSIZE );// Create a window for display.

    //imshow( "Processed Image", image );  // Show our image inside it.
    imwrite(filename,image);
    //waitKey(0); // Wait for a keystroke in the window
    return;
}
