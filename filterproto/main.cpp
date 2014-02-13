//
//  main.cpp
//  filterproto
//
//  Created by 直井真一郎 on 2014/02/10.
//  Copyright (c) 2014年 直井真一郎. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat sepConv(Mat input, int radius)
{
    
    
    Mat sep;
    Mat dst,dst2;
    
    int ksize = 2 *radius +1;
    double sigma = radius / 2.575;
    
    Mat gau = getGaussianKernel(ksize, sigma,CV_32FC1);
    
    Mat newgau = Mat(gau.rows,1,gau.type());
    gau.col(0).copyTo(newgau.col(0));
    
    
    filter2D(input, dst2, -1, newgau);
    
    
    filter2D(dst2.t(), dst, -1, newgau);
    
    
    return dst.t();
    
    
}

/** @function main */
int main( int argc, char** argv )
{
    Mat src, src_gray, blurred, dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    const char* window_name = "Laplace Demo";
    
    /// Load an image
    //src = imread( argv[1] );
    //src = imread("/Users/naoishinichirou/Downloads/royal_ascot_horses.jpg");
    src = imread("/Users/naoishinichirou/Downloads/california_t.jpeg");

    if( !src.data ) return -1;
    
    
    // 1.
    /// Remove noise by blurring with a Gaussian filter
    //GaussianBlur( src, blurred, Size(3,3), 0, 0, BORDER_DEFAULT );
    //bilateralFilter(src, blurred, 20, 90, 40);
    //bilateralFilter(src, blurred, 11, 40, 200);

    bilateralFilter(src, blurred, 11, 40, 200);
    bilateralFilter(blurred, dst, 11, 40, 200);
    bilateralFilter(dst, blurred, 11, 40, 200);
    bilateralFilter(blurred, dst, 11, 40, 200);
    bilateralFilter(dst, blurred, 11, 40, 200);
    //bilateralFilter(blurred, dst, 11, 40, 200);

    
    cvtColor( src, src_gray, CV_RGB2GRAY );
    Mat dog = sepConv(src_gray, 1) - sepConv(src_gray, 4);
    cvtColor( dog, dog, CV_GRAY2RGB );
    
    

//    Mat gaussian;
//    GaussianBlur( src, gaussian, Size(3,3), 0, 0, BORDER_DEFAULT );
//    /// Convert the image to grayscale
//    cvtColor( gaussian, src_gray, CV_RGB2GRAY );
    
    
//    /// Apply Laplace function
//    Mat abs_dst;
//    
//    Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
//    convertScaleAbs( dst, abs_dst );
//    
//    
//    cvtColor( abs_dst, dst, CV_GRAY2RGB );
    
    // 2.
    Mat median;
    medianBlur(blurred, median, 31);

    // 3.
    Mat reshaped_image = blurred.reshape(1, blurred.cols * blurred.rows);
    Mat reshaped_image32f;
    reshaped_image.convertTo(reshaped_image32f, CV_32FC1, 1.0 / 255.0);
    
    Mat labels;
    int cluster_number = 5;
    TermCriteria criteria {TermCriteria::COUNT, 100, 1};
    Mat centers;
    kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, KMEANS_RANDOM_CENTERS, centers);
    
    Mat rgb_image(src.rows, src.cols, CV_8UC3);
    MatIterator_<Vec3b> rgb_first = rgb_image.begin<Vec3b>();
    MatIterator_<Vec3b> rgb_last = rgb_image.end<Vec3b>();
    MatConstIterator_<int> label_first = labels.begin<int>();
    
    Mat centers_u8;
    centers.convertTo(centers_u8, CV_8UC1, 255.0);
    Mat centers_u8c3 = centers_u8.reshape(3);
    
    while ( rgb_first != rgb_last ) {
        const Vec3b& rgb = centers_u8c3.ptr<Vec3b>(*label_first)[0];
        *rgb_first = rgb;
        ++rgb_first;
        ++label_first;
    }
    
    // edge darken
    Mat laplacian,tmp;
    Laplacian( rgb_image, tmp, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( tmp, laplacian );
    rgb_image = rgb_image * 0.9 - laplacian * 0.1;

    
    // edge darken for median
    Laplacian( median, tmp, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( tmp, laplacian );
    median = median - sepConv(laplacian, 3);

//    Mat dog = sepConv(blurred, 1) - sepConv(blurred, 4);
//    imshow( "DoG", dog );
    
    Mat res = median * 0.5 + rgb_image * 0.7;
    res -= dog * 2;
    imshow( window_name, res );

    
    
    
    /// Show what you got
    //imshow( window_name, dst );
    
    waitKey(0);
    
    return 0;
}
