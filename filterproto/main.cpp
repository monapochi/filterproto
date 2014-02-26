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

Mat getDrawingLine(Mat src)
{
    Mat line;
    
    cvtColor( src, src, CV_RGB2GRAY );
    line = sepConv(src, 1) - sepConv(src, 4);
    
    //adaptiveThreshold(line, line, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 8);
    //threshold(line, line, 0, 255, THRESH_BINARY|THRESH_OTSU);
    
    
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Mat abs_dst,dst;
    Laplacian( src, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( dst, abs_dst );
    abs_dst -= src;
    line += abs_dst;
    
    cvtColor( line, line, CV_GRAY2RGB );

    
    
    return line;
}


Mat filter(Mat src, int median_param, int k, int lineParam, bool flagLine, bool flagResize)
{
    //
    Mat src_gray, blurred, dst, res, dog;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    if( !src.data ) return res;
    
    if(flagResize)
    {
        if (src.cols > 512) {
            float h, w, nw, nh;
            h = src.rows;
            w = src.cols;
            nw = 512;
            nh = (h/w)*nw; // Maintain aspect ratio based on a desired width.
            
            resize(src, src, Size(nw,nh),INTER_AREA);
        }
    }
    
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
    
    if(flagLine)
        dog = getDrawingLine(src);
    
    
    
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
    medianBlur(blurred, median, median_param); // 32 for testing

    
    // 3.
    Mat reshaped_image = blurred.reshape(1, blurred.cols * blurred.rows);
    Mat reshaped_image32f;
    reshaped_image.convertTo(reshaped_image32f, CV_32FC1, 1.0 / 255.0);
    
    
    Mat labels;
    int cluster_number = k; // 5 for testing
    TermCriteria criteria {TermCriteria::COUNT, 100, 1};
    Mat centers;
    //kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, KMEANS_RANDOM_CENTERS, centers);
    kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, KMEANS_PP_CENTERS, centers);
    
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
    
    res = median * 0.5 + rgb_image * 0.7;
    if(flagLine)
      res -= dog * lineParam; // 2 for testing

    
    return res;
}

//
Mat filter2(Mat src, int median_param, int k, int line_param)
{
    //
    Mat src_gray, blurred, dst, res;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    float h, w, nw, nh;
    h = src.rows;
    w = src.cols;
    nw = 32;
    nh = (h/w)*nw; // Maintain aspect ratio based on a desired width.

    if( !src.data ) return res;


    // 1. Difference of Gaussians
    cvtColor( src, src_gray, CV_RGB2GRAY );
    Mat dog = sepConv(src_gray, 2) - sepConv(src_gray, 4);
    cvtColor( dog, dog, CV_GRAY2RGB );

    
    
    // 2.
    //bilateralFilter(src, blurred, 11, 80, 800);
    Mat temp(nh, nw, src.type());
    Mat median;
    resize(src, temp, temp.size(),INTER_AREA);
    resize(temp, median, src.size(),INTER_NEAREST);
    
    medianBlur(median, median, median_param); // 32 for testing
    bilateralFilter(src, blurred, 11, 80, 800);
    imshow("test", blurred);
    
    // edge darken
    Mat laplacian,tmp;
    Laplacian( blurred, tmp, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( tmp, laplacian );
    blurred = blurred * 0.9 - laplacian * 0.1;
    
    
    // edge darken for median
    Laplacian( median, tmp, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( tmp, laplacian );
    median = median - sepConv(laplacian, 3);

    
    //    Mat dog = sepConv(blurred, 1) - sepConv(blurred, 4);
    //    imshow( "DoG", dog );
    
    res = median;
    res -= dog * line_param; // 2 for testing
    
    
    return res;
}

Mat filter3(Mat src, int median_param, int k, int line_param)
{
    //
    Mat src_gray, blurred, dst, res;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    if( !src.data ) return res;
    
    if (src.cols > 512) {
        float h, w, nw, nh;
        h = src.rows;
        w = src.cols;
        nw = 512;
        nh = (h/w)*nw; // Maintain aspect ratio based on a desired width.
        
        resize(src, src, Size(nw,nh),INTER_AREA);
    }
    
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
    
    
    // 14th,Feb,2014
    Mat tozero_img;
    //boxFilter(src, tozero_img, src.type(), Size(2,2), Point(-1,-1), false);

    //threshold(src_gray, tozero_img, 0, 255, cv::THRESH_TOZERO|cv::THRESH_OTSU);

    Mat dog = sepConv(src_gray, 1) - sepConv(src_gray, 4);
    dog=~dog;
    dog = src_gray -dog;
    cvtColor( dog, dog, CV_GRAY2RGB );
    


    
    Mat lap, temp_line;
    cvtColor( blurred, temp_line, CV_RGB2GRAY );
    Laplacian( temp_line, lap, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( lap, dog );
    dog=~dog;
    dog = temp_line -dog;
    cvtColor( dog, dog, CV_GRAY2RGB );


//    GaussianBlur( src, gauss, Size(3,3), 0, 0, BORDER_DEFAULT );
//    Laplacian( gauss, gauss, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
//    convertScaleAbs( gauss, gauss );
    
    

    //cvtColor( lap, lap, CV_GRAY2RGB );
    
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
    medianBlur(blurred, median, median_param); // 32 for testing
    
    
    // 3.
    Mat reshaped_image = blurred.reshape(1, blurred.cols * blurred.rows);
    Mat reshaped_image32f;
    reshaped_image.convertTo(reshaped_image32f, CV_32FC1, 1.0 / 255.0);
    
    
    Mat labels;
    int cluster_number = k; // 5 for testing
    TermCriteria criteria {TermCriteria::COUNT, 100, 1};
    Mat centers;
    //kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, KMEANS_RANDOM_CENTERS, centers);
    kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, KMEANS_PP_CENTERS, centers);
    
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
    
    res = median * 0.5 + rgb_image * 0.7;
    imshow( "no line", res);

    res -= dog * line_param; // 2 for testing
    
    Mat res_comp = median * 0.5 + rgb_image * 0.7;
    //imshow( "test2", res_comp);

    imshow( "test1", res);

    return res;
}


Mat dottize(Mat src, int width)
{
    Mat res, tmp;
    
    
    
    float h, w, nw, nh;
    h = src.rows;
    w = src.cols;
    nw = width ? width :256;
    nh = (h/w)*nw; // Maintain aspect ratio based on a desired width.
    
    resize(src, src, Size(nw,nh),INTER_AREA);
    
    
    
    Mat reshaped_image = src.reshape(1, src.cols * src.rows);
    Mat reshaped_image32f;
    reshaped_image.convertTo(reshaped_image32f, CV_32FC1, 1.0 / 255.0);
    
    
    Mat labels;
    int cluster_number = 16; // SFC palette color #
    TermCriteria criteria {TermCriteria::COUNT, 100, 1};
    Mat centers;
    //kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, KMEANS_RANDOM_CENTERS, centers);
    kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, KMEANS_PP_CENTERS, centers);
    
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
    
    cvtColor(rgb_image, tmp, CV_RGB2BGR555);
    cvtColor(tmp, res, CV_BGR5552RGB);

    
    return res;
}

/** @function main */
int main( int argc, char** argv )
{

#if CAMERA
    VideoCapture cap(0); // デフォルトカメラをオープン
    if(!cap.isOpened())  // 成功したかどうかをチェック
    return -1;
    
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    
    Mat paint;
    Mat input_image;
    namedWindow("cam",1);
    int count = 0;
    for(;;)
    {
        cv::Mat frame;
        cap >> frame; // カメラから新しいフレームを取得
        input_image=frame;      //matのコピーは普通に=で結べば良いみたい．
        //imshow("cam", filter(input_image, 11, 9, 3, true, true));
        //imshow("cam", ~getDrawingLine(input_image));
        if(count < 1)
        {
            paint = filter(dottize(input_image, 160), 31, 5, 2, false, true);
            resize(paint, paint, Size(320,240),INTER_NEAREST);
        }
        count++;
        if(count > 2) count = 0;
        Mat line = getDrawingLine(input_image);
        imshow("cam", paint * 1.5 - line);
        if(waitKey(30) >= 0) break;
    }
#endif

    const char* window_name = "Laplace Demo";
    
    /// Load an image
    //src = imread( argv[1] );
    //src = imread("/Users/naoishinichirou/Downloads/royal_ascot_horses.jpg");
    //Mat src = imread("/Users/naoishinichirou/Downloads/california_t.jpeg");
    //Mat src = imread("/Users/naoishinichirou/Downloads/py_cat.jpg");
    //Mat src = imread("/Users/naoishinichirou/Downloads/lenna.png");
    //Mat src = imread("/Users/naoishinichirou/Downloads/sundown-paris.jpg");
    //Mat src = imread("/Users/naoishinichirou/Downloads/py_k5ii_03.jpg");
    //Mat src = imread("/Users/naoishinichirou/Downloads/py_k5ii_06.jpg");

    //imshow( window_name, filter(src, 31, 5, 2, true, false));
    //imshow( "filter2", filter2(src, 31, 5, 2));
    //filter2(src, 31, 5, 2);
    //filter3(src, 31, 5, 2);
    
    
    //imshow( "test", filter(imread("/Users/naoishinichirou/Downloads/dcw_nishikawa_nara.jpg"), 31, 5, 2, false, true));
    //imshow( "test", dottize(imread("/Users/naoishinichirou/Downloads/py_k5ii_06.jpg")));
    //imwrite("/Users/naoishinichirou/Downloads/output4.png", dottize(filter(imread("/Users/naoishinichirou/Downloads/py_k5ii_03.jpg"), 31, 5, 2, false, true)));
    
    //imshow("test", dottize(filter(imread("/Users/naoishinichirou/Downloads/py_k5ii_06.jpg"), 31, 5, 2, false, true), 256));
    
//    // 逆にしてみた -> イマイチ?
//    imshow("test", filter(dottize(imread("/Users/naoishinichirou/Downloads/lenna.png")), 31, 5, 2, false, true));

    /*
    // 25th Feb 14
    Mat src = imread("/Users/naoishinichirou/Downloads/sundown-paris.jpg");
    Mat line = getDrawingLine(src);
    Mat linedots = dottize(line, 256);
    Mat paintdots = dottize(filter(src, 31, 5, 2, false, true), 256);
    imshow("test", paintdots + linedots);
     */
    
//    // 26th feb 14
//    Mat src = imread("/Users/naoishinichirou/Downloads/sundown-paris.jpg");
//    Mat line = getDrawingLine(src);
//    imshow("test",line);
    
    
    waitKey(0);
    
    return 0;
}
