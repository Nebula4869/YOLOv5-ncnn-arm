#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "vulkan/vulkan.h"
#include "ncnn/net.h"

using namespace std;


const int numCLasses = 80;
const int frameWidth = 1280;
const int frameHeight = 720;

vector<vector<vector<int>>> anchors{{{10, 13}, {16, 30}, {33, 23}},
                                                                               {{30, 61}, {62, 45}, {59, 119}},
                                                                               {{116, 90}, {156, 198}, {373, 326}}};
vector<vector<float>> dets;

inline float sigmoid(float x){
    return 1.0f / (1.0f + exp(-x));
}

void decodeResult(const ncnn::Mat& data, int stride, vector<vector<int>> anchors, float scoreThresh)
{
    for (int c=0; c<data.c; c++)
    {
        const float* ptr = data.channel(c);
        for (int y=0; y<data.h; y++)
        {
            float score = sigmoid(ptr[4]);
            if (score > scoreThresh)
            {
                vector<float> det(6);
                det[1] = (sigmoid(ptr[0] )* 2 - 0.5 + y % (int)(640 / stride)) * stride * frameWidth / 640; //center_x
                det[2] = (sigmoid(ptr[1]) * 2 - 0.5 + (int)(y / (640 / stride))) * stride * frameHeight / 384; //center_y
                det[3] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[c][0] * frameWidth / 640; //w
                det[4] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[c][1] * frameHeight / 384; //h

                det[1] = det[1] - det[3] / 2; //left
                det[2] = det[2] - det[4] / 2; //top
                det[3] = det[1] + det[3]; //right
                det[4] = det[2] + det[4]; //bottom

                for (int i=5; i<numCLasses+5; i++)
                {
                    float conf = sigmoid(ptr[i]);
                    if (conf * score > det[0])
                    {
                        det[0] = conf * score; //score
                        det[5] = i - 5; //class_id
                    }
                }
                dets.push_back(det);
            }
            ptr += data.w;
        }
    }
}

void nonMaxSuppression(float iouThresh)
{
    int length = dets.size();
    int index = length - 1;

    sort(dets.begin(), dets.end());
    vector<float> areas(length);
    for (int i=0; i<length; i++)
    {
        areas[i] = (dets[i][4] - dets[i][2]) * (dets[i][3] - dets[i][1]);
    }
    
    while (index  > 0)
    {
        int i = 0;
        while (i < index)
        {
            float left = max(dets[index][1], dets[i][1]);
            float top = max(dets[index][2], dets[i][2]);
            float right = min(dets[index][3], dets[i][3]);
            float bottom = min(dets[index][4], dets[i][4]);
            float overlap = max(0.0f, right - left) * max(0.0f, bottom - top);
            if (overlap / (areas[index] + areas[i] - overlap) > iouThresh)
            {
                areas.erase(areas.begin() + i);
                dets.erase(dets.begin() + i);
                index --;
            }
            else
            {
                i++;
            }
        }
        index--;
    }
}

int main()
{
    std::vector<std::string> classnames;
	std::ifstream f("../coco.names");
	std::string name = "";
	while (std::getline(f, name))
	{
		classnames.push_back(name);
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = 1;
    net.opt.num_threads=12;
    net.load_param("../yolov5s.param");
    net.load_model("../yolov5s.bin");
    ncnn::Mat input, output0, output1, output2;

    cv::Mat frame, img;
    cv::VideoCapture cap = cv::VideoCapture(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, frameWidth);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, frameHeight);
    while(cap.isOpened())
    {
        const auto t0 = std::chrono::system_clock::now();
	    cap.read(frame);
        if(frame.empty())
        {
           std::cout << "Read frame failed!" << std::endl;
           break;
        }

        ncnn::Extractor extractor = net.create_extractor();
        cv::resize(frame, img, cv::Size(640, 384));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        const auto t1 = std::chrono::system_clock::now();

        input = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_RGB, img.cols, img.rows);
        float norm[3] = {1/255.f,1/255.f,1/255.f};
        float mean[3] = {0,0,0};
        input.substract_mean_normalize(mean,norm);
        extractor.input("images", input);
        extractor.extract("output", output0);
        extractor.extract("424", output1);
        extractor.extract("444", output2);

        const auto t2 = std::chrono::system_clock::now();

        dets.clear();
        decodeResult(output0, 8, anchors[0], 0.6);
        decodeResult(output1, 16, anchors[1], 0.6);
        decodeResult(output2, 32, anchors[2], 0.6);
        nonMaxSuppression(0.5);

        const auto t3 = std::chrono::system_clock::now();

        for (int i=0; i<dets.size(); i++)
        {
                float left = dets[i][1];
                float top = dets[i][2];
                float right = dets[i][3];
                float bottom = dets[i][4];
                float score = dets[i][0];
                int classID = dets[i][5];

				cv::rectangle(frame, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(255, 255, 0), 2);

				cv::putText(frame,
					classnames[classID] + ": " + cv::format("%.2f", score),
					cv::Point(left, top),
					cv::FONT_HERSHEY_SIMPLEX, (right - left) / 200, cv::Scalar(255, 255, 0), 2);
        }
        
        cv::imshow("", frame);
        if(cv::waitKey(1)== 27) break;

        const auto t4 = std::chrono::system_clock::now();

        cout << "Inference Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() * 1e-3 << "ms  ";
        cout << "Post-processing Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() * 1e-3 << "ms  ";
        cout << "Total Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t0).count() * 1e-3 << "ms" << endl;
    }
    return 0;
}
