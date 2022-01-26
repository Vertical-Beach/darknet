
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include "BYTETracker.h"

#ifdef DPU
#include "yolorunner_dpu.h"
#else
#include "yolorunner.h"
#define YOLOV4_NMS_THRESH  0.45 // nms threshold
#define YOLOV4_CONF_THRESH 0.1 // threshold of bounding box prob
#endif

#define MIN_BOX_AREA 1024
#define FPS 5

Mat draw_detections(cv::Mat img, vector<vector<Object>> objects){
    for(auto category_objects: objects){
        for(auto obj: category_objects){
            Scalar col(0, 255, 0);
            rectangle(img, obj.rect, col, 2);
        }
    }
    return img;
}

string my_basename(string path) {
    string res = path.substr(path.find_last_of('/') + 1);
    cout << res << endl;
    res = res.substr(0, res.find_last_of('.'));
    cout << res << endl;
    return res;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [videopath] [modelconfig] [modelfile] \n", argv[0]);
        return -1;
    }
    #ifdef DPU
    YoloRunner runner = YoloRunner(argv[2], argv[3]);
    #else
    YoloRunner runner = YoloRunner(argv[2], argv[3], YOLOV4_NMS_THRESH, YOLOV4_CONF_THRESH);
    #endif
    const char* videopath = argv[1];

    VideoCapture cap(videopath);
	if (!cap.isOpened())
		return 0;

	int img_w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CV_CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CV_CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;

    #ifdef VIDEO_WRITE
    VideoWriter writer("demo.mp4", CV_FOURCC('m', 'p', '4', 'v'), fps, Size(img_w, img_h*2));
    #endif

    vector<ofstream> detection_writers;
    vector<ofstream> tracking_writers;
    for(int i = 0; i < runner.class_num; i++){
        string detection_result_path = (string)"./results/" + my_basename((string)argv[1]) + (string)"_detection_" + to_string(i) + (string)".txt";
        string tracking_result_path = (string)"./results/" + my_basename((string)argv[1])  + (string)"_tracking_" + to_string(i) + (string)".txt";
        detection_writers.push_back(ofstream(detection_result_path));
        tracking_writers.push_back(ofstream(tracking_result_path));
        cout << detection_result_path << endl;
        cout << tracking_result_path << endl;
    }
    Mat img;
    vector<BYTETracker> trackers(runner.class_num);
    for(int i = 0; i < runner.class_num; i++) trackers[i] = BYTETracker(fps, FPS);
    int num_frames = 0;
    int total_ms = 1;
    for (;;)
    {
        if(!cap.read(img)) break;
        if (img.empty()) break;
        num_frames ++;
        cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;

        vector<vector<Object>> objects = runner.Run(img);
        Mat img2 = img.clone();
        img2 = draw_detections(img2, objects);
        for(int track_class = 0; track_class < runner.class_num; track_class++){
            auto class_objects = objects[track_class];
            for(Object class_object: class_objects){
                detection_writers[track_class] << num_frames-1 << " " <<  track_class << " " << class_object.prob << " ";
                detection_writers[track_class] << class_object.rect.x << " " << class_object.rect.y << " " << class_object.rect.width << " " << class_object.rect.height << endl;
            }

            auto start = chrono::system_clock::now();
            vector<STrack> output_stracks = trackers[track_class].update(class_objects);
            auto end = chrono::system_clock::now();
            total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();
            for (int i = 0; i < output_stracks.size(); i++)
            {
                vector<float> tlwh = output_stracks[i].tlwh;
                float area = tlwh[2] * tlwh[3];
                if (area > MIN_BOX_AREA){
                    Scalar s = trackers[track_class].get_color(output_stracks[i].track_id);
                    putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5),
                            0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                    tracking_writers[track_class] << num_frames-1 << " " << output_stracks[i].track_id << " " << track_class << " " << tlwh[0] << " " << tlwh[1] << " " << tlwh[2] << " " << tlwh[3] << endl;
                }
            }
            putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()),
                    Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        }
        Mat vstackimg(img.rows + img2.rows, img.cols, CV_8UC3);
        Mat roi1(vstackimg, Rect(0, 0, img.cols, img.rows));
        img.copyTo(roi1);
        Mat roi2(vstackimg,Rect(0, img.rows, img2.cols, img2.rows));
        img2.copyTo(roi2);

        #ifdef VIDEO_WRITE
        writer.write(vstackimg);
        #endif
    }
    cap.release();
    for(int i = 0; i < runner.class_num; i++) detection_writers[i].close();
    for(int i = 0; i < runner.class_num; i++) tracking_writers[i].close();
    cout << "FPS: " << num_frames * 1000000 / total_ms << endl;

    return 0;
}
