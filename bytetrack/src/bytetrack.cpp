
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <chrono>
#include "BYTETracker.h"
#include "yolorunner.h"

#define YOLOV4_NMS_THRESH  0.45 // nms threshold
#define YOLOV4_CONF_THRESH 0.1 // threshold of bounding box prob
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
int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s [videopath] [resultpath]\n", argv[0]);
        return -1;
    }

    YoloRunner runner = YoloRunner("../yolov4_tiny/yolov4-tiny-custom.cfg", "../yolov4_tiny/yolov4-tiny-custom_best.weights", YOLOV4_NMS_THRESH, YOLOV4_CONF_THRESH);

    const char* videopath = argv[1];

    VideoCapture cap(videopath);
	if (!cap.isOpened())
		return 0;

	int img_w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CV_CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CV_CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;

    VideoWriter writer("demo.mp4", CV_FOURCC('m', 'p', '4', 'v'), fps, Size(img_w, img_h*2));

    ofstream ofs(argv[2]);
    Mat img;
    vector<BYTETracker> trackers(runner.class_num);
    for(int i = 0; i < runner.class_num; i++) trackers[i] = BYTETracker(fps, FPS);
    int num_frames = 0;
    int total_ms = 1;
	for (;;)
    {
        if(!cap.read(img))
            break;
        num_frames ++;
        if (num_frames % 20 == 0)
        {
            cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
		if (img.empty())
			break;

        vector<vector<Object>> objects = runner.Run(img);
        Mat img2 = img.clone();
        img2 = draw_detections(img2, objects);
        for(int track_class = 0; track_class < runner.class_num; track_class++){
            auto class_objects = objects[track_class];

            auto start = chrono::system_clock::now();
            vector<STrack> output_stracks = trackers[track_class].update(class_objects);
            auto end = chrono::system_clock::now();
            total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();
            for (int i = 0; i < output_stracks.size(); i++)
            {
                vector<float> tlwh = output_stracks[i].tlwh;
                // bool vertical = tlwh[2] / tlwh[3] > 1.6;
                // if (tlwh[2] * tlwh[3] > 20 && !vertical)
                float area = tlwh[2] * tlwh[3];
                if (area > MIN_BOX_AREA){
                    Scalar s = trackers[track_class].get_color(output_stracks[i].track_id);
                    putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5),
                            0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                    rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
                    ofs << num_frames-1 << " " << output_stracks[i].track_id << " " << track_class << " " << tlwh[0] << " " << tlwh[1] << " " << tlwh[2] << " " << tlwh[3] << endl;
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

        writer.write(vstackimg);
        char c = waitKey(1);
        if (c > 0)
        {
            break;
        }
    }
    cap.release();
    ofs.close();
    cout << "FPS: " << num_frames * 1000000 / total_ms << endl;

    return 0;
}
