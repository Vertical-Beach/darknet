
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include "BYTETracker.h"
#include "yolorunner.h"

#define YOLOX_NMS_THRESH  0.7 // nms threshold
#define YOLOX_CONF_THRESH 0.1 // threshold of bounding box prob
#define INPUT_W 1088  // target image size w after resize
#define INPUT_H 608   // target image size h after resize

Mat static_resize(Mat& img) {
    float r = min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    Mat re(unpad_h, unpad_w, CV_8UC3);
    resize(img, re, re.size());
    Mat out(INPUT_H, INPUT_W, CV_8UC3, Scalar(114, 114, 114));
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));
    return out;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [videopath]\n", argv[0]);
        return -1;
    }

    YoloRunner runner = YoloRunner("../yolov4_tiny/yolov4-tiny-custom.cfg", "../yolov4_tiny/yolov4-tiny-custom_best.weights");

    const char* videopath = argv[1];

    VideoCapture cap(videopath);
	if (!cap.isOpened())
		return 0;

	int img_w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CV_CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CV_CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;

    VideoWriter writer("demo.mp4", CV_FOURCC('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

    Mat img;
    vector<BYTETracker> trackers(runner.class_num);
    for(int i = 0; i < runner.class_num; i++) trackers[i] = BYTETracker(fps, 5);
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
        for(int track_class = 0; track_class < runner.class_num; track_class++){
            auto class_objects = objects[track_class];
            // auto tracker = trackers[track_class];

            auto start = chrono::system_clock::now();
            // vector<STrack> output_stracks = tracker.update(class_objects);
            vector<STrack> output_stracks = trackers[track_class].update(class_objects);
            auto end = chrono::system_clock::now();
            total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();
            for (int i = 0; i < output_stracks.size(); i++)
            {
                vector<float> tlwh = output_stracks[i].tlwh;
                bool vertical = tlwh[2] / tlwh[3] > 1.6;
                if (tlwh[2] * tlwh[3] > 20 && !vertical)
                {
                    // Scalar s = tracker.get_color(output_stracks[i].track_id);
                    Scalar s = trackers[track_class].get_color(output_stracks[i].track_id);
                    putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5),
                            0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                    rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
                }
            }
            putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()),
                    Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        }
        writer.write(img);
        char c = waitKey(1);
        if (c > 0)
        {
            break;
        }
    }
    cap.release();
    cout << "FPS: " << num_frames * 1000000 / total_ms << endl;

    return 0;
}
