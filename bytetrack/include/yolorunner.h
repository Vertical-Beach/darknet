#include <vector>
#include <iostream>
#include "darknet.h"
#include "BYTETracker.h"

using namespace cv;
using namespace std;

//https://toriten1024.hatenablog.com/entry/2021/02/02/043513
image Mat2Image(cv::Mat src){
    cv::Mat FMat;
    src.convertTo(FMat, CV_32FC3, 1.0 / 255);
    std::vector<cv::Mat> tmp;
    cv::split(FMat, tmp);
    int size = FMat.size().width * FMat.size().height;
    int fsize = size * sizeof(float);
    image retval = make_image(FMat.size().width, FMat.size().height, 3);
    float *p = retval.data;
    memcpy((unsigned char *)p, tmp[2].data, fsize);
    p += size;
    memcpy((unsigned char *)p, tmp[1].data, fsize);
    p += size;
    memcpy((unsigned char *)p, tmp[0].data, fsize);
    return retval;
}

class YoloRunner{
    public:
    network net;
    int class_num;
    public: YoloRunner(char* modelconfig_path, char* modelfile_path){
        this->net = *load_network_custom(modelconfig_path, modelfile_path, 0, 1);
        layer l = net.layers[net.n - 1];
        this->class_num = l.classes;
    }
    public: ~YoloRunner(){
        free_network(this->net);
    }
    public: vector<vector<Object>> Run(cv::Mat cvimg){
        vector<vector<Object>> detection_results(this->class_num);

        //param
        float thresh = 0.5f;
        float hier_thresh = 0.5f;
        float nms = 0.4f;

        //preprocess
        image im = Mat2Image(cvimg);
        image resized = resize_image(im, net.w, net.h);
        //run
        network_predict(net, resized.data);
        int nboxes;
        const int letter_box = 0; // keep ratio
        detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
        //nms
        do_nms_sort(dets, nboxes, this->class_num, nms);

        //remove negatives(cf. darknet.py)
        for(int i = 0; i < nboxes; i++){
            detection det = dets[i];
            for(int j = 0; j < this->class_num; j++){
                if(det.prob[j] > 0){
                    Object obj;
                    float xmin = (det.bbox.x - det.bbox.w / 2) * cvimg.cols;
                    float ymin = (det.bbox.y - det.bbox.h / 2) * cvimg.rows;
                    float box_w = det.bbox.w * cvimg.cols;
                    float box_h = det.bbox.h * cvimg.rows;
                    obj.rect = cv::Rect(xmin, ymin, box_w, box_h);
                    obj.label = j;
                    obj.prob = det.prob[j];
                    detection_results[j].push_back(obj);
                }
            }
        }

        //free
        free_detections(dets, nboxes);
        free_image(im);
        free_image(resized);

        return detection_results;
    }

};
