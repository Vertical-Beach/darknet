OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
g++ -std=c++17 -O3 -DDPU -o bytetrack src/*.cpp -lglog -lvitis_ai_library-xnnpp -lvitis_ai_library-model_config -lprotobuf -lvitis_ai_library-dpu_task ${OPENCV_FLAGS} -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -I ./include -I /usr/local/include/eigen3/
