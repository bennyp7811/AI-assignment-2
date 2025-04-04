#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utils/logger.hpp" // utils::logging::LOG_LEVEL_WARNING
#include <iostream>
#include <filesystem>
#include <random>
#include <vector>

// g++ -std=c++17 face-rec.cpp -lopencv_face -lopencv_core -lopencv_imgcodecs

cv::Rect roiRect(250, 200, 200, 200);  // Initial position and size of ROI
bool dragging = false;
cv::Point dragOffset;

// Mouse click function to move ROI
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        // If mouse click is inside the ROI, enable dragging and record offset
        if (roiRect.contains(cv::Point(x, y))) {
            dragging = true;
            dragOffset = cv::Point(x - roiRect.x, y - roiRect.y);
        }
    }
    else if (event == cv::EVENT_MOUSEMOVE && dragging) {
        // Update the ROI rectangle's position while dragging
        roiRect.x = x - dragOffset.x;
        roiRect.y = y - dragOffset.y;
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        // Stop dragging when the mouse button is released
        dragging = false;
    }
}

int main(int argc, char* argv[])
{
    namespace fs = std::filesystem;

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    cv::Mat frame;
    double fps = 30;
    const char win_name[] = "Live Video...";
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    cv::Mat grey_roi;

    // Iterate through all subdirectories, looking for .pgm files
    fs::path p(argc > 1 ? argv[1] : "../../att_faces");
    for (const auto& entry : fs::recursive_directory_iterator{ p }) {
        if (fs::is_regular_file(entry.status())) { // Was once always (wrongly) false in VS
            if (entry.path().extension() == ".pgm") {
                std::string str = entry.path().parent_path().stem().string(); // e.g. s26, s27, etc.
                int label = atoi(str.c_str() + 1); // "s1" -> 1
                images.push_back(cv::imread(entry.path().string().c_str(), cv::IMREAD_GRAYSCALE));
                labels.push_back(label);
            }
        }
    }
    std::cout << "Wait 60 secs. for camera access to be obtained..." << std::endl;
    cv::VideoCapture vid_in(0);   // Open the default camera

    if (vid_in.isOpened())
    {
        std::cout << "Camera capture obtained. Drag square around face and click space to search." << std::endl;
    }
    else
    {
        std::cerr << "error: Camera 0 could not be opened for capture.\n";
        return -1;
    }

    // Create display window and register the mouse callback for ROI movement
    cv::namedWindow(win_name);
    cv::setMouseCallback(win_name, onMouse, nullptr);

    int i{ 0 };

    while (1) {
        vid_in >> frame;
        if (frame.empty()) break;

        // Create a copy of the frame to apply greyscale
        cv::Mat displayFrame = frame.clone();
        // If the ROI is within frame bounds, convert it to greyscale for display
        if (roiRect.x >= 0 && roiRect.y >= 0 &&
            roiRect.x + roiRect.width <= displayFrame.cols &&
            roiRect.y + roiRect.height <= displayFrame.rows) {
            cv::Mat roi(displayFrame, roiRect);
            cv::Mat grayROI;
            cv::cvtColor(roi, grayROI, cv::COLOR_BGR2GRAY);
            cv::cvtColor(grayROI, grayROI, cv::COLOR_GRAY2BGR);
            grayROI.copyTo(displayFrame(roiRect));
        }

        // Draw the ROI rectangle (using the updated, movable roiRect)
        cv::rectangle(displayFrame, roiRect, cv::Scalar{ 0, 0, 255 });
        cv::imshow(win_name, displayFrame);

        int code = cv::waitKey(1000 / fps); // wait time in msecs
        if (code == 27) // Escape key to exit
            break;
        else if (code == 32) // Space key triggers face recognition
        {
            // Ensure ROI is within frame bounds
            if (roiRect.x >= 0 && roiRect.y >= 0 &&
                roiRect.x + roiRect.width <= frame.cols &&
                roiRect.y + roiRect.height <= frame.rows) {

                // Extract the ROI from the frame
                cv::Mat roi(frame, roiRect), grey_roi;
                cv::cvtColor(roi, grey_roi, cv::COLOR_BGR2GRAY);

                // Resize ROI to 92 x 112 required for recognition
                cv::Mat small_roi;
                cv::resize(grey_roi, small_roi, cv::Size(92, 112));

                std::cout << "\nThinking...";
                cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
                model->train(images, labels);
                int predictedLabel = model->predict(small_roi);

                std::cout << "\nPredicted class = " << predictedLabel << '\n';
            }
            else {
                std::cerr << "Error: ROI is out of bounds." << std::endl;
            }
        }
    }

    vid_in.release();
    return 0;
}