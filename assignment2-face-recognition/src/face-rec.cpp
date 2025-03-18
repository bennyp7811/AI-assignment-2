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

int main(int argc, char *argv[])
{
  namespace fs = std::filesystem;

  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

  std::vector<cv::Mat> images;
  std::vector<int>     labels;

  // Iterate through all subdirectories, looking for .pgm files
  fs::path p(argc > 1 ? argv[1] : "../../att_faces");
  for (const auto &entry : fs::recursive_directory_iterator{ p }) {
    if (fs::is_regular_file(entry.status())) { // Was once always (wrongly) false in VS
      if (entry.path().extension() == ".pgm") {
        std::string str = entry.path().parent_path().stem().string(); // s26 s27 etc.
        int label = atoi(str.c_str() + 1); // s1 -> 1 (pointer arithmetic)
        images.push_back(cv::imread(entry.path().string().c_str(), cv::IMREAD_GRAYSCALE));
        labels.push_back(label);
      }
    }
  }

// Randomly choose an image and test the system
  std::random_device dev{};
  std::mt19937 generator{dev()};
  std::uniform_int_distribution<int> dist{0, static_cast<int>(images.size() - 1)};
  int rand_image_id = dist(generator); // random image id

  cv::Mat testSample = images[rand_image_id];
  int     testLabel  = labels[rand_image_id];
  std::cout << "Actual class    = " << testLabel << '\n';
  std::cout << " training...";

  cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
  model->train(images, labels);
  int predictedLabel = model->predict(testSample);
  std::cout << "\nPredicted class = " << predictedLabel << '\n';

  return 0;
}
