#include <sys/stat.h>
#include <glob.h>

#include <iostream>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include "Tracker.h"

namespace filesystem {
    inline bool exists(const std::string &name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }

    inline std::vector<std::string> glob(const std::string& pat){
        using namespace std;
        glob_t glob_result;
        glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
        std::vector<string> ret;
        for(unsigned int i=0;i<glob_result.gl_pathc;++i){
            ret.push_back(string(glob_result.gl_pathv[i]));
        }
        globfree(&glob_result);
        return ret;
    }
}

void generateCalibration(const std::string& calibrationVideo, const std::string& calibrationFile) {
    cv::Size patternSize (5, 4);

    std::vector<cv::Point3f> patternCornersTemplate;
    for( int i = 0; i < patternSize.height; ++i ) {
        for (int j = 0; j < patternSize.width; ++j) {
            patternCornersTemplate.push_back(cv::Point3f(j, i, 0));
        }
    }

    cv::VideoCapture stream(calibrationVideo);
    cv::Size size(stream.get(cv::CAP_PROP_FRAME_WIDTH), stream.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::vector<std::vector<cv::Point3f>> patternCorners;
    std::vector<std::vector<cv::Point2f>> imageCorners;

    std::cout << "Looking for calibration pattern" << std::endl;
    // HARDCODE: I'm assuming that the chess board stays roughly in the middle
    cv::Rect roi (100, 100, size.width-200, size.height-200);
    cv::Mat frame, gray;
    while (stream.read(frame)) {
        cv::Mat cropped (size, CV_8U, 255);
        cv::cvtColor(frame(roi), gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, cropped(roi), 100, 255, cv::THRESH_BINARY);

        std::vector<cv::Point2f> corners;
        bool patternFound = cv::findChessboardCorners(cropped, patternSize, corners, 0);

        if (patternFound) {
            cv::cornerSubPix(cropped, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

            patternCorners.push_back(patternCornersTemplate);
            imageCorners.push_back(corners);
        }
    }

    std::cout << "Calibrating" << std::endl;
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F), distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F);
    cv::calibrateCamera(patternCorners, imageCorners, size, cameraMatrix, distortionCoefficients, rvecs, tvecs);

    std::cout << "Saving" << std::endl;
    cv::FileStorage calibration (calibrationFile, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);

    calibration << "camera_matrix" << cameraMatrix;
    calibration << "distortion_coefficients" << distortionCoefficients;
    calibration << "size" << size;
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        throw std::runtime_error("Usage:\n"
         "      tracker <data_path> <calibration_video>");
    }

    const std::string dataPath (argv[1]);
    const std::string calibrationVideo (dataPath + "/" + argv[2]);
    const std::string calibrationFile = dataPath + "/calibration.yaml";

    if (!filesystem::exists(calibrationFile)) {
        generateCalibration(calibrationVideo, calibrationFile);
    }

    cv::FileStorage calibration (calibrationFile, cv::FileStorage::READ);

    cv::Size size;
    cv::Mat cameraMatrix, distortionCoefficients;
    calibration["camera_matrix"] >> cameraMatrix;
    calibration["distortion_coefficients"] >> distortionCoefficients;
    calibration["size"] >> size;

    cv::Mat map1, map2;
    cv::Mat optimalCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, size, 1);
    cv::initUndistortRectifyMap(cameraMatrix, distortionCoefficients, cv::Mat(), optimalCameraMatrix, size, CV_16SC2, map1, map2);

    // HARDCODE: Minimum time to consider a blob active
    std::chrono::seconds minTimeAlive (5);
    for(const std::string& video: filesystem::glob(dataPath + "/*.mp4")) {
        if (video != calibrationVideo) {
            cv::Ptr<cv::BackgroundSubtractor> backgroundSegmentator = cv::createBackgroundSubtractorMOG2(500, 350, true);

            // HARDCODE: Just to remove some noisy artifacts we set the min area to 1000
            Tracker tracker (1000, 0.5, std::chrono::seconds(1));

            cv::VideoCapture stream(video);

            // Force learning the first frame as background
            double learningRate = 1;
            cv::Mat frame, foreground, out;
            while (stream.read(frame)) {
                std::chrono::milliseconds timestamp (uint64_t(stream.get(cv::CAP_PROP_POS_MSEC)));
                cv::remap(frame, frame, map1, map2, cv::INTER_LINEAR);

                backgroundSegmentator->apply(frame, foreground, learningRate);

                if (learningRate == 1) {
                    learningRate = 0.0001;
                }

                cv::threshold(foreground, foreground, 150, 255, cv::THRESH_BINARY);
                cv::erode(foreground, foreground, cv::Mat(), cv::Point(-1, -1), 2);
                cv::blur(foreground, foreground, cv::Size(11, 11));
                cv::dilate(foreground, foreground, cv::Mat(), cv::Point(-1, -1), 11);

                std::vector<Blob::Ptr> blobs = tracker.track(foreground, timestamp);

                cv::cvtColor(frame, out, cv::COLOR_BGR2HSV);
                for(Blob::Ptr blob: blobs) {
                    cv::Scalar color ((5*blob->getId())%180, 255, 255);

                    cv::rectangle(out, blob->getBoundingRect(), color, blob->getTimeAlive() < minTimeAlive?1:5);

                    cv::circle(out, cv::Point(blob->getLastPrediction().x+blob->getLastPrediction().width/2, blob->getLastPrediction().y+blob->getLastPrediction().height/2), 5, color);

                    std::vector<cv::Point> points;
                    for (const auto& history: blob->getHistory()) {
                        points.emplace_back(history.second.x + history.second.width / 2, history.second.y + history.second.height / 2);
                    }
                    const cv::Point *data = points.data();
                    const int size = points.size();
                    cv::polylines(out, &data, &size, 1, false, color);
                }
                cv::cvtColor(out, frame, cv::COLOR_HSV2BGR);

                cv::imshow(video, frame);
                cv::waitKey(1);
            }
        }
    }

    return 0;
}