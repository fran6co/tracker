#include <sys/stat.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

namespace filesystem {
    inline bool exists(const std::string &name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
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

    cv::Mat frame, gray;
    while (stream.read(frame)) {
        std::vector<cv::Point2f> corners;
        bool patternFound = cv::findChessboardCorners(frame, patternSize, corners);

        if (patternFound) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

            patternCorners.push_back(patternCornersTemplate);
            imageCorners.push_back(corners);
        }
    }

    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F), distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F);
    cv::calibrateCamera(patternCorners, imageCorners, size, cameraMatrix, distortionCoefficients, rvecs, tvecs);

    cv::FileStorage calibration (calibrationFile, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);

    calibration << "camera_matrix" << cameraMatrix;
    calibration << "distortion_coefficients" << distortionCoefficients;
}


#include <stdexcept>


#include <iostream>

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

    cv::Mat cameraMatrix, distortionCoefficients;
    calibration["camera_matrix"] >> cameraMatrix;
    calibration["distortion_coefficients"] >> distortionCoefficients;

    cv::VideoCapture stream(calibrationVideo);
    cv::Size size(stream.get(cv::CAP_PROP_FRAME_WIDTH), stream.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::Mat map1, map2;
    cv::Mat optimalCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, size, 1);
    cv::initUndistortRectifyMap(cameraMatrix, distortionCoefficients, cv::Mat(), optimalCameraMatrix, size, CV_16SC2, map1, map2);

    cv::Mat frame;
    while (stream.read(frame)) {
        cv::remap(frame, frame, map1, map2, cv::INTER_LINEAR);

        cv::imshow("debug", frame);
        cv::waitKey(1);
    }

    return 0;
}