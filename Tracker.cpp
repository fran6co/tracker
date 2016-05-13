//
// Created by Francisco Facioni on 12/5/16.
//

#include "Tracker.h"

#include <opencv2/imgproc.hpp>

class Tracker::Impl {
public:
    Impl(double blobMinSize)
            : blobMinSize(blobMinSize) {

    }

    std::vector<cv::Rect> track(const cv::Mat& foreground) {
        std::vector<cv::Rect> blobs;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(foreground, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for(const auto& contour: contours) {
            double area = cv::contourArea(contour);
            if (area >= blobMinSize) {
                blobs.push_back(cv::boundingRect(contour));
            }
        }

        return blobs;
    }

    double blobMinSize;
};

Tracker::Tracker(double blobMinSize)
        : impl(new Impl(blobMinSize)) {

}

std::vector<cv::Rect> Tracker::track(const cv::Mat& foreground) {
    return impl->track(foreground);
}
