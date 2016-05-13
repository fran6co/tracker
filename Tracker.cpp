//
// Created by Francisco Facioni on 12/5/16.
//

#include "Tracker.h"

#include <opencv2/imgproc.hpp>

class Blob::Impl {
public:

    Impl(const cv::Rect& rect)
            : rect(rect)
            , id(lastId++)
    {
    }

    cv::Rect rect;
    uint64_t id;
    static uint64_t lastId;
};

uint64_t Blob::Impl::lastId = 0;

Blob::Blob(const cv::Rect& rect)
    : impl(new Impl(rect))
{
}

const cv::Rect& Blob::getBoundingRect() const {
    return impl->rect;
}

uint64_t Blob::getId() const {
    return impl->id;
}

class Tracker::Impl {
public:
    Impl(double blobMinSize)
            : blobMinSize(blobMinSize) {

    }

    std::vector<Blob::Ptr> track(const cv::Mat& foreground) {
        std::vector<Blob::Ptr> blobs;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(foreground, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for(const auto& contour: contours) {
            double area = cv::contourArea(contour);
            if (area >= blobMinSize) {
                blobs.emplace_back(new Blob(cv::boundingRect(contour)));
            }
        }

        return blobs;
    }

    double blobMinSize;
};

Tracker::Tracker(double blobMinSize)
        : impl(new Impl(blobMinSize)) {

}

std::vector<Blob::Ptr> Tracker::track(const cv::Mat& foreground) {
    return impl->track(foreground);
}
