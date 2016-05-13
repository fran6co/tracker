//
// Created by Francisco Facioni on 12/5/16.
//

#include "Tracker.h"

#include <opencv2/imgproc.hpp>

#include "HungarianMatching.h"

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

// Take in account the size differences when matching
int squaredRectDistance(const cv::Rect& a, const cv::Rect& b) {
    int dx = (a.x + a.width / 2) - (b.x + b.width / 2);
    int dy = (a.y + a.height / 2) - (b.y + b.height / 2);
    int dw = a.width - b.width;
    int dh = a.height - b.height;
    int pd = dx * dx + dy * dy;
    int sd = dw * dw + dh * dh;
    return pd + sd;
}

class Tracker::Impl {
public:
    Impl(double blobMinSize)
            : blobMinSize(blobMinSize) {

    }

    std::vector<Blob::Ptr> track(const cv::Mat& foreground) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(foreground, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> blobs;
        for(const auto& contour: contours) {
            double area = cv::contourArea(contour);
            if (area >= blobMinSize) {
                blobs.push_back(cv::boundingRect(contour));
            }
        }

        previousBlobs = currentBlobs;
        currentBlobs.clear();

        cv::Mat distance (blobs.size(), previousBlobs.size(), CV_32S);
        for(int j=0;j<previousBlobs.size();j++){
            Blob::Ptr previousBlob = previousBlobs[j];
            for(int i=0;i<blobs.size();i++){
                const cv::Rect& blob = blobs[i];
                distance.at<int32_t>(i, j) = squaredRectDistance(blob, previousBlob->getBoundingRect());
            }
        }

        // Cap the distances, to avoid matching things too far away
        const double maximumDistance = std::sqrt(blobMinSize/M_PI)*10;
        auto matches = hungarianMatching(distance, maximumDistance*maximumDistance);

        for (auto match: matches) {
            currentBlobs.push_back(previousBlobs[match.second]);
        }

        for(int i=0;i<blobs.size();i++) {
            if (matches.find(i) == matches.end()) {
                currentBlobs.emplace_back(new Blob(blobs[i]));
            }
        }

        return currentBlobs;
    }

    double blobMinSize;
    std::vector<Blob::Ptr> previousBlobs, currentBlobs;
};

Tracker::Tracker(double blobMinSize)
        : impl(new Impl(blobMinSize)) {

}

std::vector<Blob::Ptr> Tracker::track(const cv::Mat& foreground) {
    return impl->track(foreground);
}
