//
// Created by Francisco Facioni on 12/5/16.
//

#ifndef TRACKER_TRACKER_H
#define TRACKER_TRACKER_H

#include <memory>
#include <vector>
#include <opencv2/core.hpp>

class Blob {
public:
    typedef std::shared_ptr<Blob> Ptr;

    Blob(const cv::Rect& rect)
        : rect(rect)
        , id(lastId++)
    {
    }

    const cv::Rect& getBoundingRect() const {
        return rect;
    }

    uint64_t getId() const {
        return id;
    }

    cv::Rect rect;
    uint64_t id;
    static uint64_t lastId;
};

class Tracker {
    class Impl;
    std::shared_ptr<Impl> impl;
public:
    Tracker(double blobMinSize);

    std::vector<Blob::Ptr> track(const cv::Mat& foreground);
};


#endif //TRACKER_TRACKER_H
