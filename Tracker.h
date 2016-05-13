//
// Created by Francisco Facioni on 12/5/16.
//

#ifndef TRACKER_TRACKER_H
#define TRACKER_TRACKER_H

#include <memory>
#include <vector>
#include <opencv2/core.hpp>

class Blob {
    friend class Tracker;

    class Impl;
    std::shared_ptr<Impl> impl;
public:
    typedef std::pair<std::chrono::nanoseconds, cv::Rect> History;
    typedef std::shared_ptr<Blob> Ptr;

    Blob(const cv::Rect& rect, const std::chrono::nanoseconds&, double accelerationNoise);

    const cv::Rect& getBoundingRect() const;
    uint64_t getId() const;

    std::vector<History> getHistory() const;
    std::chrono::nanoseconds getTimeAlive() const;
};

class Tracker {
    class Impl;
    std::shared_ptr<Impl> impl;
public:
    Tracker(double blobMinSize, double accelerationNoise);

    std::vector<Blob::Ptr> track(const cv::Mat& foreground, const std::chrono::nanoseconds& timestamp);
};


#endif //TRACKER_TRACKER_H
