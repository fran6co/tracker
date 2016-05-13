//
// Created by Francisco Facioni on 12/5/16.
//

#ifndef TRACKER_TRACKER_H
#define TRACKER_TRACKER_H

#include <memory>
#include <vector>
#include <opencv2/core.hpp>

class Tracker {
    class Impl;
    std::shared_ptr<Impl> impl;
public:
    Tracker(double blobMinSize);

    std::vector<cv::Rect> track(const cv::Mat& foreground);
};


#endif //TRACKER_TRACKER_H
