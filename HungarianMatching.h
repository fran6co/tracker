//
// Created by Francisco Facioni on 12/5/16.
//

#ifndef TRACKER_HUNGARIANMATCHING_H
#define TRACKER_HUNGARIANMATCHING_H

#include <map>
#include <opencv2/core/core.hpp>

// Got it from http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html, this is O(n^4)
// It can be implemented as O(n^3) with https://github.com/maandree/hungarian-algorithm-n3
std::map<int, int> hungarianMatching(const cv::Mat& _distances, int maxDistance);

#endif //TRACKER_HUNGARIANMATCHING_H
