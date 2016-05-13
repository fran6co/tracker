//
// Created by Francisco Facioni on 12/5/16.
//

#include "Tracker.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "HungarianMatching.h"

class Blob::Impl {
public:
    Impl(const cv::Rect& rect, const std::chrono::nanoseconds& timestamp, double accelerationNoise)
            : rect(rect)
            , id(lastId++)
            , accelerationNoise(accelerationNoise)
            , tracker(4, 2, 0)
    {
        cv::setIdentity(tracker.transitionMatrix);

        cv::Point2f position (rect.x + rect.width/2, rect.y + rect.height/2);

        tracker.statePre.at<float>(0) = position.x;
        tracker.statePre.at<float>(1) = position.y;
        tracker.statePre.at<float>(2) = 0;
        tracker.statePre.at<float>(3) = 0;

        tracker.statePost.at<float>(0) = position.x;
        tracker.statePost.at<float>(1) = position.y;

        cv::setIdentity(tracker.measurementMatrix);
        cv::setIdentity(tracker.measurementNoiseCov, cv::Scalar::all(0.1));
        cv::setIdentity(tracker.processNoiseCov);
        cv::setIdentity(tracker.errorCovPost, cv::Scalar::all(.1));

        history.emplace_back(timestamp, rect);
    }

    void update(const cv::Rect& track, const std::chrono::nanoseconds& timestamp) {
        cv::Mat measurement(2,1,CV_32FC1);
        measurement.at<float>(0) = track.x + track.width/2;
        measurement.at<float>(1) = track.y + track.height/2;

        cv::Mat correct = tracker.correct(measurement);

        rect = track;

        history.emplace_back(timestamp, cv::Rect(correct.at<float>(0)-rect.width/2, correct.at<float>(1)-rect.height/2, rect.width, rect.height));
    }

    cv::Rect predict(const std::chrono::nanoseconds& timestamp) {
        std::chrono::duration<double> _deltaTime = timestamp - history.back().first;
        double deltaTime = _deltaTime.count();

        tracker.transitionMatrix = (cv::Mat_<float>(4, 4) <<
                1 ,0 ,deltaTime ,0,
                0 ,1 ,0         ,deltaTime,
                0 ,0 ,1         ,0,
                0 ,0 ,0         ,1
        );

        tracker.processNoiseCov = (cv::Mat_<float>(4, 4) <<
                std::pow(deltaTime,4.0)/4.0 ,0                           ,std::pow(deltaTime,3.0)/2.0 ,0,
                0                           ,std::pow(deltaTime,4.0)/4.0 ,0                           ,std::pow(deltaTime,3.0)/2.0,
                std::pow(deltaTime,3.0)/2.0 ,0                           ,std::pow(deltaTime,2.0)     ,0,
                0                           ,std::pow(deltaTime,3.0)/2.0 ,0                           ,std::pow(deltaTime,2.0)
        );

        tracker.processNoiseCov *= accelerationNoise;

        cv::Mat prediction = tracker.predict();

        return cv::Rect(prediction.at<float>(0)-rect.width/2, prediction.at<float>(1)-rect.height/2, rect.width, rect.height);
    }

    cv::Rect rect;
    uint64_t id;
    double accelerationNoise;

    static uint64_t lastId;

    cv::KalmanFilter tracker;

    std::vector<History> history;
};

uint64_t Blob::Impl::lastId = 0;

Blob::Blob(const cv::Rect& rect, const std::chrono::nanoseconds& timestamp, double accelerationNoise)
    : impl(new Impl(rect, timestamp, accelerationNoise))
{
}

const cv::Rect& Blob::getBoundingRect() const {
    return impl->rect;
}

uint64_t Blob::getId() const {
    return impl->id;
}

std::vector<Blob::History> Blob::getHistory() const {
    return impl->history;
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
    Impl(double blobMinSize, double accelerationNoise)
            : blobMinSize(blobMinSize)
            , accelerationNoise(accelerationNoise)
    {
    }

    std::vector<Blob::Ptr> track(const cv::Mat& foreground, const std::chrono::nanoseconds& timestamp) {
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
            cv::Rect prediction = previousBlob->impl->predict(timestamp);
            for(int i=0;i<blobs.size();i++){
                const cv::Rect& blob = blobs[i];
                distance.at<int32_t>(i, j) = squaredRectDistance(blob, prediction);
            }
        }

        // Cap the distances, to avoid matching things too far away
        const double maximumDistance = std::sqrt(blobMinSize/M_PI)*10;
        auto matches = hungarianMatching(distance, maximumDistance*maximumDistance);

        for (auto match: matches) {
            Blob::Ptr previousBlob = previousBlobs[match.second];
            previousBlob->impl->update(blobs[match.first], timestamp);
            currentBlobs.push_back(previousBlob);
        }

        for(int i=0;i<blobs.size();i++) {
            if (matches.find(i) == matches.end()) {
                currentBlobs.emplace_back(new Blob(blobs[i], timestamp, accelerationNoise));
            }
        }

        return currentBlobs;
    }

    double blobMinSize;
    double accelerationNoise;
    std::vector<Blob::Ptr> previousBlobs, currentBlobs;
};

Tracker::Tracker(double blobMinSize, double accelerationNoise)
        : impl(new Impl(blobMinSize, accelerationNoise)) {

}

std::vector<Blob::Ptr> Tracker::track(const cv::Mat& foreground, const std::chrono::nanoseconds& timestamp) {
    return impl->track(foreground, timestamp);
}
