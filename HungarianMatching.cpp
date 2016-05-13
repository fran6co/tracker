#include "HungarianMatching.h"

enum {
    UNMARKED = 0,
    MARKED,
    PRIME
};

std::map<int, int> hungarianMatching(const cv::Mat& _distances, int maxDistance) {
    cv::Mat distances = _distances.clone();

    for (int i = 0; i < distances.rows; i++) {
        int32_t *distancesRow = distances.ptr<int32_t>(i);
        for (int j = 0; j < distances.cols; j++) {
            int32_t& distance = distancesRow[j];

            if (distance > maxDistance) {
                distance = maxDistance;
            }
        }
    }

    cv::Mat assigments = distances.clone();

    // Always put as rows the biggest dimension
    bool isTransposed = assigments.cols < assigments.rows;
    if (isTransposed) {
        cv::transpose(assigments, assigments);
    }

    // Search the minimum, this should generate some 0s
    for (int i = 0; i < assigments.rows; i++) {
        int32_t* assigmentsRow = assigments.ptr<int32_t>(i);
        int32_t minDistance = std::numeric_limits<int32_t>::max();
        for(int j=0;j< assigments.cols;j++){
            if (assigmentsRow[j] < minDistance) {
                minDistance = assigmentsRow[j];
            }
        }
        for(int j=0;j< assigments.cols;j++){
            assigmentsRow[j] -= minDistance;
        }
    }

    // Step 2
    cv::Mat marks (assigments.size(), CV_8U, cv::Scalar(UNMARKED));
    // Create matches preserving the 0s
    std::vector<bool> colCovered(marks.cols, false), rowCovered(marks.rows, false);
    for (int i = 0; i < assigments.rows; i++) {
        const int32_t *assigmentsRow = assigments.ptr<int32_t>(i);
        uint8_t *marksRow = marks.ptr<uint8_t>(i);
        for (int j = 0; j < assigments.cols; j++) {
            if (!rowCovered[i] && !colCovered[j] && assigmentsRow[j] == 0) {
                colCovered[j] = true;
                rowCovered[i] = true;
                marksRow[j] = MARKED;
            }
        }
    }

    while(true) {
        // Step 3
        // Count covered columns. If we cover all the possible matches, return
        std::vector<bool> colCovered(marks.cols, false), rowCovered(marks.rows, false);
        for (int i = 0; i < marks.rows; i++) {
            const uint8_t *marksRow = marks.ptr<uint8_t>(i);
            for (int j = 0; j < marks.cols; j++) {
                if (marksRow[j] == MARKED) {
                    colCovered[j] = true;
                }
            }
        }
        int markedColumns = 0;
        for (int j = 0; j < marks.cols; j++) {
            if (colCovered[j]) {
                markedColumns++;
            }
        }
        if (markedColumns >= assigments.cols || markedColumns >= assigments.rows) {
            // That's it!
            break;
        }

        // Step4
        std::vector<cv::Point> marked;
        while (true) {
            cv::Point zero;

            bool foundAnyZero = false;
            for (int i = 0; i < assigments.rows; i++) {
                const int32_t *assigmentsRow = assigments.ptr<int32_t>(i);
                for (int j = 0; j < assigments.cols; j++) {
                    if (assigmentsRow[j] == 0 && !colCovered[j] && !rowCovered[i]) {
                        zero = cv::Point(j, i);
                        foundAnyZero = true;
                        break;
                    }
                }
                if (foundAnyZero) {
                    break;
                }
            }

            if (!foundAnyZero) {
                // Step 6
                double minDistance = std::numeric_limits<double>::max();
                for (int i = 0; i < assigments.rows; i++) {
                    if (rowCovered[i]) {
                        continue;
                    }
                    const int32_t *assigmentsRow = assigments.ptr<int32_t>(i);
                    for (int j = 0; j < assigments.cols; j++) {
                        if (!colCovered[j]) {
                            if (assigmentsRow[j] < minDistance) {
                                minDistance = assigmentsRow[j];
                            }
                        }
                    }
                }
                for (int i = 0; i < assigments.rows; i++) {
                    int32_t *assigmentsRow = assigments.ptr<int32_t>(i);
                    for (int j = 0; j < assigments.cols; j++) {
                        if (rowCovered[i]) {
                            assigmentsRow[j] += minDistance;
                        }
                        if (!colCovered[j]) {
                            assigmentsRow[j] -= minDistance;
                        }
                    }
                }
                continue;
            } else {
                marks.at<uint8_t>(zero.y, zero.x) = PRIME;

                bool foundMarkInRow = false;
                const uint8_t *marksRow = marks.ptr<uint8_t>(zero.y);
                for (int j = 0; j < marks.cols; j++) {
                    if (marksRow[j] == MARKED) {
                        zero.x = j;
                        foundMarkInRow = true;
                        break;
                    }
                }

                if (foundMarkInRow) {
                    rowCovered[zero.y] = true;
                    colCovered[zero.x] = false;
                } else {
                    marked.push_back(zero);
                    break;
                }
            }
        }

        // Step 5
        while (true) {
            cv::Point zero = marked.back();

            bool foundMarkInColumn = false;
            for (int i = 0; i < marks.rows; i++) {
                if (marks.at<uint8_t>(i, zero.x) == MARKED) {
                    zero.y = i;
                    foundMarkInColumn = true;
                    break;
                }
            }

            if (foundMarkInColumn) {
                marked.push_back(zero);

                const uint8_t* marksRow = marks.ptr<uint8_t>(zero.y);
                for (int j = 0; j < marks.cols; j++) {
                    if (marksRow[j] == PRIME) {
                        zero.x = j;
                    }
                }
                marked.push_back(zero);
            } else {
                for(const cv::Point &point: marked) {
                    uint8_t &mark = marks.at<uint8_t>(point.y, point.x);
                    if (mark == MARKED) {
                        mark = UNMARKED;
                    } else {
                        mark = MARKED;
                    }
                }

                for (int i = 0; i < marks.rows; i++) {
                    uint8_t *marksRow = marks.ptr<uint8_t>(i);
                    for (int j = 0; j < marks.cols; j++) {
                        if (marksRow[j] == PRIME) {
                            marksRow[j] = UNMARKED;
                        }
                    }
                }

                break;
            }
        }
    }

    if (isTransposed) {
        cv::transpose(marks, marks);
    }

    std::map<int, int> matches;
    for (int i = 0; i < marks.rows; i++) {
        uint8_t* marksRow = marks.ptr<uint8_t>(i);
        const int32_t* distancesRow = distances.ptr<int32_t>(i);

        for (int j = 0; j < marks.cols; j++) {
            uint8_t& mark = marksRow[j];

            if (MARKED == mark && distancesRow[j] < maxDistance) {
                matches[i] = j;
                break;
            }
        }
    }

    return matches;
}
