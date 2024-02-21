The ground truth used to calculate the accuracy of different metrics in the paper are the minimum and maximum lateral angles in each data file. Each metric is represented as follows:

The MRDE accuracy is calculated using the "PoI_GT_minLatAngle" and "PoI_GT_maxLatAngle".

The MinDT accuracy is calculated using the "PoI_GT_adjustedMinLatAngle" and "PoI_GT_adjustedMaxLatAngle".

The SegObj accuracy is calculated using the "PoI_GT_visibleMinLatAngle" and "PoI_GT_visibleMaxLatAngle".

Please refer to the main code to see how the accuracy is calculated for each metric and refer to the paper for metric explanation.
