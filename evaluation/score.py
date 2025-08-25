"""
File containing auxiliar score functions
"""

#Standard imports
from SoccerNet.Evaluation.ActionSpotting import average_mAP
import numpy as np


def compute_amAP(targets_numpy, detections_numpy, closests_numpy, framerate=25, metric = 'tight', event_team = False):

    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":
        deltas = np.array([1])  # np.arange(1)*1 + 1
    elif metric == "at2":
        deltas = np.array([2])
    elif metric == "at3":
        deltas = np.array([3])
    elif metric == "at4":
        deltas = np.array([4])
    elif metric == "at5":
        deltas = np.array([5])

    if event_team:
        ntargets = np.zeros(targets_numpy[0].shape[1])
        for i in range(len(targets_numpy)):
            ntargets += targets_numpy[i].sum(axis=0)

    mAP, mAP_per_class, mAP_visible, mAP_per_class_visible, mAP_unshown, mAP_per_class_unshown = average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate=framerate, deltas = deltas)

    if event_team:
        mAP_per_class = mAP_per_class * ntargets
        mAP_per_class = [(mAP_per_class[i*2] + mAP_per_class[(i*2)+1]) / (ntargets[i*2] + ntargets[i*2+1]) for i in range(len(mAP_per_class) // 2)]
        mAP = np.mean(mAP_per_class)

        mAP_per_class_visible = mAP_per_class_visible * ntargets
        mAP_per_class_visible = [(mAP_per_class_visible[i*2] + mAP_per_class_visible[(i*2)+1]) / (ntargets[i*2] + ntargets[i*2+1]) for i in range(len(mAP_per_class_visible) // 2)]
        mAP_visible = np.mean(mAP_per_class_visible)

        mAP_per_class_unshown = mAP_per_class_unshown * ntargets
        mAP_per_class_unshown = [(mAP_per_class_unshown[i*2] + mAP_per_class_unshown[(i*2)+1]) / (ntargets[i*2] + ntargets[i*2+1]) for i in range(len(mAP_per_class_unshown) // 2)]
        mAP_unshown = np.mean(mAP_per_class_unshown)

    return {"mAP": mAP, "mAP_per_class": mAP_per_class, "mAP_visible": mAP_visible, "mAP_per_class_visible": mAP_per_class_visible, 
            "mAP_unshown": mAP_unshown, "mAP_per_class_unshown": mAP_per_class_unshown}