# vim: expandtab:ts=4:sw=4
import argparse
import numpy as np
import motmetrics as mm
import pandas as pd

def run(sequence_dir, result_file, gt_file=None, offset=0, verbose=0):
    """Run tracking result visualization.

    Parameters
    ----------
    sequence_dir : str
        Path to the  sequence directory.
    result_file : str
        Path to the tracking output file in MOTChallenge ground truth format.
    gt_file : Optional[str]
        Path to the ground truth file.
    offset: Optional[int]
        Offset in number of frames
    """
    groundtruth = np.load(gt_file)
    hypotheses = np.load(result_file)


    print(result_file)
    # List all default metrics
    mh = mm.metrics.create()


    #print(mh.list_metrics_markdown())

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    for frame in range(groundtruth[:, 0].astype(np.int).max()):
        gt = groundtruth[groundtruth[:, 0].astype(np.int) == frame]

        #gt = gt[gt[:,1]==2.0]

        hyp = hypotheses[hypotheses[:, 0].astype(np.int) == frame]

        #hyp = hyp[(hyp[:, 1] == 5.0) | (hyp[:, 1] == 7.0)]



        #hyp = gt

        # Call update once for per frame. For now, assume distances between
        # frame objects / hypotheses are given.
        if(gt.shape[0]>0):
            distances = mm.distances.iou_matrix(gt[:,2:6], hyp[:,2:6], max_iou=0.5)
            acc.update(
                gt[:,1],  # Ground truth objects in this frame
                hyp[:, 1],  # Detector hypotheses in this frame
                distances
            )
    summary = mh.compute(acc, metrics=['num_predictions', 'num_matches', 'num_objects', 'num_switches','num_ascend','num_transfer','num_migrate', 'recall',
                                       'precision', 'idp', 'idr', 'idf1',
                                    'mota', 'motp'], name='acc')

    """
    
    To tune deepsort we want to reduce num_transfer
    
    To tune the post clustering, we want to reduce num_switch while keeping num_transfer low
        
    """
    MATCH = (acc.mot_events['Type'] == 'MATCH')
    A = (acc.mot_events['Type'] == 'ASCEND')
    S = (acc.mot_events['Type'] == 'SWITCH')
    T = (acc.mot_events['Type'] == 'TRANSFER')
    M = (acc.mot_events['Type'] == 'MIGRATE')
    total_n_annotations = 0
    purity=[]


    for i in np.unique(hypotheses[:, 1]):
        n_targets = np.unique(acc.mot_events[MATCH|T][acc.mot_events['HId'] == i]['OId'].values).shape[0]

        n_matches = acc.mot_events[MATCH|T][acc.mot_events['HId'] == i]['OId'].value_counts()

        if n_targets > 0:

            total_n_annotations += n_targets

            purity.append(n_matches.max()/n_matches.sum()*100)
            h = n_matches/n_matches.sum()*100
            if verbose>0:
                print(
                    "Hypothesis ID: {}, Number of ground truth targets: {}, Purity: {} %".format(i, n_targets, purity[-1]))
                print("Ground truth / Volume:")
                print(h)
                print("---------------------------------------------------------------------------------------------")
        else:
            if verbose>0:
                print("Hypothesis ID: {} => no match with ground truth".format(i))

    summary.insert(0, "Purity", np.mean(purity))
    summary.insert(0, "test", total_n_annotations)
    summary.insert(0, "N_a", total_n_annotations / np.unique(hypotheses[:, 1]).shape[0])
    summary.insert(0, "Number of IDs", [int(np.unique(hypotheses[:, 1]).shape[0])])


    summary.to_csv(result_file.replace(".npy","_score.csv"), index=False)
    print("Metrics summary stored in {}".format(result_file.replace(".npy","_score.csv")))


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Siamese Tracking")
    parser.add_argument(
        "--sequence_dir", help="Path to the sequence directory.",
        default=None, required=True)
    parser.add_argument(
        "--result_file", help="Tracking output in MOTChallenge file format.",
        default=None, required=True)
    parser.add_argument(
        "--offset", help="Frame offset. Default to 0", default=0)
    parser.add_argument(
        "--verbose", help="Verbose level.", default=0)
    parser.add_argument(
        "--gt_file", help="Path to ground truth file",
        default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.result_file,
        args.gt_file, int(args.offset), int(args.verbose))

"""



Recall = (num matches + num switches) / num_objects

"""


"""Name|Description
:---|:---
num_frames|Total number of frames.
obj_frequencies|Total number of occurrences of individual objects over all frames.
pred_frequencies|Total number of occurrences of individual predictions over all frames.
num_matches|Total number matches.
num_switches|Total number of track switches.
num_transfer|Total number of track transfer.
num_ascend|Total number of track ascend.
num_migrate|Total number of track migrate.
num_false_positives|Total number of false positives (false-alarms).
num_misses|Total number of misses.
num_detections|Total number of detected objects including matches and switches.
num_objects|Total number of unique object appearances over all frames.
num_predictions|Total number of unique prediction appearances over all frames.
num_unique_objects|Total number of unique object ids encountered.
track_ratios|Ratio of assigned to total appearance count per unique object id.
mostly_tracked|Number of objects tracked for at least 80 percent of lifespan.
partially_tracked|Number of objects tracked between 20 and 80 percent of lifespan.
mostly_lost|Number of objects tracked less than 20 percent of lifespan.
num_fragmentations|Total number of switches from tracked to not tracked.
motp|Multiple object tracker precision.
mota|Multiple object tracker accuracy.
precision|Number of detected objects over sum of detected and false positives.
recall|Number of detections over number of objects.
id_global_assignment|ID measures: Global min-cost assignment for ID measures.
idfp|ID measures: Number of false positive matches after global min-cost matching.
idfn|ID measures: Number of false negatives matches after global min-cost matching.
idtp|ID measures: Number of true positives matches after global min-cost matching.
idp|ID measures: global min-cost precision.
idr|ID measures: global min-cost recall.
idf1|ID measures: global min-cost F1 score.
"""