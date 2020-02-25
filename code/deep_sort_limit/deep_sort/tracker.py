# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

from . import kalman_filter

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=100, n_init=50, max_tracks=10, metric_param=0):
        self.metric = metric
        self.metric_param = metric_param
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.max_tracks = max_tracks
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        gate = len([t for t in self.tracks if t.is_confirmed()]) < self.max_tracks#True until all tracks are created
        #print([t.time_since_update for t in self.tracks if t.is_confirmed()])
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections, gate)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        # Update track set.
        #print(gate, len(detections), unmatched_detections)
        if gate:
            for detection_idx in unmatched_detections:
                self._initiate_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections, gate=True):
        #print("MATCH")
        def gated_metric(tracks, dets, track_indices, detection_indices, metric_param=0):
            features = np.array([dets[i].feature for i in detection_indices])

            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices, gate=gate, metric_param=metric_param)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        #LA MODIFICATION PRINCIPALE EST ICI
        # AU LIEU d'appliquer le gate sur la matrices des distances, on autorise deepsort à associer les détections restantes uniquement sur la base du filtre de Kalman


        """
        if gate:
            threshold = self.metric.matching_threshold
        else:
            threshold = 1e+5 - 1"""
        # Associate confirmed tracks using appearance features.


        threshold = self.metric.matching_threshold

        """matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                                        gated_metric, threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)"""

        # AUTRE MODIFICATION, on ne fait plus la "matching cascade" mais directement le matching

        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.min_cost_matching(
                gated_metric, threshold, self.tracks, detections,
                confirmed_tracks)


        """

               on laisse le gating threshold sur la cosine distance mais on refait un tour avec un plus grand metric_param (importance de la distance de Mahalanobis)
               pour recoller les morceaux à partir du filtre de kalman quand une détection semble trop différente visuelement de tous les tracklets
        Cela l'empêche de créer des nouvelles identités à cause de trop grosses variations d'apparences, et permet de conserver l'identité même lorsque le joueur se retourne (par exemple)
        

               """


        gate_kalman = kalman_filter.chi2inv95[4]

        matches_c, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.min_cost_matching(
                lambda a,b,c,d: gated_metric(a,b,c,d,self.metric_param), self.metric_param*gate_kalman + (1-self.metric_param)*threshold, self.tracks, detections,
                unmatched_tracks_a, unmatched_detections)


        """if len(matches_c) > 0:
            print("raccord !")"""

        matches_a = matches_a + matches_c


        #print(gate, unmatched_detections)

        # Associate remaining tracks together with unconfirmed tracks using IOU
        if gate:
            iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
            unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
            matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        else:
            matches = matches_a
            unmatched_tracks = unmatched_tracks_a
        #print("----END MATCH")
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        #print("Nouvel id: ", self._next_id)
        self._next_id += 1
