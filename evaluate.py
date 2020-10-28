from collections import defaultdict
import numpy as np

from common import viou


def eval_proposal_scores(gt_relations, pred_relations, viou_threshold):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((len(pred_relations))) * -np.inf
    for pred_idx, pred_relation in enumerate(pred_relations):
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_relation in enumerate(gt_relations):
            if not gt_detected[gt_idx]\
                    and tuple(pred_relation['triplet']) == tuple(gt_relation['triplet']):
                s_iou = viou(pred_relation['sub_traj'], pred_relation['duration'],
                        gt_relation['sub_traj'], gt_relation['duration'])
                o_iou = viou(pred_relation['obj_traj'], pred_relation['duration'],
                        gt_relation['obj_traj'], gt_relation['duration'])
                ov = min(s_iou, o_iou)
                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
        if k_max >= 0:
            hit_scores[pred_idx] = pred_relation['score']
            gt_detected[k_max] = True
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def evaluate(groundtruth, prediction, viou_threshold=0.5,
        det_nreturns=[25, 50]):
    """ evaluate visual relation detection and visual 
    relation tagging.
    """
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    tot_gt_relations = 0
    print('Computing Spatial-Temporal Recall over {} videos...'.format(len(groundtruth)))
    for vid, gt_relations in groundtruth.items():
        if len(gt_relations)==0:
            continue
        tot_gt_relations += len(gt_relations)
        predict_relations = prediction.get(vid, [])
        # compute spatial-temporal recalls for interactivity proposals
        det_prec, det_rec, det_scores = eval_proposal_scores(
                gt_relations, predict_relations, viou_threshold)
        tp = np.isfinite(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
    # calculate recall for interactivity proposal
    rec_at_n = dict()
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        rec_at_n[nre] = rec[-1]
    print('Spatial-Temporal Recall@25: {}'.format(rec_at_n[25]))
    print('Spatial-Temporal Recall@50: {}'.format(rec_at_n[50]))
    return rec_at_n


if __name__ == "__main__":
    import json
    from argparse import ArgumentParser
    import pickle

    groundtruth = 'kiev_val_gt.json'
    prediction = 'kiev_val_pred.json'

    print('Loading ground truth from {}'.format(groundtruth))
    with open(groundtruth, 'r') as fp:
        gt = json.load(fp)
    print('Number of videos in ground truth: {}'.format(len(gt)))

    print('Loading prediction from {}'.format(prediction))
    with open(prediction, 'r') as fp:
        pred = json.load(fp)
    kiev_val_videos = pickle.load(open('kiev_val_video_list.pkl', 'rb'))
    print('Number of videos in prediction: {}'.format(len(pred)))  # one validation video has no output
    for k in kiev_val_videos:
        if k not in pred:
            print('Video {} has no prediction.'.format(k))

    rec_at_n = evaluate(gt, pred)
