"""
File containing main evaluation functions
"""

#Standard imports
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import copy
import os
import json
import zipfile

#Local imports
# from util.score import compute_mAPs
from util.io import store_json_snb, load_text
from util.score import compute_amAP
from util.utils import LoadJsonFromZip, getListGames, EVENT_DICTIONARY_BALL

#Constants
TOLERANCES_SN = [3, 6]
WINDOWS_SN = [3, 6]
TOLERANCES_SNB = [6, 12]
WINDOWS_SNB = [6, 12]
INFERENCE_BATCH_SIZE = 4
FPS_SN = 25

GAMES_SNB = {
        'train': ["england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich",
            "england_efl/2019-2020/2019-10-01 - Hull City - Sheffield Wednesday",
            "england_efl/2019-2020/2019-10-01 - Brentford - Bristol City",
            "england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest"],
        'val' : ["england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End"],
        'test': ["england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town",
            "england_efl/2019-2020/2019-10-01 - Reading - Fulham"],
        'challenge': ["england_efl/2019-2020/2019-10-02 - Cardiff City - Queens Park Rangers",
            "england_efl/2019-2020/2019-10-01 - Wigan Athletic - Birmingham City"]
        }

def process_frame_predictions(dataset, classes, pred_dict, threshold=0.01):
    
    classes_inv = {v: k for k, v in classes.items()}

    fps_dict = {}
    for video, _, fps in dataset.videos:
        fps_dict[video] = fps

    pred_events = []
    for video, (scores, support) in (sorted(pred_dict.items())):
        if np.min(support) == 0:
            support[support == 0] = 1
        assert np.min(support) > 0, (video, support.tolist())
        scores /= support[:, None]
        
        events = []
        for i in range(scores.shape[0]):
            for j in classes_inv:
                if scores[i, j] >= threshold:
                    label = classes_inv[j]
                    if '-' in label:
                        team = label.split('-')[1]
                        label = label.split('-')[0]
                        events.append({
                            'label': label,
                            'team': team,
                            'frame': i,
                            'score': scores[i, j].item()
                        })
                    else:
                        events.append({
                            'label': classes_inv[j],
                            'frame': i,
                            'score': scores[i, j].item()
                        })
        pred_events.append({
            'video': video, 'events': events,
            'fps': fps_dict[video]})
        
    return pred_events


def mAPevaluate(model, dataset, classes, printed=True, event_team = False, metric = 'at1'):
    
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))
        
    batch_size = INFERENCE_BATCH_SIZE

    for clip in tqdm(DataLoader(
        dataset, num_workers=4*2, pin_memory=True,
        batch_size=batch_size
    )):
        
        _, batch_pred_scores = model.predict(clip['frame'])

        for i in range(clip['frame'].shape[0]):
            video = clip['video'][i]
            scores, support = pred_dict[video]
            pred_scores = batch_pred_scores[i]
            start = clip['start'][i].item()
            if start < 0:
                pred_scores = pred_scores[-start:, :]
                start = 0
            end = start + pred_scores.shape[0]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:end - start, :]

            scores[start:end, :] += pred_scores
            support[start:end] += (pred_scores.sum(axis=1) != 0) * 1

    
    detections_numpy = list()
    targets_numpy = list()
    closests_numpy = list()
    for (game, value) in pred_dict.items():
        scores = value[0]
        scores[scores == 0] = -1
        detections_numpy.append(scores[:, 1:]) # Remove background class
        labels = np.zeros((scores.shape[0], len(classes)))
        label = dataset.get_labels(game)
        label_idx = label.nonzero()[0]
        
        for idx in label_idx:
            labels[idx, label[idx]-1] = 1 # Remove background class

        targets_numpy.append(labels)

        closest_numpy = np.zeros(labels.shape) - 1
        # Get the closest action index
        for c in np.arange(labels.shape[-1]):
            indexes = np.where(labels[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = labels[indexes[i], c]
        closests_numpy.append(closest_numpy)

    results = compute_amAP(targets_numpy, detections_numpy, closests_numpy, framerate=FPS_SN/dataset._stride, metric = metric, event_team = event_team)
    if printed:
        print_results(results, classes, metric, event_team = event_team)
    
    return results['mAP']

def mAPevaluateCodabench(SoccerNet_path, Predictions_path, prediction_file="results_spotting.json", split = "test", printed=False, event_team=True, metric="at1"):

    #Compute metric
    detections_numpy = list()
    targets_numpy = list()
    closests_numpy = list()

    list_games = getListGames(split=split)

    #Update classes to start at 1 for label2vector and pred2vector
    classes = EVENT_DICTIONARY_BALL
    if event_team:
        classes2 = {}
        for k, v in classes.items():
            classes2[k + '-left'] = v*2 +1
            classes2[k + '-right'] = v*2 +2
        classes = classes2
    else:
        classes2 = {}
        for k, v in classes.items():
            classes2[k] = v*2 +1
        classes = classes2

    #We reload predictions & labels for consistency in the framerate
    for game in (list_games):
        if zipfile.is_zipfile(SoccerNet_path):
            labels = LoadJsonFromZip(SoccerNet_path, os.path.join(game, 'Labels-ball.json'))
        else:
            labels = json.load(open(os.path.join(SoccerNet_path, game, 'Labels-ball.json')))
        num_classes = len(classes)
        # convert labels to vector
        labels = label2vector(labels, num_classes=num_classes, EVENT_DICTIONARY=classes, framerate=FPS_SN, event_team = event_team)
        
        if zipfile.is_zipfile(Predictions_path):
            predictions = LoadJsonFromZip(Predictions_path, os.path.join(game, prediction_file))
        else:
            predictions = json.load(open(os.path.join(Predictions_path, game, prediction_file)))
        predictions = predictions2vector(predictions, num_classes=num_classes, EVENT_DICTIONARY=classes, framerate=FPS_SN, event_team = event_team)

        targets_numpy.append(labels)
        detections_numpy.append(predictions)

        closest_numpy = np.zeros(labels.shape) - 1
        # Get the closest action index
        for c in np.arange(labels.shape[-1]):
            indexes = np.where(labels[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = labels[indexes[i], c]
        closests_numpy.append(closest_numpy)

    results = compute_amAP(targets_numpy, detections_numpy, closests_numpy, framerate=FPS_SN, metric = metric, event_team = event_team)
    

    if event_team:
        # Additional results without considering the team
        detections_numpy = list()
        targets_numpy = list()
        closests_numpy = list()

        aux_classes = {k.split('-')[0]: (v//2) for k, v in classes.items() if v % 2 == 0}

        for game in (list_games):
            if zipfile.is_zipfile(SoccerNet_path):
                labels = LoadJsonFromZip(SoccerNet_path, os.path.join(game, 'Labels-ball.json'))
            else:
                labels = json.load(open(os.path.join(SoccerNet_path, game, 'Labels-ball.json')))
            num_classes = len(aux_classes)
            # convert labels to vector
            labels = label2vector(labels, num_classes=num_classes, EVENT_DICTIONARY=aux_classes, framerate=FPS_SN, event_team = False)

            if zipfile.is_zipfile(Predictions_path):
                predictions = LoadJsonFromZip(Predictions_path, os.path.join(game, prediction_file))
            else:
                predictions = json.load(open(os.path.join(Predictions_path, game, prediction_file)))
            predictions = predictions2vector(predictions, num_classes=num_classes, EVENT_DICTIONARY=aux_classes, framerate=FPS_SN, event_team = False)

            targets_numpy.append(labels)
            detections_numpy.append(predictions)

            closest_numpy = np.zeros(labels.shape) - 1
            # Get the closest action index
            for c in np.arange(labels.shape[-1]):
                indexes = np.where(labels[:, c] != 0)[0].tolist()
                if len(indexes) == 0:
                    continue
                indexes.insert(0, -indexes[0])
                indexes.append(2 * closest_numpy.shape[0])
                for i in np.arange(len(indexes) - 2) + 1:
                    start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                    stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                    closest_numpy[start:stop, c] = labels[indexes[i], c]
            closests_numpy.append(closest_numpy)

        results2 = compute_amAP(targets_numpy, detections_numpy, closests_numpy, framerate=FPS_SN, metric = metric, event_team = False)
        
        results['mAP_no_team'] = results2['mAP']
        results['mAP_per_class_no_team'] = results2['mAP_per_class']
        results['mAP_visible_no_team'] = results2['mAP_visible']

    if printed:
        print_results(results, classes, metric, event_team = event_team)

    return results

def print_results(results, classes, metric, event_team = False):
    classes_inv = {v: k for k, v in classes.items()}
    print('--------------------------------------------------')
    print('mAP results for metric:', metric)
    print('--------------------------------------------------')
    print('mAP - {:0.2f}'.format(results['mAP'] * 100))
    print('mAP per class:')
    if not event_team:
        for i in range(len(classes)):
            print('{} - {:0.2f}'.format(classes_inv[i+1], results['mAP_per_class'][i] * 100))
    else:
        for i in range(len(classes) // 2):
            print('{} - {:0.2f}'.format(classes_inv[i*2+1].split('-')[0], results['mAP_per_class'][i] * 100))
    print('--------------------------------------------------')
    if 'mAP_no_team' in results.keys():
        print('mAP without considering the team - {:0.2f}'.format(results['mAP_no_team'] * 100))
        print('mAP per class without considering the team:')
        for i in range(len(classes) // 2):
            print('{} - {:0.2f}'.format(classes_inv[i*2+1].split('-')[0], results['mAP_per_class_no_team'][i] * 100))
        print('--------------------------------------------------')
    return


def mAPevaluateTest(model, split, dataset, classes, printed=True, event_team = False, metric = 'at1', pred_file = None, postprocessing = 'SNMS'):

    if dataset._dataset == 'soccernet':
        windows = WINDOWS_SN
    elif dataset._dataset == 'soccernetball':
        windows = WINDOWS_SNB
    
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))
        
    batch_size = INFERENCE_BATCH_SIZE

    for clip in tqdm(DataLoader(
        dataset, num_workers=4*2, pin_memory=True,
        batch_size=batch_size
    )):
        
        _, batch_pred_scores = model.predict(clip['frame'])

        for i in range(clip['frame'].shape[0]):
            video = clip['video'][i]
            scores, support = pred_dict[video]
            pred_scores = batch_pred_scores[i]
            start = clip['start'][i].item()
            if start < 0:
                pred_scores = pred_scores[-start:, :]
                start = 0
            end = start + pred_scores.shape[0]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:end - start, :]

            scores[start:end, :] += pred_scores
            support[start:end] += (pred_scores.sum(axis=1) != 0) * 1

    pred_events = process_frame_predictions(dataset, classes, pred_dict, threshold = 0.01)

    if postprocessing == 'NMS':
        pred_events = non_maximum_supression(pred_events, window = windows[0], threshold=0.01)
    elif postprocessing == 'SNMS':
        pred_events = soft_non_maximum_supression(pred_events, window = windows[1], threshold=0.01)

    #Store predictions
    store_json_snb(pred_file, pred_events, stride = dataset._stride)
    
    if split == 'challenge':
        return None

    #Compute metric
    detections_numpy = list()
    targets_numpy = list()
    closests_numpy = list()

    #Get labels path
    if dataset._dataset == 'soccernet':
        labels_path = load_text(os.path.join('data', 'soccernet', 'labels_path.txt'))[0]
        label_file = 'Labels-v2.json'
    elif dataset._dataset == 'soccernetball':
        labels_path = load_text(os.path.join('data', 'soccernetball', 'labels_path.txt'))[0]
        label_file = 'Labels-ball.json'

    #We reload predictions & labels for consistency in the framerate
    for game in tqdm(GAMES_SNB[split]):
        labels = json.load(open(os.path.join(labels_path, game, label_file)))
        num_classes = len(classes)
        # convert labels to vector
        labels = label2vector(labels, num_classes=num_classes, EVENT_DICTIONARY=classes, framerate=FPS_SN, event_team = event_team)
        
        predictions = json.load(open(os.path.join(pred_file, game, 'results_spotting.json')))
        predictions = predictions2vector(predictions, num_classes=num_classes, EVENT_DICTIONARY=classes, framerate=FPS_SN, event_team = event_team)

        targets_numpy.append(labels)
        detections_numpy.append(predictions)

        closest_numpy = np.zeros(labels.shape) - 1
        # Get the closest action index
        for c in np.arange(labels.shape[-1]):
            indexes = np.where(labels[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = labels[indexes[i], c]
        closests_numpy.append(closest_numpy)

    results = compute_amAP(targets_numpy, detections_numpy, closests_numpy, framerate=FPS_SN, metric = metric, event_team = event_team)
    

    if event_team:
        # Additional results without considering the team
        detections_numpy = list()
        targets_numpy = list()
        closests_numpy = list()

        aux_classes = {k.split('-')[0]: (v//2) for k, v in classes.items() if v % 2 == 0}

        for game in tqdm(GAMES_SNB[split]):
            labels = json.load(open(os.path.join(labels_path, game, label_file)))
            num_classes = len(aux_classes)
            # convert labels to vector
            labels = label2vector(labels, num_classes=num_classes, EVENT_DICTIONARY=aux_classes, framerate=FPS_SN, event_team = False)

            predictions = json.load(open(os.path.join(pred_file, game, 'results_spotting.json')))
            predictions = predictions2vector(predictions, num_classes=num_classes, EVENT_DICTIONARY=aux_classes, framerate=FPS_SN, event_team = False)

            targets_numpy.append(labels)
            detections_numpy.append(predictions)

            closest_numpy = np.zeros(labels.shape) - 1
            # Get the closest action index
            for c in np.arange(labels.shape[-1]):
                indexes = np.where(labels[:, c] != 0)[0].tolist()
                if len(indexes) == 0:
                    continue
                indexes.insert(0, -indexes[0])
                indexes.append(2 * closest_numpy.shape[0])
                for i in np.arange(len(indexes) - 2) + 1:
                    start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                    stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                    closest_numpy[start:stop, c] = labels[indexes[i], c]
            closests_numpy.append(closest_numpy)

        results2 = compute_amAP(targets_numpy, detections_numpy, closests_numpy, framerate=FPS_SN, metric = metric, event_team = False)
        
        results['mAP_no_team'] = results2['mAP']
        results['mAP_per_class_no_team'] = results2['mAP_per_class']
        results['mAP_visible_no_team'] = results2['mAP_visible']

    if printed:
        print_results(results, classes, metric, event_team = event_team)

    return results

def non_maximum_supression(pred, window, threshold = 0.0):
    preds = copy.deepcopy(pred)
    new_pred = []
    for video_pred in preds:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        i = 0
        for v in events_by_label.values():
            if type(window) is not list:
                class_window = window
            else:
                class_window = window[i]
                i += 1
            while(len(v) > 0):
                e1 = max(v, key=lambda x:x['score'])
                if e1['score'] < threshold:
                    break
                pos1 = [pos for pos, e in enumerate(v) if e['frame'] == e1['frame']][0]
                events.append(copy.deepcopy(e1))
                v.pop(pos1)
                list_pos = [pos for pos, e in enumerate(v) if ((e['frame'] >= e1['frame']-class_window) & (e['frame'] <= e1['frame']+class_window))]
                for pos in list_pos[::-1]: #reverse order to avoid movement of positions in the list
                    v.pop(pos)

        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred

def soft_non_maximum_supression(pred, window, threshold = 0.01):
    preds = copy.deepcopy(pred)
    new_pred = []
    for video_pred in preds:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        i = 0
        for v in events_by_label.values():
            if type(window) is not list:
                class_window = window
            else:
                class_window = window[i]
                i += 1
            while(len(v) > 0):
                e1 = max(v, key=lambda x:x['score'])
                if e1['score'] < threshold:
                    break
                pos1 = [pos for pos, e in enumerate(v) if e['frame'] == e1['frame']][0]
                events.append(copy.deepcopy(e1))
                list_pos = [pos for pos, e in enumerate(v) if ((e['frame'] >= e1['frame']-class_window) & (e['frame'] <= e1['frame']+class_window))]
                for pos in list_pos:
                    v[pos]['score'] = v[pos]['score'] * (np.abs(e1['frame'] - v[pos]['frame'])) ** 2 / ((class_window+0) ** 2)
                v.pop(pos1)

        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred

def label2vector(labels, num_classes=17, framerate=2, EVENT_DICTIONARY={}, event_team = False):

    vector_size = 120*60*framerate

    label_half1 = np.zeros((vector_size, num_classes))

    for annotation in labels["annotations"]:

        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        # annotation at millisecond precision
        if "position" in annotation:
            frame = int(framerate * ( int(annotation["position"])/1000 ))
        # annotation at second precision
        else:
            frame = framerate * ( seconds + 60 * minutes )

        if not event_team:
            label = EVENT_DICTIONARY[event]-1
        else:
            event = event + '-' + annotation['team']
            label = EVENT_DICTIONARY[event]-1
        # print(event, label, half)

        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        if half == 1:
            frame = min(frame, vector_size-1)
            label_half1[frame][label] = value

    return label_half1

def predictions2vector(predictions, num_classes=17, framerate=2, EVENT_DICTIONARY={}, event_team = False):


    vector_size = 120*60*framerate

    prediction_half1 = np.zeros((vector_size, num_classes))-1

    for annotation in predictions["predictions"]:

        time = int(annotation["position"])
        event = annotation["label"]

        frame = int(framerate * ( time/1000 ))

        if not event_team:
            label = EVENT_DICTIONARY[event]-1
        else:
            event = event + '-' + annotation['team']
            label = EVENT_DICTIONARY[event]-1

        value = annotation["confidence"]

        frame = min(frame, vector_size-1)
        prediction_half1[frame][label] = value

    return prediction_half1