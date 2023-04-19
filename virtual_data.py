import sys
import pickle
import cv2

import numpy as np
import xml.etree.ElementTree as ET

from os.path import join, abspath, exists
from os import listdir, makedirs
from sklearn.model_selection import train_test_split, KFold
import functools
import math


class VIRTUAL(object):
    def __init__(self, data_path='', regen_pkl=False):
        self._year = '2021'
        self._name = 'virtual-pedcross'
        self._regen_pkl = regen_pkl
        self._image_ext = '.png'

        self._virtual_path = data_path if data_path else self._get_default_path()
        assert exists(self._virtual_path), \
            'Jaad path does not exist: {}'.format(self._virtual_path)

        self._data_path = join(self._virtual_path,'data')
        self._data_split_ids_path = join(self._virtual_path, 'split_ids')


    # Path generators
    @property
    def cache_path(self):
        """
        Generate a path to save cache files
        :return: Cache file folder path
        """
        cache_path = abspath(join(self._virtual_path, 'data_cache'))
        if not exists(cache_path):
            makedirs(cache_path)
        return cache_path


    def _get_video_ids_split(self, image_set):
        """
        Returns a list of video ids for a given data split
        :param image_set: Data split, train, test, val
        :return: The list of video ids
        """
        vid_ids = []
        sets = [image_set] if image_set != 'all' else ['train', 'test', 'val']
        for s in sets:
            vid_id_file = join(self._data_split_ids_path, s + '.txt')
            with open(vid_id_file, 'rt') as fid:
                vid_ids.extend([x.strip() for x in fid.readlines()])
        return vid_ids

    def _get_video_ids(self):
        """
        Returns a list of all video ids
        :return: The list of video ids
        """
        return [vid for vid in listdir(self._data_path)]

    def _get_image_path(self, vid, fid):
        """
          Generates the image path given ids
          :param vid: Video id
          :param fid: Frame id
          :return: Return the path to the given image
          """
        return join(self._data_path, vid,'images',
                    '{}_{:06d}.png'.format(vid.split('_')[-1],fid))

    def _print_dict(self, dic):
        """
         Prints a dictionary, one key-value pair per line
         :param dic: Dictionary
         """
        for k, v in dic.items():
            print('%s: %s' % (str(k), str(v)))


    def _squarify(self, bbox, ratio, img_width):
        """
        Changes is the ratio of bounding boxes to a fixed ratio
        :param bbox: Bounding box
        :param ratio: Ratio to be changed to
        :param img_width: Image width
        :return: Squarified boduning box
        """
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])
        width_change = height * ratio - width

        bbox[0] = bbox[0] - width_change / 2
        bbox[2] = bbox[2] + width_change / 2
        if bbox[0] < 0:
            bbox[0] = 0

        # check whether the new bounding box goes beyond image boarders
        # If this is the case, the bounding box is shifted back
        if bbox[2] > img_width:
            bbox[0] = bbox[0] - bbox[2] + img_width
            bbox[2] = img_width
        return bbox

    def _get_box_scale(self,x,scale):
        if x < 0:
            return float(0)
        elif x > scale:
            return float(scale)
        else:
            return float(x)

    def _get_box_x1y1x2y2(self, box_xywh, w, h):

        x1 = math.floor(box_xywh[0] * w - box_xywh[2] * w / 2)
        y1 = math.floor(box_xywh[1] * h - box_xywh[3] * h / 2)
        x2 = math.ceil(box_xywh[0] * w + box_xywh[2] * w / 2)
        y2 = math.ceil(box_xywh[1] * h + box_xywh[3] * h / 2)


        return [self._get_box_scale(x1,w), self._get_box_scale(y1,h),
                self._get_box_scale(x2,w), self._get_box_scale(y2,h)]

    def _get_annotations(self, vid):
        """
        Generates a dictinary of annotations by parsing the video XML file
        :param vid: The id of video to parse
        :return: A dictionary of annotations
        """
        if 'not_crossing' in vid:
            crossing = 0
        else:
            crossing = 1
        label_path = join(self._data_path, vid, 'labels')

        ped_annt = 'ped_annotations'

        annotations = {}
        frame_labels = [frame for frame in listdir(label_path)]
        annotations['num_frames'] = len(frame_labels)
        annotations['width'] = 1280
        annotations['height'] = 720
        annotations[ped_annt] = {}
        ids = []

        for frame_label in frame_labels:
            frame = int(frame_label.split('.')[0].split('_')[1])
            file = open(join(label_path,frame_label), 'r')
            ff = file.readlines()
            print(join(label_path,frame_label))
            file.close()

            for i,f in enumerate(ff):

                id = vid.split('_')[-1] + '_' + str(i)
                if id not in ids:
                    annotations[ped_annt][id] = {'frames': [], 'bbox': [], 'occlusion': []}
                    annotations[ped_annt][id]['behavior'] = {'cross': []}
                    annotations[ped_annt][id]['attributes'] = {'crossing': []}
                    ids.append(id)
                    annotations[ped_annt][id]['attributes']['crossing'] = crossing

                tmp = ff[i].split(' ')

                bbox = self._get_box_x1y1x2y2([float(xx) for xx in tmp[1:5]],1280,720)
                annotations[ped_annt][id]['bbox'].append(bbox)
                annotations[ped_annt][id]['frames'].append(frame)
                annotations[ped_annt][id]['occlusion'].append(0)
                annotations[ped_annt][id]['behavior']['cross'].append(crossing)



        return annotations


    def generate_database(self):

        print('---------------------------------------------------------')
        print("Generating database for Virtual PedCross")

        # Generates a list of behavioral xml file names for  videos
        cache_file = join(self.cache_path, 'virtual_database.pkl')
        if exists(cache_file) and not self._regen_pkl:
            with open(cache_file, 'rb') as fid:
                try:
                    database = pickle.load(fid)
                except:
                    database = pickle.load(fid, encoding='bytes')
            print('virtual database loaded from {}'.format(cache_file))
            return database

        video_ids = sorted(self._get_video_ids())
        database = {}
        for vid in video_ids:
            print('Getting annotations for %s' % vid)
            vid_annotations = self._get_annotations(vid)
            # vid_attributes = self._get_ped_attributes(vid)
            # vid_appearance = self._get_ped_appearance(vid)
            # vid_veh_annotations = self._get_vehicle_attributes(vid)
            # vid_traffic_annotations = self._get_traffic_attributes(vid)

            # Combining all annotations
            # vid_annotations['vehicle_annotations'] = vid_veh_annotations
            # vid_annotations['traffic_annotations'] = vid_traffic_annotations
            # for ped in vid_annotations['ped_annotations']:
            #     try:
            #         vid_annotations['ped_annotations'][ped]['attributes'] = vid_attributes[ped]
            #     except KeyError:
            #         vid_annotations['ped_annotations'][ped]['attributes'] = {}
            #     try:
            #         vid_annotations['ped_annotations'][ped]['appearance'] = vid_appearance[ped]
            #     except KeyError:
            #         vid_annotations['ped_annotations'][ped]['appearance'] = {}

            database[vid] = vid_annotations

        with open(cache_file, 'wb') as fid:
            pickle.dump(database, fid, pickle.HIGHEST_PROTOCOL)
        print('The database is written to {}'.format(cache_file))

        return database


    def _get_data_ids(self, image_set, params):
        """
        A helper function to generate set id and ped ids (if needed) for processing
        :param image_set: Image-set to generate data
        :param params: Data generation params
        :return: Set and pedestrian ids
        """
        _pids = None

        if params['data_split_type'] == 'default':
            return self._get_video_ids_split(image_set), _pids


    def generate_data_trajectory_sequence(self, image_set, **opts):
        """
        Generates pedestrian tracks
        :param image_set: the split set to produce for. Options are train, test, val.
        :param opts:
                'fstride': Frequency of sampling from the data.
                'sample_type': Whether to use 'all' pedestrian annotations or the ones
                                    with 'beh'avior only.
                'subset': The subset of data annotations to use. Options are: 'default': Includes high resolution and
                                                                                         high visibility videos
                                                                           'high_visibility': Only videos with high
                                                                                             visibility (include low
                                                                                              resolution videos)
                                                                           'all': Uses all videos
                'height_rng': The height range of pedestrians to use.
                'squarify_ratio': The width/height ratio of bounding boxes. A value between (0,1]. 0 the original
                                        ratio is used.
                'data_split_type': How to split the data. Options: 'default', predefined sets, 'random', randomly split the data,
                                        and 'kfold', k-fold data split (NOTE: only train/test splits).
                'seq_type': Sequence type to generate. Options: 'trajectory', generates tracks, 'crossing', generates
                                  tracks up to 'crossing_point', 'intention' generates tracks similar to human experiments
                'min_track_size': Min track length allowable.
                'random_params: Parameters for random data split generation. (see _get_random_pedestrian_ids)
                'kfold_params: Parameters for kfold split generation. (see _get_kfold_pedestrian_ids)
        :return: Sequence data
        """
        params = {'fstride': 1,
                  'sample_type': 'all',  # 'beh'
                  'subset': 'default',
                  'height_rng': [0, float('inf')],
                  'squarify_ratio': 0,
                  'data_split_type': 'default',  # kfold, random, default
                  'seq_type': 'intention',
                  'min_track_size': 15,
                  'random_params': {'ratios': None,
                                    'val_data': True,
                                    'regen_data': False},
                  'kfold_params': {'num_folds': 5, 'fold': 1}}
        assert all(k in params for k in opts.keys()), "Wrong option(s)."\
        "Choose one of the following: {}".format(list(params.keys()))
        params.update(opts)

        print('---------------------------------------------------------')
        print("Generating action sequence data")
        self._print_dict(params)

        annot_database = self.generate_database()

        if params['seq_type'] == 'crossing':
            sequence = self._get_crossing(image_set, annot_database, **params)


        return sequence


    def _get_center(self, box):
        """
        Calculates the center coordinate of a bounding box
        :param box: Bounding box coordinates
        :return: The center coordinate
        """
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def _get_continuous(self, frames):
        for i in range(len(frames)-1):
            if not frames[i]+1 in frames:
                return False
        return True


    def _get_crossing(self, image_set, annotations, **params):
        """
        Generates crossing data.
        :param image_set: Data split to use
        :param annotations: Annotations database
        :param params: Parameters to generate data (see generade_database)
        :return: A dictionary of trajectories
        """

        print('---------------------------------------------------------')
        print("Generating crossing data")

        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']
        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq = []
        vehicle_seq = []
        activities = []


        video_ids, _pids = self._get_data_ids(image_set, params)

        for vid in sorted(video_ids):
            img_width = annotations[vid]['width']
            img_height = annotations[vid]['height']
            pid_annots = annotations[vid]['ped_annotations']

            for pid in sorted(pid_annots):

                num_pedestrians += 1

                frame_ids = pid_annots[pid]['frames']

                event_frame = -1
                end_idx = -3

                boxes = pid_annots[pid]['bbox'][:end_idx + 1]
                frame_ids = frame_ids[: end_idx + 1]
                images = [self._get_image_path(vid, f) for f in frame_ids]
                occlusions = pid_annots[pid]['occlusion'][:end_idx + 1]

                if height_rng[0] > 0 or height_rng[1] < float('inf'):
                    images, boxes, frame_ids, occlusions = self._height_check(height_rng,
                                                                              frame_ids, boxes,
                                                                              images, occlusions)


                if not self._get_continuous(frame_ids):
                    continue


                if len(boxes) / seq_stride < params['min_track_size']:
                    continue

                if sq_ratio:
                    boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                image_seq.append(images[::seq_stride])
                box_seq.append(boxes[::seq_stride])
                center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
                occ_seq.append(occlusions[::seq_stride])

                ped_ids = [[pid]] * len(boxes)
                pids_seq.append(ped_ids[::seq_stride])

                intent = [[1]] * len(boxes)
                vehicle = [[0]] * len(boxes) # none vehicle
                acts = [[int(pid_annots[pid]['attributes']['crossing'] > 0)]] * len(boxes)

                intent_seq.append(intent[::seq_stride])
                activities.append(acts[::seq_stride])
                vehicle_seq.append(vehicle[::seq_stride])


        print('Split: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {'image': image_seq,
                'pid': pids_seq,
                'bbox': box_seq,
                'center': center_seq,
                'occlusion': occ_seq,
                'vehicle_act': vehicle_seq,
                'intent': intent_seq,
                'activities': activities,
                'image_dimension': (img_width, img_height)}