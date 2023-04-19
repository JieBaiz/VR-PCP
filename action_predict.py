from typing import Any
import h5py
import time
import yaml
import wget
import cv2
import tensorflow as tf
import numpy as np
from utils import *
from base_models import AlexNet, C3DNet, convert_to_fcn
from base_models import I3DNet
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.layers import GRU, LSTM, GRUCell
from tensorflow.keras.layers import Dropout, LSTMCell, RNN
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Average, Add
from tensorflow.keras.layers import ConvLSTM2D, Conv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import vgg16, resnet50
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, dot, concatenate, Activation,GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import kld,binary_crossentropy,MSLE
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization,MaxPool2D,ReLU,GlobalAvgPool2D,DepthwiseConv2D,MaxPooling2D,AveragePooling2D,add
#from tensorflow.keras.layers.embeddings import Embedding
from tensorflow.keras.utils import Sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from PIL import Image
# from keras_bert import get_custom_objects
from scipy.stats import wasserstein_distance
# TODO: Make all global class parameters to minimum , e.g. no model generation
class ActionPredict(object):
    """
        A base interface class for creating prediction models
    """

    def __init__(self,
                 global_pooling='avg',
                 regularizer_val=0.0001,
                 backbone='c3d',
                 **kwargs):
        """
        Class init function
        Args:
            global_pooling: Pooling method for generating convolutional features
            regularizer_val: Regularization value for training
            backbone: Backbone for generating convolutional features
        """
        # Network parameters
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)
        self._global_pooling = global_pooling
        self._backbone = backbone
        self._generator = None # use data generator for train/test 

    # Processing images anf generate features
    def load_images_crop_and_process(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     process=True,
                                     regen_data=False):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'none' (no cropping)
                        'bbox' (crop using bounding box coordinates),
                        'context' (A region containing pedestrian and their local surround)
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            crop_mode: How to resize ond/or pad the corpped images (see utils.img_pad)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
            target_dim: Dimension of final visual features
            process: Whether process the raw images using a neural network
            regen_data: Whether regenerate visual features. This will overwrite the cached features
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """

        # load the feature files if exists
        print("Generating {} features crop_type={} crop_mode={}\
              \nsave_path={}, ".format(data_type, crop_type, crop_mode,
                                       save_path))
        preprocess_dict = {'vgg16': vgg16.preprocess_input, 'resnet50': resnet50.preprocess_input}
        backbone_dict = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50}

        preprocess_input = preprocess_dict.get(self._backbone, None)
        if process:
            assert (self._backbone in ['vgg16', 'resnet50']), "{} is not supported".format(self._backbone)

        convnet = backbone_dict[self._backbone](input_shape=(224, 224, 3),
                                                include_top=False, weights='imagenet') if process else None
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False
                set_id = imp.split('\\')[-3]
                vid_id = imp.split('\\')[-2]
                img_name = imp.split('\\')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)

                # Modify the path depending on crop mode
                if crop_type == 'none':
                    img_save_path = os.path.join(img_save_folder, img_name + '.pkl')
                else:
                    img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')

                # Check whether the file exists
                if os.path.exists(img_save_path) and not regen_data:
                    if not self._generator:
                        with open(img_save_path, 'rb') as fid:
                            try:
                                img_features = pickle.load(fid)
                            except:
                                img_features = pickle.load(fid, encoding='bytes')
                else:
                    if 'flip' in imp:
                        imp = imp.replace('_flip', '')
                        flip_image = True
                    if crop_type == 'none':
                        img_data = cv2.imread(imp)
                        img_features = cv2.resize(img_data, target_dim)
                        if flip_image:
                            img_features = cv2.flip(img_features, 1)
                    else:
                        img_data = cv2.imread(imp)
                        if flip_image:
                            img_data = cv2.flip(img_data, 1)
                        if crop_type == 'bbox':
                            b = list(map(int, b[0:4]))
                            cropped_image = img_data[b[1]:b[3], b[0]:b[2], :]
                            img_features = img_pad(cropped_image, mode=crop_mode, size=target_dim[0])
                        elif 'context' in crop_type:
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])

                        elif 'surround' in crop_type:
                            b_org = list(map(int, b[0:4])).copy()
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            img_data[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = 128
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

                            img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))
                    if preprocess_input is not None:
                        img_features = preprocess_input(img_features)
                    if process:
                        expanded_img = np.expand_dims(img_features, axis=0)
                        img_features = convnet.predict(expanded_img)
                    # Save the file
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)

                # if using the generator save the cached features path and size of the features                                   
                if process and not self._generator:
                    if self._global_pooling == 'max':
                        img_features = np.squeeze(img_features)
                        img_features = np.amax(img_features, axis=0)
                        img_features = np.amax(img_features, axis=0)
                    elif self._global_pooling == 'avg':
                        img_features = np.squeeze(img_features)
                        img_features = np.average(img_features, axis=0)
                        img_features = np.average(img_features, axis=0)
                    else:
                        img_features = img_features.ravel()

                if self._generator:
                    img_seq.append(img_save_path)
                else:
                    img_seq.append(img_features)
            sequences.append(img_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        if self._generator:
            with open(sequences[0][0], 'rb') as fid:
                feat_shape = pickle.load(fid).shape
            if process:
                if self._global_pooling in ['max', 'avg']:
                    feat_shape = feat_shape[-1]
                else:
                    feat_shape = np.prod(feat_shape)
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]

        return sequences, feat_shape

        # Processing images anf generate features

    def get_optical_flow(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     regen_data=False):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'none' (no cropping)
                        'bbox' (crop using bounding box coordinates),
                        'context' (A region containing pedestrian and their local surround)
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            crop_mode: How to resize ond/or pad the corpped images (see utils.img_pad)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
            target_dim: Dimension of final visual features
            regen_data: Whether regenerate visual features. This will overwrite the cached features
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """

        # load the feature files if exists
        print("Generating {} features crop_type={} crop_mode={}\
               \nsave_path={}, ".format(data_type, crop_type, crop_mode, save_path))
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        # flow size (h,w)
        flow_size = read_flow_file(img_sequences[0][0].replace('images', 'optical_flow').replace('png', 'flo')).shape
        img_size = cv2.imread(img_sequences[0][0]).shape
        # A ratio to adjust the dimension of bounding boxes (w,h)
        box_resize_coef = (flow_size[1]/img_size[1], flow_size[0]/img_size[0])

        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            flow_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                optflow_save_folder = os.path.join(save_path, set_id, vid_id)
                ofp = imp.replace('images', 'optical_flow').replace('png', 'flo')
                # Modify the path depending on crop mode
                if crop_type == 'none':
                    optflow_save_path = os.path.join(optflow_save_folder, img_name + '.flo')
                else:
                    optflow_save_path = os.path.join(optflow_save_folder, img_name + '_' + p[0] + '.flo')

                # Check whether the file exists
                if os.path.exists(optflow_save_path) and not regen_data:
                    if not self._generator:
                        ofp_data = read_flow_file(optflow_save_path)
                else:
                    if 'flip' in imp:
                        ofp = ofp.replace('_flip', '')
                        flip_image = True
                    if crop_type == 'none':
                        ofp_image = read_flow_file(ofp)
                        ofp_data = cv2.resize(ofp_image, target_dim)
                        if flip_image:
                            ofp_data = cv2.flip(ofp_data, 1)
                    else:
                        ofp_image = read_flow_file(ofp)
                        # Adjust the size of bbox according to the dimensions of flow map
                        b = list(map(int, [b[0] * box_resize_coef[0], b[1] * box_resize_coef[1],
                                           b[2] * box_resize_coef[0], b[3] * box_resize_coef[1]]))
                        if flip_image:
                            ofp_image = cv2.flip(ofp_image, 1)
                        if crop_type == 'bbox':
                            cropped_image = ofp_image[b[1]:b[3], b[0]:b[2], :]
                            ofp_data = img_pad(cropped_image, mode=crop_mode, size=target_dim[0])
                        elif 'context' in crop_type:
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, ofp_image.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            cropped_image = ofp_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            ofp_data = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        elif 'surround' in crop_type:
                            b_org = b.copy()
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, ofp_image.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            ofp_image[b_org[1]:b_org[3], b_org[0]: b_org[2], :] = 0
                            cropped_image = ofp_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            ofp_data = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))

                    # Save the file
                    if not os.path.exists(optflow_save_folder):
                        os.makedirs(optflow_save_folder)
                    write_flow(ofp_data, optflow_save_path)

                # if using the generator save the cached features path and size of the features
                if self._generator:
                    flow_seq.append(optflow_save_path)
                else:
                    flow_seq.append(ofp_data)
            sequences.append(flow_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        if self._generator:
            feat_shape = read_flow_file(sequences[0][0]).shape
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]
        return sequences, feat_shape

    def get_data_sequence(self, data_type, data_raw, opts):
        """
        Generates raw sequences from a given dataset
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            opts:  Options for generating data samples
        Returns:
            A list of data samples extracted from raw data
            Positive and negative data counts
        """
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'crossing': data_raw['activities'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        normalize = opts['normalize_boxes']

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
        else:
            overlap = opts['overlap'] # if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res
            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs

            for seq in data_raw['bbox']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                range(start_idx, end_idx + 1, olap_res)])
        if normalize:
            for k in d.keys():
                if k != 'tte':
                    if k != 'box' and k != 'center':
                        for i in range(len(d[k])):
                            d[k][i] = d[k][i][1:]
                    else:
                        for i in range(len(d[k])):
                            d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                d[k] = np.array(d[k])
        else:
            for k in d.keys():
                d[k] = np.array(d[k])

        d['crossing'] = np.array(d['crossing'])[:, 0, :]
        pos_count = np.count_nonzero(d['crossing'])
        neg_count = len(d['crossing']) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count

    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        """
        Balances the ratio of positive and negative data samples. The less represented
        data type is augmented by flipping the sequences
        Args:
            d: Sequence of data samples
            img_width: Width of the images
            balance_tag: The tag to balance the data based on
        """
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                gt_augment = 1
            else:
                gt_augment = 0

            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]]
                                       for c in flipped]
                            d[k].append(flipped)
                        if k == 'box':
                            flipped = d[k][i].copy()
                            flipped = [np.array([img_width - b[2], b[1], img_width - b[0], b[3]])
                                       for b in flipped]
                            d[k].append(flipped)
                        if k == 'image':
                            flipped = d[k][i].copy()
                            flipped = [im.replace('.png', '_flip.png') for im in flipped]
                            d[k].append(flipped)
                        if k in ['speed', 'ped_id', 'crossing', 'walking', 'looking']:
                            d[k].append(d[k][i].copy())

            gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(42)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]

            # update the data
            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))

    def get_context_data(self, model_opts, data, data_type, feature_type):
        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')
        process = model_opts.get('process', True)
        aux_name = [self._backbone]
        if not process:
            aux_name.append('raw')
        aux_name = '_'.join(aux_name).strip('_')
        eratio = model_opts['enlarge_ratio']
        dataset = model_opts['dataset']

        data_gen_params = {'data_type': data_type, 'crop_type': 'none',
                           'target_dim': model_opts.get('target_dim', (224, 224))}
        if 'local_box' in feature_type:
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif 'local_context' in feature_type:
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'surround' in feature_type:
            data_gen_params['crop_type'] = 'surround'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'scene_context' in feature_type:
            data_gen_params['crop_type'] = 'none'

        save_folder_name = feature_type
        if 'flow' not in feature_type:
            save_folder_name = '_'.join([feature_type, aux_name])
            if 'local_context' in feature_type or 'surround' in feature_type:
                save_folder_name = '_'.join([save_folder_name, str(eratio)])
        data_gen_params['save_path'], _ = get_path(save_folder=save_folder_name,
                                                   dataset=dataset, save_root_folder='../data/features')
        if 'flow' in feature_type:
            return self.get_optical_flow(data['image'],
                                         data['box_org'],
                                         data['ped_id'],
                                         **data_gen_params)
        else:
            return self.load_images_crop_and_process(data['image'],
                                                     data['box_org'],
                                                     data['ped_id'],
                                                     process=process,
                                                     **data_gen_params)

    def get_data(self, data_type, data_raw, model_opts):
        """
        Generates data train/test/val data
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            model_opts: Model options for generating data
        Returns:
            A dictionary containing, data, data parameters used for model generation,
            effective dimension of data (the number of rgb images to be used calculated accorfing
            to the length of optical flow window) and negative and positive sample counts
        """

        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type:
                features, feat_shape = self.get_context_data(model_opts, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)

        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing']) # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'image': data['image'],
                'tte': data['tte'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def log_configs(self, config_path, batch_size, epochs,
                    lr, model_opts):

        # TODO: Update config by adding network attributes
        """
        Logs the parameters of the model and training
        Args:
            config_path: The path to save the file
            batch_size: Batch size of training
            epochs: Number of epochs for training
            lr: Learning rate of training
            model_opts: Data generation parameters (see get_data)
        """
        # Save config and training param files
        with open(config_path, 'wt') as fid:
            yaml.dump({'model_opts': model_opts, 
                       'train_opts': {'batch_size':batch_size, 'epochs': epochs, 'lr': lr}},
                       fid, default_flow_style=False)
        # with open(config_path, 'wt') as fid:
        #     fid.write("####### Model options #######\n")
        #     for k in opts:
        #         fid.write("%s: %s\n" % (k, str(opts[k])))

        #     fid.write("\n####### Network config #######\n")
        #     # fid.write("%s: %s\n" % ('hidden_units', str(self._num_hidden_units)))
        #     # fid.write("%s: %s\n" % ('reg_value ', str(self._regularizer_value)))

        #     fid.write("\n####### Training config #######\n")
        #     fid.write("%s: %s\n" % ('batch_size', str(batch_size)))
        #     fid.write("%s: %s\n" % ('epochs', str(epochs)))
        #     fid.write("%s: %s\n" % ('lr', str(lr)))

        print('Wrote configs to {}'.format(config_path))

    def class_weights(self, apply_weights, sample_count):
        """
        Computes class weights for imbalanced data used during training
        Args:
            apply_weights: Whether to apply weights
            sample_count: Positive and negative sample counts
        Returns:
            A dictionary of class weights or None if no weights to be calculated
        """
        if not apply_weights:
            return None

        total = sample_count['neg_count'] + sample_count['pos_count']
        # formula from sklearn
        #neg_weight = (1 / sample_count['neg_count']) * (total) / 2.0
        #pos_weight = (1 / sample_count['pos_count']) * (total) / 2.0
        
        # use simple ratio
        neg_weight = sample_count['pos_count']/total
        pos_weight = sample_count['neg_count']/total

        print("### Class weights: negative {:.3f} and positive {:.3f} ###".format(neg_weight, pos_weight))
        return {0: neg_weight, 1: pos_weight}

    def get_callbacks(self, learning_scheduler, model_path):
        """
        Creates a list of callabcks for training
        Args:
            learning_scheduler: Whether to use callbacks
        Returns:
            A list of call backs or None if learning_scheduler is false
        """
        callbacks = None

        # Set up learning schedulers
        if learning_scheduler:
            callbacks = []
            if 'early_stop' in learning_scheduler:
                default_params = {'monitor': 'val_loss',
                                  'min_delta': 1.0, 'patience': 5,
                                  'verbose': 1}
                default_params.update(learning_scheduler['early_stop'])
                callbacks.append(EarlyStopping(**default_params))

            if 'plateau' in learning_scheduler:
                default_params = {'monitor': 'val_loss',
                                  'factor': 0.2, 'patience': 5,
                                  'min_lr': 1e-08, 'verbose': 1}
                default_params.update(learning_scheduler['plateau'])
                callbacks.append(ReduceLROnPlateau(**default_params))

            if 'checkpoint' in learning_scheduler:
                default_params = {'filepath': model_path, 'monitor': 'val_loss',
                                  'save_best_only': True, 'save_weights_only': False,
                                  'save_freq': 'epoch', 'verbose': 2}
                default_params.update(learning_scheduler['checkpoint'])
                callbacks.append(ModelCheckpoint(**default_params))

        return callbacks

    def get_optimizer(self, optimizer):
        """
        Return an optimizer object
        Args:
            optimizer: The type of optimizer. Supports 'adam', 'sgd', 'rmsprop'
        Returns:
            An optimizer object
        """
        assert optimizer.lower() in ['adam', 'sgd', 'rmsprop'], \
        "{} optimizer is not implemented".format(optimizer)
        if optimizer.lower() == 'adam':
            return Adam
        elif optimizer.lower() == 'sgd':
            return SGD
        elif optimizer.lower() == 'rmsprop':
            return RMSprop

    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=60,
              lr=0.0005,
              optimizer='adam',
              learning_scheduler=None,
              model_opts=None):
        """
        Trains the models
        Args:
            data_train: Training data
            data_val: Validation data
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            optimizer: Optimizer for training
            learning_scheduler: Whether to use learning schedulers
            model_opts: Model options
        Returns:
            The path to the root folder of models
        """
        learning_scheduler = learning_scheduler or {}
        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, _ = get_path(**path_params, file_name='model.h5')

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size}) 

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0]

        # Create model
        train_model = self.get_model(data_train['data_params'])

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        callbacks = self.get_callbacks(learning_scheduler, model_path)


        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs,
                         lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    # Test Functions
    def test(self, data_test, model_path=''):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
            save_results: Save output of the model for visualization and analysis
        Returns:
            Evaluation metrics
        """
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)
            # try:
            #     model_opts = pickle.load(fid)
            # except:
            #     model_opts = pickle.load(fid, encoding='bytes')


        test_model = load_model(os.path.join(model_path, 'model.h5'))
        test_model.summary()

        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})

        test_results = test_model.predict(test_data['data'][0],
                                          batch_size=1, verbose=1)
        acc = accuracy_score(test_data['data'][1], np.round(test_results))
        f1 = f1_score(test_data['data'][1], np.round(test_results))
        auc = roc_auc_score(test_data['data'][1], np.round(test_results))
        roc = roc_curve(test_data['data'][1], test_results)
        precision = precision_score(test_data['data'][1], np.round(test_results))
        recall = recall_score(test_data['data'][1], np.round(test_results))
        pre_recall = precision_recall_curve(test_data['data'][1], test_results)
        
        # THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
            pickle.dump({'tte': test_data['tte'],
                         'pid': test_data['ped_id'],
                         'gt':test_data['data'][1],
                         'y': test_results,
                         'image': test_data['image']}, picklefile)


        print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

            with open(save_results_path, 'w') as fid:
                yaml.dump(results, fid)
        return acc, auc, f1, precision, recall

    def get_model(self, data_params):
        """
        Generates a model
        Args:
            data_params: Data parameters to use for model generation
        Returns:
            A model
        """
        raise NotImplementedError("get_model should be implemented")

    # Auxiliary function
    def _gru(self, name='gru', r_state=False, r_sequence=False):
        """
        A helper function to create a single GRU unit
        Args:
            name: Name of the layer
            r_state: Whether to return the states of the GRU
            r_sequence: Whether to return a sequence
        Return:
            A GRU unit
        """
        return GRU(units=self._num_hidden_units,
                   return_state=r_state,
                   return_sequences=r_sequence,
                   stateful=False,
                   kernel_regularizer=self._regularizer,
                   recurrent_regularizer=self._regularizer,
                   bias_regularizer=self._regularizer,
                   name=name)

    def _lstm(self, name='lstm', r_state=False, r_sequence=False):
        """
        A helper function to create a single LSTM unit
        Args:
            name: Name of the layer
            r_state: Whether to return the states of the LSTM
            r_sequence: Whether to return a sequence
        Return:
            A LSTM unit
        """
        return LSTM(units=self._num_hidden_units,
                    return_state=r_state,
                    return_sequences=r_sequence,
                    stateful=False,
                    kernel_regularizer=self._regularizer,
                    recurrent_regularizer=self._regularizer,
                    bias_regularizer=self._regularizer,
                    name=name)

    def create_stack_rnn(self, size, r_state=False, r_sequence=False):
        """
        Creates a stack of recurrent cells
        Args:
            size: The size of stack
            r_state: Whether to return the states of the GRU
            r_sequence: Whether the last stack layer to return a sequence
        Returns:
            A stacked recurrent model
        """
        cells = []
        for i in range(size):
            cells.append(self._rnn_cell(units=self._num_hidden_units,
                                        kernel_regularizer=self._regularizer,
                                        recurrent_regularizer=self._regularizer,
                                        bias_regularizer=self._regularizer, ))
        return RNN(cells, return_sequences=r_sequence, return_state=r_state)


def MLP(dropout=0.5, datasize = (16,4)):
    input_data = Input(shape=datasize)
    #model = Flatten(name='mlp_flatten')(input_data)
    model = Dense(64, activation='relu', name='fc1')(input_data)
    model = Dense(128, activation='relu', name='mlp_fc1')(model)
    model = Dense(256, activation='relu', name='mlp_fc2')(model)
    model = Dropout(dropout)(model)
    net_model = Model(input_data, model,name = 'MLP')
    return net_model


def attention_3d_block(hidden_states, dense_size=128, modality=''):
    """
    Many-to-one attention mechanism for Keras.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, 128)
    @author: felixhao28.
    """
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec'+modality)(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state'+modality)(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score'+modality)
    attention_weights = Activation('softmax', name='attention_weight'+modality)(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector'+modality)
    pre_activation = concatenate([context_vector, h_t], name='attention_output'+modality)
    attention_vector = Dense(dense_size, use_bias=False, activation='tanh', name='attention_vector'+modality)(pre_activation)
    return attention_vector


class TEACHER(ActionPredict):
    def __init__(self,
                 num_hidden_units=256,
                 distilling = False,
                 **kwargs):
        """
        Class init function

        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._distilling = distilling
        assert self._backbone in ['c3d', 'i3d'], 'Incorrect backbone {}! Should be C3D or I3D'.format(self._backbone)
        self._3dconv = C3DNet if self._backbone == 'c3d' else I3DNet

    def get_context_data(self, model_opts, data, data_type, feature_type):
        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')
        process = model_opts.get('process', True)
        aux_name = [self._backbone]
        if not process:
            aux_name.append('raw')
        aux_name = '_'.join(aux_name).strip('_')
        eratio = model_opts['enlarge_ratio']

        dataset = model_opts['dataset']

        data_gen_params = {'data_type': data_type, 'crop_type': 'none',
                           'target_dim': model_opts.get('target_dim', (224, 224))}
        if 'local_box' in feature_type:
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif 'local_context' in feature_type or 'local_flo' in feature_type:
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'surround' in feature_type:
            data_gen_params['crop_type'] = 'surround'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'scene_context' in feature_type:
            data_gen_params['crop_type'] = 'none'

        save_folder_name = feature_type
        if 'flow' not in feature_type:
            save_folder_name = '_'.join([feature_type, aux_name])
            if 'local_context' in feature_type or 'surround' in feature_type:
                save_folder_name = '_'.join([save_folder_name, str(eratio)])
        data_gen_params['save_path'], _ = get_path(save_folder=save_folder_name,
                                                   dataset=dataset, save_root_folder='../data/features')
        if 'flo' in feature_type:
            return self.get_optical_flow(data['image'],
                                         data['box_org'],
                                         data['ped_id'],
                                         process=process,
                                         **data_gen_params)
        else:
            return self.load_images_crop_and_process(data['image'],
                                                     data['box_org'],
                                                     data['ped_id'],
                                                     process=process,
                                                     **data_gen_params)

    def get_optical_flow(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     process=True,
                                     regen_data=False):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'none' (no cropping)
                        'bbox' (crop using bounding box coordinates),
                        'context' (A region containing pedestrian and their local surround)
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            crop_mode: How to resize ond/or pad the corpped images (see utils.img_pad)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
            target_dim: Dimension of final visual features
            process: Whether process the raw images using a neural network
            regen_data: Whether regenerate visual features. This will overwrite the cached features
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """

        # load the feature files if exists
        print("Generating {} features crop_type={} crop_mode={}\
              \nsave_path={}, ".format(data_type, crop_type, crop_mode,
                                       save_path))
        preprocess_dict = {'vgg16': vgg16.preprocess_input, 'resnet50': resnet50.preprocess_input}
        backbone_dict = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50}

        preprocess_input = preprocess_dict.get(self._backbone, None)
        if process:
            assert (self._backbone in ['vgg16', 'resnet50']), "{} is not supported".format(self._backbone)

        convnet = backbone_dict[self._backbone](input_shape=(224, 224, 3),
                                                include_top=False, weights='imagenet') if process else None
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                ofp = imp.replace('images', 'optical_flow')

                flip_image = False
                set_id = ofp.split('\\')[-3]
                vid_id = ofp.split('\\')[-2]
                img_name = ofp.split('\\')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)

                # Modify the path depending on crop mode
                if crop_type == 'none':
                    img_save_path = os.path.join(img_save_folder, img_name + '.pkl')
                else:
                    img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')

                # Check whether the file exists
                if os.path.exists(img_save_path) and not regen_data:
                    if not self._generator:
                        with open(img_save_path, 'rb') as fid:
                            try:
                                img_features = pickle.load(fid)
                            except:
                                img_features = pickle.load(fid, encoding='bytes')
                else:
                    if 'flip' in ofp:
                        ofp = ofp.replace('_flip', '')
                        flip_image = True
                    if crop_type == 'none':
                        img_data = cv2.imread(ofp)
                        img_features = cv2.resize(img_data, target_dim)
                        if flip_image:
                            img_features = cv2.flip(img_features, 1)
                    else:
                        img_data = cv2.imread(ofp)


                        if flip_image:
                            img_data = cv2.flip(img_data, 1)
                        if crop_type == 'bbox':
                            b = list(map(int, b[0:4]))
                            cropped_image = img_data[b[1]:b[3], b[0]:b[2], :]
                            img_features = img_pad(cropped_image, mode=crop_mode, size=target_dim[0])
                        elif 'context' in crop_type:
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])

                        elif 'surround' in crop_type:
                            b_org = list(map(int, b[0:4])).copy()
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            img_data[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = 128
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

                            img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))
                    if preprocess_input is not None:
                        img_features = preprocess_input(img_features)
                    if process:
                        expanded_img = np.expand_dims(img_features, axis=0)
                        img_features = convnet.predict(expanded_img)
                    # Save the file
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)

                # if using the generator save the cached features path and size of the features
                if process and not self._generator:
                    if self._global_pooling == 'max':
                        img_features = np.squeeze(img_features)
                        img_features = np.amax(img_features, axis=0)
                        img_features = np.amax(img_features, axis=0)
                    elif self._global_pooling == 'avg':
                        img_features = np.squeeze(img_features)
                        img_features = np.average(img_features, axis=0)
                        img_features = np.average(img_features, axis=0)
                    else:
                        img_features = img_features.ravel()

                if self._generator:
                    img_seq.append(img_save_path)
                else:
                    img_seq.append(img_features)
            sequences.append(img_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        if self._generator:
            with open(sequences[0][0], 'rb') as fid:
                feat_shape = pickle.load(fid).shape
            if process:
                if self._global_pooling in ['max', 'avg']:
                    feat_shape = feat_shape[-1]
                else:
                    feat_shape = np.prod(feat_shape)
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]

        return sequences, feat_shape

        # Processing images anf generate features


    def get_data(self, data_type, data_raw, model_opts):
        #assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']

        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type:
                if self._backbone == 'c3d':
                    model_opts_3d['target_dim'] = (112, 112)
                model_opts_3d['process'] = False
                features, feat_shape = self.get_context_data(model_opts_3d, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)

        one_hot_label = np.eye(2)[data['crossing']].reshape(-1,2)
        data.update({'crossing':one_hot_label})

        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_sizes):

        #data_sizes = data_params['data_sizes']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)
        attention_size = self._num_hidden_units

        conv3d_model = self._3dconv()
        #network_inputs.append(conv3d_model.input)


        for i in range(0, core_size-1):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + str(i)))
            x = conv3d_model(network_inputs[i])
            x = Dense(name='output_' + str(i), units=attention_size, activation='sigmoid')(x)
            encoder_outputs.append(x)

        network_inputs.append(Input(shape=data_sizes[core_size-1],  name='input_' + str(core_size-1)))
        x = MLP(datasize = data_sizes[core_size-1])(network_inputs[core_size-1])
        x = attention_3d_block(x, dense_size= attention_size, modality='_box')
        encoder_outputs.append(x)

        att_enc_out = []

        for i, enc_out in enumerate(encoder_outputs):
            x = Lambda(lambda x: K.expand_dims(x, axis=1))(enc_out)
            att_enc_out.append(x) # att_enc_out:local_flo, scene_context, local_context, box

        x = att_enc_out[0]
        if not len(att_enc_out)==1:
            for i in range(1,len(att_enc_out)):
                x = Concatenate(name='concat_modalities_'+str(i), axis=1)([x,att_enc_out[i]])
                encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality_'+str(i))
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(encodings)
        else:   # att_enc_out: box
            encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

        #encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

        model_output = Dropout(0.5)(encodings)
        model_output = Dense(10, activation='relu')(model_output)
        # model_output = Dense(1, activation='sigmod')(model_output)

        # output = Dense(2, activation='softmax',
        #                      name='output_dense',
        #                      activity_regularizer=regularizers.l2(0.001))(model_output)
        # dis_output = Dense(2, activation='softmax',
        #                      name='output_dense',
        #                      activity_regularizer=regularizers.l2(0.001))(model_output/2.0)

        if self._distilling:
            dis_output = Dense(2, activation='softmax',
                               name='output_dense',
                               activity_regularizer=regularizers.l2(0.001))(model_output / 2.0)
            net_model = Model(inputs=network_inputs,
                              outputs=[encodings,dis_output],name= 'teacher')
        else:
            output = Dense(2, activation='softmax',
                           name='output_dense',
                           activity_regularizer=regularizers.l2(0.001))(model_output)
            net_model = Model(inputs=network_inputs,
                              outputs=output,name= 'teacher')
            net_model.summary()
        # for i, layer in enumerate(net_model.layers):
        #     print(i, layer.name, layer.trainable)
        # sss
        plot_model(net_model, to_file='./plot/teacher.png')

        return net_model

    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=60,
              lr=0.0005,
              optimizer='adam',
              learning_scheduler=None,
              model_opts=None):
        """
        Trains the models
        Args:
            data_train: Training data
            data_val: Validation data
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            optimizer: Optimizer for training
            learning_scheduler: Whether to use learning schedulers
            model_opts: Model options
        Returns:
            The path to the root folder of models
        """
        learning_scheduler = learning_scheduler or {}
        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, _ = get_path(**path_params, file_name='model.h5')
        model_weights_path, _ = get_path(**path_params, file_name='model_weights.h5')

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0]

        # Create model
        train_model = self.get_model(data_train['data_params']['data_sizes'])

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        callbacks = self.get_callbacks(learning_scheduler, model_path)


        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)
            train_model.save_weights(model_weights_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs,
                         lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    def test(self, data_test, model_path=''):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
            save_results: Save output of the model for visualization and analysis
        Returns:
            Evaluation metrics
        """
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)
            # try:
            #     model_opts = pickle.load(fid)
            # except:
            #     model_opts = pickle.load(fid, encoding='bytes')

        test_model = load_model(os.path.join(model_path, 'model.h5'))
        test_model.summary()

        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})

        test_results = test_model.predict(test_data['data'][0],
                                          batch_size=1, verbose=1)

        a = test_data['data'][1][:,1].reshape(-1,1)
        b = test_results[:,1].reshape(-1,1)
        acc = accuracy_score(a, np.round(b))
        f1 = f1_score(a, np.round(b))
        auc = roc_auc_score(a, np.round(b))
        roc = roc_curve(a, b)
        precision = precision_score(a, np.round(b))
        recall = recall_score(a, np.round(b))
        pre_recall = precision_recall_curve(a, b)

        # THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
            pickle.dump({'tte': test_data['tte'],
                         'pid': test_data['ped_id'],
                         'gt': test_data['data'][1],
                         'y': test_results,
                         'image': test_data['image']}, picklefile)

        print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

            with open(save_results_path, 'w') as fid:
                yaml.dump(results, fid)
        return acc, auc, f1, precision, recall


class ResNet(ActionPredict):

    def __init__(self, model_dim=256, net_name='resnet18', distilling=True, **kwargs):
        self._model_dim = model_dim
        self._net_name = net_name
        self._distilling = distilling
        super(ResNet, self).__init__(**kwargs)

    def conv2d_bn(self, input, kernel_num, kernel_size=3, strides=1, layer_name='', padding_mode='same'):
        conv1 = Conv2D(kernel_num, kernel_size, strides=strides, padding=padding_mode, name=layer_name + '_conv1')(
            input)
        batch1 = BatchNormalization(name=layer_name + '_bn1')(conv1)
        return batch1

    def shortcut(self, fx, x, padding_mode='same', layer_name=''):
        layer_name += '_shortcut'
        if x.shape[-1] != fx.shape[-1]:
            k = fx.shape[-1]
            k = int(k)
            identity = self.conv2d_bn(x, kernel_num=k, kernel_size=1, padding_mode=padding_mode, layer_name=layer_name)
        else:
            identity = x
        return Add(name=layer_name + '_add')([identity, fx])

    def bottleneck(self, input, kernel_num, strides=1, layer_name='bottleneck', padding_mode='same'):
        k1, k2, k3 = kernel_num
        conv1 = self.conv2d_bn(input, kernel_num=k1, kernel_size=1, strides=strides, padding_mode=padding_mode,
                               layer_name=layer_name + '_1')
        relu1 = ReLU(name=layer_name + '_relu1')(conv1)
        conv2 = self.conv2d_bn(relu1, kernel_num=k2, kernel_size=3, strides=strides, padding_mode=padding_mode,
                               layer_name=layer_name + '_2')
        relu2 = ReLU(name=layer_name + '_relu2')(conv2)
        conv3 = self.conv2d_bn(relu2, kernel_num=k3, kernel_size=1, strides=strides, padding_mode=padding_mode,
                               layer_name=layer_name + '_3')
        # print(conv3.shape, input.shape)
        shortcut_add = self.shortcut(fx=conv3, x=input, layer_name=layer_name)
        relu3 = ReLU(name=layer_name + '_relu3')(shortcut_add)

        return relu3

    def basic_block(self, input, kernel_num=64, strides=1, layer_name='basic', padding_mode='same'):
        # k1, k2 = kernel
        conv1 = self.conv2d_bn(input, kernel_num=kernel_num, strides=strides, kernel_size=3,
                               layer_name=layer_name + '_1', padding_mode=padding_mode)
        relu1 = ReLU(name=layer_name + '_relu1')(conv1)
        conv2 = self.conv2d_bn(relu1, kernel_num=kernel_num, strides=strides, kernel_size=3,
                               layer_name=layer_name + '_2', padding_mode=padding_mode)
        relu2 = ReLU(name=layer_name + '_relu2')(conv2)

        shortcut_add = self.shortcut(fx=relu2, x=input, layer_name=layer_name)
        relu3 = ReLU(name=layer_name + '_relu3')(shortcut_add)
        return relu3

    def make_layer(self, input, block, block_num, kernel_num, layer_name=''):
        x = input
        for i in range(1, block_num + 1):
            x = block(x, kernel_num=kernel_num, strides=1, layer_name=layer_name + str(i), padding_mode='same')
        return x

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]


        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:

            features = data[d_type]
            feat_shape = features.shape[1:]

            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned

        one_hot_label = np.eye(2)[data['crossing']].reshape(-1,2)
        data.update({'crossing':one_hot_label})
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_sizes):

        # data_sizes = data_params['data_sizes']

        network_inputs = Input(shape=(data_sizes[-1]), dtype='float32', name="inputs")

        embeddings = Embedding(self._model_dim)(network_inputs)

        embeddings = tf.expand_dims(embeddings, -1)

        block_setting = {}
        block_setting['resnet18'] = {'block': self.basic_block, 'block_num': [2, 2, 2, 2],
                                     'kernel_num': [64, 128, 256, 512]}
        block_setting['resnet34'] = {'block': self.basic_block, 'block_num': [3, 4, 6, 3],
                                     'kernel_num': [64, 128, 256, 512]}
        block_setting['resnet50'] = {'block': self.bottleneck, 'block_num': [3, 4, 6, 3],
                                     'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                    [256, 256, 1024], [512, 512, 2048]]}
        block_setting['resnet101'] = {'block': self.bottleneck, 'block_num': [3, 4, 23, 3],
                                      'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                     [256, 256, 1024], [512, 512, 2048]]}
        block_setting['resnet152'] = {'block': self.bottleneck, 'block_num': [3, 8, 36, 3],
                                      'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                     [256, 256, 1024], [512, 512, 2048]]}
        # self._net_name = 'resnet18' if not block_setting.__contains__(self._net_name) else self._net_name
        block_num = block_setting[self._net_name]['block_num']
        kernel_num = block_setting[self._net_name]['kernel_num']
        block = block_setting[self._net_name]['block']

        conv1 = self.conv2d_bn(embeddings, 64, kernel_size=7, strides=2, layer_name='first_conv')
        pool1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name='pool1')(conv1)

        conv = pool1
        for i in range(4):
            conv = self.make_layer(conv, block=block, block_num=block_num[i], kernel_num=kernel_num[i],
                                   layer_name='layer' + str(i + 1))

        encodings = GlobalAvgPool2D(name='globalavgpool')(conv)
        encodings = Dense(256, activation='relu')(encodings)

        x = Dropout(0.5)(encodings)
        x = Dense(10, activation='relu')(x)

        outputs = Dense(2, activation='softmax',
                        name='output_dense',
                        activity_regularizer=regularizers.l2(0.001))(x)

        dis_outputs = Dense(2, activation='softmax',
                            name='dis_output_dense',
                            activity_regularizer=regularizers.l2(0.001))(x / 2.0)

        if self._distilling:
            net_model = Model(inputs=network_inputs,
                              outputs=[encodings, dis_outputs, outputs],
                              name='student')
        else:
            net_model = Model(inputs=network_inputs,
                              outputs=outputs,
                              name='student')
            net_model.summary()

        return net_model

    def resnet(self,network_inputs):


        embeddings = Embedding(self._model_dim)(network_inputs)

        embeddings = tf.expand_dims(embeddings, -1)

        block_setting = {}
        block_setting['resnet18'] = {'block': self.basic_block, 'block_num': [2, 2, 2, 2],
                                     'kernel_num': [64, 128, 256, 512]}
        block_setting['resnet34'] = {'block': self.basic_block, 'block_num': [3, 4, 6, 3],
                                     'kernel_num': [64, 128, 256, 512]}
        block_setting['resnet50'] = {'block': self.bottleneck, 'block_num': [3, 4, 6, 3],
                                     'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                    [256, 256, 1024], [512, 512, 2048]]}
        block_setting['resnet101'] = {'block': self.bottleneck, 'block_num': [3, 4, 23, 3],
                                      'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                     [256, 256, 1024], [512, 512, 2048]]}
        block_setting['resnet152'] = {'block': self.bottleneck, 'block_num': [3, 8, 36, 3],
                                      'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                     [256, 256, 1024], [512, 512, 2048]]}
        # self._net_name = 'resnet18' if not block_setting.__contains__(self._net_name) else self._net_name
        block_num = block_setting[self._net_name]['block_num']
        kernel_num = block_setting[self._net_name]['kernel_num']
        block = block_setting[self._net_name]['block']

        conv1 = self.conv2d_bn(embeddings, 64, kernel_size=7, strides=2, layer_name='first_conv')
        pool1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name='pool1')(conv1)

        conv = pool1
        for i in range(4):
            conv = self.make_layer(conv, block=block, block_num=block_num[i], kernel_num=kernel_num[i],
                                   layer_name='layer' + str(i + 1))

        encodings = GlobalAvgPool2D(name='globalavgpool')(conv)
        encodings = Dense(256, activation='relu')(encodings)

        x = Dropout(0.5)(encodings)
        x = Dense(10, activation='relu')(x)

        outputs = Dense(2, activation='softmax',
                        name='output_dense',
                        activity_regularizer=regularizers.l2(0.001))(x)

        dis_outputs = Dense(2, activation='softmax',
                            name='dis_output_dense',
                            activity_regularizer=regularizers.l2(0.001))(x / 2.0)

        if self._distilling:
            return encodings, dis_outputs, outputs
        else:
            return outputs


    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=60,
              lr=0.0005,
              optimizer='adam',
              learning_scheduler=None,
              model_opts=None):
        """
        Trains the models
        Args:
            data_train: Training data
            data_val: Validation data
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            optimizer: Optimizer for training
            learning_scheduler: Whether to use learning schedulers
            model_opts: Model options
        Returns:
            The path to the root folder of models
        """
        learning_scheduler = learning_scheduler or {}
        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, _ = get_path(**path_params, file_name='model.h5')
        model_weights_path, _ = get_path(**path_params, file_name='model_weights.h5')

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0]


        # Create model
        train_model = self.get_model(data_train['data_params']['data_sizes'])

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        #train_model.compile(loss=hard_loss, optimizer=optimizer, metrics=['accuracy'])
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        callbacks = self.get_callbacks(learning_scheduler, model_path)

        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)
            train_model.save_weights(model_weights_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs,
                         lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    def test(self, data_test, model_path=''):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
            save_results: Save output of the model for visualization and analysis
        Returns:
            Evaluation metrics
        """
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)
            # try:
            #     model_opts = pickle.load(fid)
            # except:
            #     model_opts = pickle.load(fid, encoding='bytes')

        custom_objects = {'Embedding': Embedding,'PositionEncoding':PositionEncoding,'MultiHeadAttention':MultiHeadAttention,
                          'PositionWiseFeedForward':PositionWiseFeedForward,'LayerNormalization1':LayerNormalization1,
                          'hard_loss':hard_loss}
        test_model = load_model(os.path.join(model_path, 'model.h5'),custom_objects =custom_objects)
        test_model.summary()

        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})

        test_results = test_model.predict(test_data['data'][0],
                                          batch_size=1, verbose=1)

        a = test_data['data'][1][:,1].reshape(-1,1)
        b = test_results[:,1].reshape(-1,1)
        acc = accuracy_score(a, np.round(b))
        f1 = f1_score(a, np.round(b))
        auc = roc_auc_score(a, np.round(b))
        roc = roc_curve(a, b)
        precision = precision_score(a, np.round(b))
        recall = recall_score(a, np.round(b))
        pre_recall = precision_recall_curve(a, b)

        # THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
            pickle.dump({'tte': test_data['tte'],
                         'pid': test_data['ped_id'],
                         'gt': a,
                         'y': test_results,
                         'image': test_data['image']}, picklefile)

        print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

            with open(save_results_path, 'w') as fid:
                yaml.dump(results, fid)
        return acc, auc, f1, precision, recall

    def test1(self, data_test, model_path=''):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
            save_results: Save output of the model for visualization and analysis
        Returns:
            Evaluation metrics
        """
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)
        opts['model_opts']['obs_input_type']= ['box']

            # try:
            #     model_opts = pickle.load(fid)
            # except:
            #     model_opts = pickle.load(fid, encoding='bytes')

        # custom_objects = {'Embedding': Embedding,'PositionEncoding':PositionEncoding,'MultiHeadAttention':MultiHeadAttention,
        #                   'PositionWiseFeedForward':PositionWiseFeedForward,'LayerNormalization1':LayerNormalization1,
        #                   'hard_loss':hard_loss}
        # custom_objects = {'Embedding': Embedding, 'PositionEncoding': PositionEncoding,
        #                   'MultiHeadAttention': MultiHeadAttention,
        #                   'PositionWiseFeedForward': PositionWiseFeedForward,
        #                   'LayerNormalization1': LayerNormalization1,
        #                   'DistillingLoss': DistillingLoss}
        custom_objects = {'Embedding': Embedding, 'PositionEncoding': PositionEncoding,
                          'MultiHeadAttention': MultiHeadAttention,
                          'PositionWiseFeedForward': PositionWiseFeedForward,
                          'LayerNormalization1': LayerNormalization1,
                          'hard_loss': hard_loss,
                          'fea_loss': fea_loss,
                          'soft_loss': soft_loss,
                          'Fea_Loss': Fea_Loss,
                          'Soft_Loss': Soft_Loss}



        # ts_model = load_model(os.path.join(model_path, 'model.h5'),custom_objects =custom_objects)
        # test_model = ts_model.layers[2]
        # test_model.save(os.path.join(model_path, 'student_model.h5'))
        test_model = load_model(os.path.join(model_path, 'model.h5'), custom_objects=custom_objects)

        test_model.summary()
        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})
        test_results = test_model.predict(test_data['data'][0],
                                          batch_size=1, verbose=1)


        a = test_data['data'][1][:, 1].reshape(-1, 1)
        b = test_results[:, 1].reshape(-1, 1)

        acc = accuracy_score(a, np.round(b))
        f1 = f1_score(a, np.round(b))
        auc = roc_auc_score(a, np.round(b))
        roc = roc_curve(a, b)
        precision = precision_score(a, np.round(b))
        recall = recall_score(a, np.round(b))
        pre_recall = precision_recall_curve(a, b)

        # THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        # with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
        #     pickle.dump({'tte': test_data['tte'],
        #                  'pid': test_data['ped_id'],
        #                  'gt': a,
        #                  'y': test_results,
        #                  'image': test_data['image']}, picklefile)

        print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))

        # save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        # if not os.path.exists(save_results_path):
        #     results = {'acc': acc,
        #                'auc': auc,
        #                'f1': f1,
        #                'roc': roc,
        #                'precision': precision,
        #                'recall': recall,
        #                'pre_recall_curve': pre_recall}
        #
        #     with open(save_results_path, 'w') as fid:
        #         yaml.dump(results, fid)

        return acc, auc, f1, precision, recall




    def __init__(self, model_dim=256, distilling=False, **kwargs):
        self._model_dim = model_dim

        self._distilling = distilling
        super(ShuffleNet, self).__init__(dynamic=True)

    def _group_conv(self, x, filters, kernel, stride, groups):

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        in_channels = K.int_shape(x)[channel_axis]

        # number of input channels per group
        nb_ig = in_channels // groups
        # number of output channels per group
        nb_og = filters // groups

        gc_list = []
        # Determine whether the number of filters is divisible by the number of groups
        assert filters % groups == 0

        for i in range(groups):
            if channel_axis == -1:
                x_group = Lambda(lambda z: z[:, :, :, i * nb_ig: (i + 1) * nb_ig])(x)
            else:
                x_group = Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :, :])(x)
            gc_list.append(Conv2D(filters=nb_og, kernel_size=kernel, strides=stride,
                                  padding='same', use_bias=False)(x_group))

        return Concatenate(axis=channel_axis)(gc_list)

    def _channel_shuffle(self, x, groups):

        if K.image_data_format() == 'channels_last':
            height, width, in_channels = K.int_shape(x)[1:]
            channels_per_group = in_channels // groups
            pre_shape = [-1, height, width, groups, channels_per_group]
            dim = (0, 1, 2, 4, 3)
            later_shape = [-1, height, width, in_channels]
        else:
            in_channels, height, width = K.int_shape(x)[1:]
            channels_per_group = in_channels // groups
            pre_shape = [-1, groups, channels_per_group, height, width]
            dim = (0, 2, 1, 3, 4)
            later_shape = [-1, in_channels, height, width]

        x = Lambda(lambda z: K.reshape(z, pre_shape))(x)
        x = Lambda(lambda z: K.permute_dimensions(z, dim))(x)
        x = Lambda(lambda z: K.reshape(z, later_shape))(x)

        return x

    def _shufflenet_unit(self, inputs, filters, kernel, stride, groups, stage, bottleneck_ratio=0.25):

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        in_channels = K.int_shape(inputs)[channel_axis]
        bottleneck_channels = int(filters * bottleneck_ratio)

        if stage == 2:
            x = Conv2D(filters=bottleneck_channels, kernel_size=kernel, strides=1,
                       padding='same', use_bias=False)(inputs)
        else:
            x = self._group_conv(inputs, bottleneck_channels, (1, 1), 1, groups)
        x = BatchNormalization(axis=channel_axis)(x)
        x = ReLU()(x)

        x = self._channel_shuffle(x, groups)

        x = DepthwiseConv2D(kernel_size=kernel, strides=stride, depth_multiplier=1,
                            padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if stride == 2:
            x = self._group_conv(x, filters - in_channels, (1, 1), 1, groups)
            x = BatchNormalization(axis=channel_axis)(x)
            avg = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(inputs)
            x = Concatenate(axis=channel_axis)([x, avg])
        else:
            x = self._group_conv(x, filters, (1, 1), 1, groups)
            x = BatchNormalization(axis=channel_axis)(x)
            x = add([x, inputs])

        return x

    def _stage(self, x, filters, kernel, groups, repeat, stage):

        x = self._shufflenet_unit(x, filters, kernel, 2, groups, stage)

        for i in range(1, repeat):
            x = self._shufflenet_unit(x, filters, kernel, 1, groups, stage)

        return x

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]


        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:

            features = data[d_type]
            feat_shape = features.shape[1:]

            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned

        one_hot_label = np.eye(2)[data['crossing']].reshape(-1,2)
        data.update({'crossing':one_hot_label})
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_sizes):

        network_inputs = Input(shape=(data_sizes[-1]), dtype='float32', name="inputs")

        embeddings = Embedding(self._model_dim)(network_inputs)

        embeddings = tf.expand_dims(embeddings, -1)

        x = Conv2D(24, (3, 3), strides=2, padding='same', use_bias=True, activation='relu')(embeddings)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

        x = self._stage(x, filters=384, kernel=(3, 3), groups=8, repeat=4, stage=2)
        x = self._stage(x, filters=768, kernel=(3, 3), groups=8, repeat=8, stage=3)
        x = self._stage(x, filters=1536, kernel=(3, 3), groups=8, repeat=4, stage=4)

        encodings = GlobalAveragePooling2D()(x)
        encodings = Dense(256, activation='relu')(encodings)

        x = Dropout(0.5)(encodings)
        x = Dense(10, activation='relu')(x)

        outputs = Dense(2, activation='softmax',
                        name='output_dense',
                        activity_regularizer=regularizers.l2(0.001))(x)

        dis_outputs = Dense(2, activation='softmax',
                            name='dis_output_dense',
                            activity_regularizer=regularizers.l2(0.001))(x / 2.0)

        if self._distilling:
            net_model = Model(inputs=network_inputs,
                              outputs=[encodings, dis_outputs, outputs],
                              name='student')
        else:
            net_model = Model(inputs=network_inputs,
                              outputs=outputs,
                              name='student')
            net_model.summary()

        return net_model

    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=60,
              lr=0.0005,
              optimizer='adam',
              learning_scheduler=None,
              model_opts=None):
        """
        Trains the models
        Args:
            data_train: Training data
            data_val: Validation data
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            optimizer: Optimizer for training
            learning_scheduler: Whether to use learning schedulers
            model_opts: Model options
        Returns:
            The path to the root folder of models
        """
        learning_scheduler = learning_scheduler or {}
        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, _ = get_path(**path_params, file_name='model.h5')
        model_weights_path, _ = get_path(**path_params, file_name='model_weights.h5')

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0]

        # Create model
        train_model = self.get_model(data_train['data_params']['data_sizes'])

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        callbacks = self.get_callbacks(learning_scheduler, model_path)

        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)
            train_model.save_weights(model_weights_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs,
                         lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    def test(self, data_test, model_path=''):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
            save_results: Save output of the model for visualization and analysis
        Returns:
            Evaluation metrics
        """
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)
            # try:
            #     model_opts = pickle.load(fid)
            # except:
            #     model_opts = pickle.load(fid, encoding='bytes')

        custom_objects = {'Embedding': Embedding,'PositionEncoding':PositionEncoding,'MultiHeadAttention':MultiHeadAttention,
                          'PositionWiseFeedForward':PositionWiseFeedForward,'LayerNormalization1':LayerNormalization1}
        test_model = load_model(os.path.join(model_path, 'model.h5'),custom_objects =custom_objects)
        test_model.summary()

        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})

        test_results = test_model.predict(test_data['data'][0],
                                          batch_size=1, verbose=1)

        a = test_data['data'][1][:,1].reshape(-1,1)
        b = test_results[:,1].reshape(-1,1)
        acc = accuracy_score(a, np.round(b))
        f1 = f1_score(a, np.round(b))
        auc = roc_auc_score(a, np.round(b))
        roc = roc_curve(a, b)
        precision = precision_score(a, np.round(b))
        recall = recall_score(a, np.round(b))
        pre_recall = precision_recall_curve(a, b)

        # THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
            pickle.dump({'tte': test_data['tte'],
                         'pid': test_data['ped_id'],
                         'gt': a,
                         'y': test_results,
                         'image': test_data['image']}, picklefile)

        print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

            with open(save_results_path, 'w') as fid:
                yaml.dump(results, fid)
        return acc, auc, f1, precision, recall

def action_prediction(model_name):
    for cls in ActionPredict.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    raise Exception('Model {} is not valid!'.format(model_name))


class DataGenerator(Sequence):

    def __init__(self,
                 data=None,
                 labels=None,
                 data_sizes=None,
                 process=False,
                 global_pooling=None,
                 input_type_list=None,
                 batch_size=32,
                 shuffle=True,
                 to_fit=True,
                 stack_feats=False):
        self.data = data
        self.labels = labels
        self.process = process
        self.global_pooling = global_pooling
        self.input_type_list = input_type_list
        self.batch_size = 1 if len(self.labels) < batch_size  else batch_size        
        self.data_sizes = data_sizes
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.stack_feats = stack_feats
        self.indices = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data[0])/self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size: (index+1)*self.batch_size]

        X = self._generate_X(indices)
        if self.to_fit:
            y = self._generate_y(indices)
            return X, y
        else:
            return X

    def _get_img_features(self, cached_path):
        with open(cached_path, 'rb') as fid:
            try:
                img_features = pickle.load(fid)
            except:
                img_features = pickle.load(fid, encoding='bytes')
        if self.process:
            if self.global_pooling == 'max':
                img_features = np.squeeze(img_features)
                img_features = np.amax(img_features, axis=0)
                img_features = np.amax(img_features, axis=0)
            elif self.global_pooling == 'avg':
                img_features = np.squeeze(img_features)
                img_features = np.average(img_features, axis=0)
                img_features = np.average(img_features, axis=0)
            else:
                img_features = img_features.ravel()        
        return img_features

    def _generate_X(self, indices):
        X = []
        for input_type_idx, input_type in enumerate(self.input_type_list):
            features_batch = np.empty((self.batch_size, *self.data_sizes[input_type_idx]))
            num_ch = features_batch.shape[-1]//len(self.data[input_type_idx][0])
            for i, index in enumerate(indices):
                if isinstance(self.data[input_type_idx][index][0], str):
                    cached_path_list = self.data[input_type_idx][index]
                    for j, cached_path in enumerate(cached_path_list):
                        if 'flow' in input_type:
                            img_features = read_flow_file(cached_path)
                        else:
                            img_features = self._get_img_features(cached_path)

                        if len(cached_path_list) == 1:
                            # for static model if only one image in the sequence
                            features_batch[i, ] = img_features
                        else:
                            if self.stack_feats and 'flow' in input_type:
                                features_batch[i,...,j*num_ch:j*num_ch+num_ch] = img_features
                            else:
                                features_batch[i, j, ] = img_features
                else:
                    features_batch[i, ] = self.data[input_type_idx][index]
            X.append(features_batch)
        return X

    def _generate_y(self, indices):
        return np.array(self.labels[indices])

class Fea_Loss(Layer):
    def __init__(self, **kwargs):

        super(Fea_Loss, self).__init__(**kwargs)



    def call(self, inputs):

        source, target = inputs

        return MSLE(source, target)



    def compute_output_shape(self, input_shape):
        return input_shape[0]

class Soft_Loss(Layer):
    def __init__(self, T = 2,**kwargs):
        self._T = T
        super(Soft_Loss, self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config().copy()
        config.update({'T': self._T})
        return config

    def call(self, inputs):
        softtarget, softpredict = inputs

        return self._T * kld(softtarget, softpredict)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class Distilling(ActionPredict):


    #  ResNet  PIE 代码
    def __init__(self,
                 num_hidden_units=256,
                 model_dim=32, n_heads=8, encoder_stack=6,
                 decoder_stack=6, feed_forward_size=2048, dropout_rate=0.1,
                 T = 2, student_model='ResNet',
                 **kwargs):
        """
        Class init function

        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._encoder_stack = encoder_stack
        self._decoder_stack = decoder_stack
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate
        self._T = T
        self._student_model = student_model
        # Network parameters
        self._num_hidden_units = num_hidden_units
        assert self._backbone in ['c3d', 'i3d'], 'Incorrect backbone {}! Should be C3D or I3D'.format(self._backbone)
        self._3dconv = C3DNet if self._backbone == 'c3d' else I3DNet
    def get_context_data(self, model_opts, data, data_type, feature_type):
        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')
        process = model_opts.get('process', True)
        aux_name = [self._backbone]
        if not process:
            aux_name.append('raw')
        aux_name = '_'.join(aux_name).strip('_')
        eratio = model_opts['enlarge_ratio']
        dataset = model_opts['dataset']

        data_gen_params = {'data_type': data_type, 'crop_type': 'none',
                           'target_dim': model_opts.get('target_dim', (224, 224))}
        if 'local_box' in feature_type:
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif 'local_context' in feature_type or 'local_flo' in feature_type:
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'surround' in feature_type:
            data_gen_params['crop_type'] = 'surround'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'scene_context' in feature_type:
            data_gen_params['crop_type'] = 'none'

        save_folder_name = feature_type
        if 'flow' not in feature_type:
            save_folder_name = '_'.join([feature_type, aux_name])
            if 'local_context' in feature_type or 'surround' in feature_type:
                save_folder_name = '_'.join([save_folder_name, str(eratio)])
        data_gen_params['save_path'], _ = get_path(save_folder=save_folder_name,
                                                   dataset=dataset, save_root_folder='../data/features')
        if 'flo' in feature_type:
            return self.get_optical_flow(data['image'],
                                         data['box_org'],
                                         data['ped_id'],
                                         process=process,
                                         **data_gen_params)
        else:
            return self.load_images_crop_and_process(data['image'],
                                                     data['box_org'],
                                                     data['ped_id'],
                                                     process=process,
                                                     **data_gen_params)

    def get_optical_flow(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     process=True,
                                     regen_data=False):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'none' (no cropping)
                        'bbox' (crop using bounding box coordinates),
                        'context' (A region containing pedestrian and their local surround)
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            crop_mode: How to resize ond/or pad the corpped images (see utils.img_pad)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
            target_dim: Dimension of final visual features
            process: Whether process the raw images using a neural network
            regen_data: Whether regenerate visual features. This will overwrite the cached features
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """

        # load the feature files if exists
        print("Generating {} features crop_type={} crop_mode={}\
              \nsave_path={}, ".format(data_type, crop_type, crop_mode,
                                       save_path))
        preprocess_dict = {'vgg16': vgg16.preprocess_input, 'resnet50': resnet50.preprocess_input}
        backbone_dict = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50}

        preprocess_input = preprocess_dict.get(self._backbone, None)
        if process:
            assert (self._backbone in ['vgg16', 'resnet50']), "{} is not supported".format(self._backbone)

        convnet = backbone_dict[self._backbone](input_shape=(224, 224, 3),
                                                include_top=False, weights='imagenet') if process else None
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                ofp = imp.replace('images', 'optical_flow')
                flip_image = False
                set_id = ofp.split('\\')[-3]
                vid_id = ofp.split('\\')[-2]
                img_name = ofp.split('\\')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)

                # Modify the path depending on crop mode
                if crop_type == 'none':
                    img_save_path = os.path.join(img_save_folder, img_name + '.pkl')
                else:
                    img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')

                # Check whether the file exists
                if os.path.exists(img_save_path) and not regen_data:
                    if not self._generator:
                        with open(img_save_path, 'rb') as fid:
                            try:
                                img_features = pickle.load(fid)
                            except:
                                img_features = pickle.load(fid, encoding='bytes')
                else:
                    if 'flip' in ofp:
                        ofp = ofp.replace('_flip', '')
                        flip_image = True
                    if crop_type == 'none':
                        img_data = cv2.imread(ofp)
                        img_features = cv2.resize(img_data, target_dim)
                        if flip_image:
                            img_features = cv2.flip(img_features, 1)
                    else:
                        img_data = cv2.imread(ofp)
                        if flip_image:
                            img_data = cv2.flip(img_data, 1)
                        if crop_type == 'bbox':
                            b = list(map(int, b[0:4]))
                            cropped_image = img_data[b[1]:b[3], b[0]:b[2], :]
                            img_features = img_pad(cropped_image, mode=crop_mode, size=target_dim[0])
                        elif 'context' in crop_type:
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])

                        elif 'surround' in crop_type:
                            b_org = list(map(int, b[0:4])).copy()
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            img_data[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = 128
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

                            img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))
                    if preprocess_input is not None:
                        img_features = preprocess_input(img_features)
                    if process:
                        expanded_img = np.expand_dims(img_features, axis=0)
                        img_features = convnet.predict(expanded_img)
                    # Save the file
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)

                # if using the generator save the cached features path and size of the features
                if process and not self._generator:
                    if self._global_pooling == 'max':
                        img_features = np.squeeze(img_features)
                        img_features = np.amax(img_features, axis=0)
                        img_features = np.amax(img_features, axis=0)
                    elif self._global_pooling == 'avg':
                        img_features = np.squeeze(img_features)
                        img_features = np.average(img_features, axis=0)
                        img_features = np.average(img_features, axis=0)
                    else:
                        img_features = img_features.ravel()

                if self._generator:
                    img_seq.append(img_save_path)
                else:
                    img_seq.append(img_features)
            sequences.append(img_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        if self._generator:
            with open(sequences[0][0], 'rb') as fid:
                feat_shape = pickle.load(fid).shape
            if process:
                if self._global_pooling in ['max', 'avg']:
                    feat_shape = feat_shape[-1]
                else:
                    feat_shape = np.prod(feat_shape)
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]

        return sequences, feat_shape

        # Processing images anf generate features


    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type:
                if self._backbone == 'c3d':
                    model_opts_3d['target_dim'] = (112, 112)
                model_opts_3d['process'] = False
                features, feat_shape = self.get_context_data(model_opts_3d, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)

        one_hot_label = np.eye(2)[data['crossing']].reshape(-1, 2)
        data.update({'crossing': one_hot_label})

        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}
    def get_model(self, data_params):
        data_types = data_params['data_types']
        data_sizes = data_params['data_sizes']
        network_inputs = []
        core_size = len(data_sizes)
        if core_size == 4: #scene_context,local_context,local_flo,box
            _dir = '17Jan2022-20h30m08s'
        elif core_size == 3: #scene_context,local_context,box
            _dir = '18Jan2022-19h04m47s'
        elif core_size == 2: #scene_context,box
            _dir = '19Jan2022-10h00m47s'
        elif core_size == 1: #box
            _dir = '17Jan2022-19h58m46s'
        else:
            print('input error')
            exit()

        teacher_model_path = os.path.join(r'G:\B\pedestrian\mystudent\data\models\virtual\TEACHER',_dir,
                                          'model_weights.h5')

        teacher_model = TEACHER(distilling=True).get_model(data_sizes)
        #teacher_model.summary()
        teacher_model.load_weights(teacher_model_path , by_name=True)
        teacher_model.trainable = False

        if self._student_model == 'Transformer':
            student_model = Transformer(distilling=True).get_model(data_sizes)
        elif self._student_model == 'ResNet':
            student_model = ResNet(distilling=True).get_model(data_sizes)
        elif self._student_model == 'MobileNet':
            student_model = MobileNet(distilling=True).get_model(data_sizes)
        elif self._student_model == 'ShuffleNet':
            student_model = ShuffleNet(distilling=True).get_model(data_sizes)
        else:
            print('student_model error')
            exit()

        for i in range(core_size):
            network_inputs.append(Input(shape=data_sizes[i], name=data_types[i]))
        #network_inputs.append(Input(shape=(1), name='label'))

        t_feature,t_dis_out = teacher_model(network_inputs)
        s_feature,s_dis_out,s_out = student_model(network_inputs[-1])

        fea_loss = Fea_Loss()([t_feature, s_feature])
        soft_loss = Soft_Loss()([t_dis_out,s_dis_out])


        model = Model(inputs=network_inputs, outputs=[s_out,fea_loss,soft_loss],name= 'distilling')

        print('mm'*20)
        model.summary()


        return model



    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=60,
              lr=0.0005,
              optimizer='adam',
              learning_scheduler=None,
              model_opts=None):
        """
        Trains the models
        Args:
            data_train: Training data
            data_val: Validation data
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            optimizer: Optimizer for training
            learning_scheduler: Whether to use learning schedulers
            model_opts: Model options
        Returns:
            The path to the root folder of models
        """
        learning_scheduler = learning_scheduler or {}
        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, _ = get_path(**path_params, file_name='model.h5')
        model_weights_path, _ = get_path(**path_params, file_name='model_weights.h5')

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0]

        # Create model
        train_model = self.get_model(data_train['data_params'])

        print('121'*10)
        train_model.summary()


        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model.compile(loss=[hard_loss,fea_loss,soft_loss], optimizer=optimizer, metrics={'student': 'accuracy'},loss_weights=[1,1,1])
        #train_model.compile(loss=[fea_loss], optimizer=optimizer, metrics=['accuracy'])

        callbacks = self.get_callbacks(learning_scheduler, model_path)



        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight= class_w,
                                  verbose=1,
                                  callbacks=callbacks)

        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)
            train_model.save_weights(model_weights_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs,
                         lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    def test(self, data_test, model_path=''):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
            save_results: Save output of the model for visualization and analysis
        Returns:
            Evaluation metrics
        """
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)
            # try:
            #     model_opts = pickle.load(fid)
            # except:
            #     model_opts = pickle.load(fid, encoding='bytes')

        custom_objects = {'Embedding': Embedding, 'PositionEncoding': PositionEncoding,
                          'MultiHeadAttention': MultiHeadAttention,
                          'PositionWiseFeedForward': PositionWiseFeedForward,
                          'LayerNormalization1': LayerNormalization1,
                          'hard_loss': hard_loss,
                          'fea_loss': fea_loss,
                          'soft_loss': soft_loss,
                          'Fea_Loss': Fea_Loss,
                          'Soft_Loss': Soft_Loss}
        test_model = load_model(os.path.join(model_path, 'model.h5'), custom_objects=custom_objects)
        test_model.summary()

        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})

        test_results = test_model.predict(test_data['data'][0],
                                          batch_size=1, verbose=1)[0]

        a = test_data['data'][1][:, 1].reshape(-1, 1)
        b = test_results[:,1].reshape(-1,1)

        acc = accuracy_score(a, np.round(b))
        f1 = f1_score(a, np.round(b))
        auc = roc_auc_score(a, np.round(b))
        roc = roc_curve(a, b)
        precision = precision_score(a, np.round(b))
        recall = recall_score(a, np.round(b))
        pre_recall = precision_recall_curve(a, b)

        # THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
            pickle.dump({'tte': test_data['tte'],
                         'pid': test_data['ped_id'],
                         'gt': a,
                         'y': test_results,
                         'image': test_data['image']}, picklefile)

        print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision,
                                                                                      recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

            with open(save_results_path, 'w') as fid:
                yaml.dump(results, fid)
        return acc, auc, f1, precision, recall