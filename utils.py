import os
import sys
import copy
import numpy as np
import itertools
from datetime import datetime
import h5py
import pickle
import joblib
import argparse
import pprint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import RobustScaler, QuantileTransformer


def create_directories(base_dir, epochs=None, log_prefix=''):

    current_time = datetime.utcnow().isoformat()

    epoch_str = f'_{epochs}_epochs' if epochs is not None else ''

    if log_prefix != '':
        if type(log_prefix)==str and ',' in log_prefix:
            log_prefix = log_prefix.replace(',','_')
        log_dir = base_dir + f'logs/{log_prefix}/{current_time}{epoch_str}/'
    else:
        log_dir = base_dir + f'logs/{current_time}{epoch_str}/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fig_dir = log_dir + 'figures/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    saved_models_dir = log_dir + 'saved_models/'
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)

    lr_finder_dir = log_dir + 'lr_finder/'
    if not os.path.exists(lr_finder_dir):
        os.makedirs(lr_finder_dir)

    dirs = {'log'           : log_dir,
            'fig'           : fig_dir,
            'saved_models'  : saved_models_dir,
            'lr_finder'     : lr_finder_dir}

    return dirs


def merge_dicts(base_dict, subset_dict=None, in_depth=False):
    """
    For every key in the dictionary subset_dict that is also a key in the
    base_dict dictionary, overwrite the value of that key in base_dict with the
    one from subset_dict, and return the resulting (partially overwritten)
    version of base_dict.
    """

    if subset_dict is None:
        return base_dict

    merged_dict = copy.deepcopy(base_dict)

    for key in subset_dict:
        if not isinstance(subset_dict[key], dict):
            merged_dict[key] = subset_dict[key]
        else:
            if not merged_dict[key]: # Is empty
                merged_dict[key] = subset_dict[key]
            else:
                if not isinstance(merged_dict[key], dict) or not in_depth:
                    merged_dict[key] = subset_dict[key]
                else:
                    merged_dict[key] = merge_dicts(merged_dict[key], subset_dict[key], in_depth=True)

    return merged_dict


def save_dict(dictionary, path, overwrite=False, save_pkl=False):
    """
    Args :
        dictionary :
            *dict*

            Dictionary to be saved.
        path :
            *str*

            Path to which the dictionary should be saved. The path must have a
            file ending (i.e. '.*') to work properly with the 'save_pkl' option.
            If the path already exists, the given dict will be appended
            to what is already in the file.
        overwrite :
            *bool*

            Whether to overwrite an already existing text file in the same path.
            Picled files are always overwritten, if they exist.
        save_pkl :
            *bool*

            Saves dictionary as a pickled file, in addition to the text file.
    """

    if os.path.exists(path):
        mod = 'a'
        if overwrite:
            mod = 'w+'
    else:
        mod = 'w+'

    with open(path, mod) as f:
        pprint.sorted = lambda x, key=None: x # disables sorting
        f.write(pprint.pformat(dictionary))

    if save_pkl:
        path_for_pkl = path.rsplit('.', 1)[0] + '.pkl'
        with open(path_for_pkl, 'wb+') as f:
            pickle.dump(dictionary,f)


def drop_key(dictionary, key_to_be_dropped):
    """Returns the input dictionary without the key_to_be_dropped, out of place."""

    if type(key_to_be_dropped)==str:
        if key_to_be_dropped in dictionary:
            return {key:dictionary[key] for key in dictionary if key!=key_to_be_dropped}
        else:
            return dictionary
    else: # List of keys to be dropped
        for key_in_list in key_to_be_dropped:
            dictionary = drop_key(dictionary, key_in_list)
        return dictionary


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def point_asdict(search_space_as_dict, point_as_list):
    return {key:val for key,val in zip(search_space_as_dict,point_as_list)}


def point_aslist(search_space_as_dict, point_as_dict):
    return [point_as_dict[key] for key in search_space_as_dict]


def str2bool(v):
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_str_combs(base, extension):
    fix = [base, extension]
    return [''.join(name) for name in itertools.chain.from_iterable([itertools.product(fix[i], ['','_'], fix[i+1]) for i in [0,-1]])]


def boolify(obj):
    return bool(obj.size if isinstance(obj, np.ndarray) else obj)


def set_auto_lr(params):

    def _get_auto_lr(optimizer_name, batch_size, range_factor=3):

        # Set scaling of learning rate based on optimizer
        if optimizer_name.lower() in ['nadam']:
            optimum_lr = 2.5*1e-5 * np.sqrt(batch_size)

        # Other optimizers could be implemented as similarly:
        # elif optimizer_name.lower() in ['padam']:
        #     optimum_lr = 1e-3 * np.sqrt(batch_size)
        #
        # elif optimizer_name.lower() in ['yogi']:
        #     optimum_lr = 3*1e-5 * np.sqrt(batch_size)

        else:
            raise NotImplementedError(f'Optimizer {optimizer_name} not implemented in get_auto_lr.')

        return optimum_lr, [optimum_lr * range_factor**(-0.5),
                            optimum_lr * range_factor**(0.5)]

    if isinstance(params['optimizer'], dict):
        optimizer_name = params['optimizer']['class_name']
    else:
        optimizer_name = params['optimizer']
    lr, lr_schedule_range = _get_auto_lr(optimizer_name, params['batch_size'])
    if params['lr_schedule']['name'] is not None:
        params['lr_schedule']['range'] = lr_schedule_range
        print('Learning rate range for learning rate schedule '
              f'{params["lr_schedule"]["name"]} set to '
              f'({params["lr_schedule"]["range"][0]:.5e}, '
              f'{params["lr_schedule"]["range"][1]:.5e}) '
              'by auto_lr.')
    else:
        if isinstance(params['optimizer'], dict):
            params['optimizer']['config']['lr'] = lr
        else:
            params['optimizer'] = {'class_name':params['optimizer'],
                                   'config':{'lr':lr}}
        print('Learning rate has been set to  '
              f'{params["optimizer"]["config"]["lr"]:.5e} by auto_lr.')

    return params


def make_mask(data_path, n_points, rm_bad_reco=False, zee_only=False,
              lh_cut_name=None, rm_conv=None, rm_unconv=None):

    def _report_num_discarded(mask, mask_name=None):
        if mask_name is not None:
            print(mask_name + ':')
        for set_name in mask:
            print(f'Flagged {np.sum(np.logical_not(mask[set_name]))} data points (out of {len(mask[set_name])}, or {np.sum(np.logical_not(mask[set_name]))/len(mask[set_name])*100:.2f}%) from the {set_name} set for removal.')
        print('')

    mask_exists = False

    print('\nMASKS:\n')

    # Remove badly reconstructed datapoints
    if rm_bad_reco:
        with h5py.File(data_path, 'r') as hf:
            reco_vars = load_atlas_data(path=data_path, n_points=n_points, scalar_names=['p_e', 'p_truth_E'], verbose=False)
        mask = {set_name:np.abs(reco_vars[set_name]['scalars'][:,0] / reco_vars[set_name]['scalars'][:,1] - 1) <= 0.5 for set_name in reco_vars}
        # k = 5
        # mask = {set_name:(reco_vars[set_name]['scalars'][:,0] / reco_vars[set_name]['scalars'][:,1] <= k) *
        #                  (reco_vars[set_name]['scalars'][:,0] / reco_vars[set_name]['scalars'][:,1] > 1/k) for set_name in reco_vars}
        _report_num_discarded(mask, mask_name='rm_bad_reco')
        mask_exists = True

    # Remove all but Zee
    if zee_only:
        with h5py.File(data_path, 'r') as hf:
            channel = load_atlas_data(path=data_path, n_points=n_points, scalar_names=['mcChannelNumber'], verbose=False)

        # Combine if needed
        if mask_exists:
            mask_new = {set_name:channel[set_name]['scalars'][:,0] == 361106 for set_name in channel}
            mask = {set_name:mask[set_name] * mask_new[set_name] for set_name in mask}
            _report_num_discarded(mask_new, mask_name='zee_only')
        else:
            mask = {set_name:channel[set_name]['scalars'][:,0] == 361106 for set_name in channel}
            _report_num_discarded(mask, mask_name='zee_only')
        mask_exists = True

    # Remove all that doesn't make some PID cut
    if lh_cut_name is not None:
        with h5py.File(data_path, 'r') as hf:
            lh_value = load_atlas_data(path=data_path, n_points=n_points, scalar_names=[lh_cut_name], verbose=False)

        # Combine if needed
        if mask_exists:
            mask_new = {set_name:lh_value[set_name]['scalars'][:,0] == 1 for set_name in lh_value}
            mask = {set_name:mask[set_name] * mask_new[set_name] for set_name in mask}
            _report_num_discarded(mask_new, mask_name=lh_cut_name)
        else:
            mask = {set_name:lh_value[set_name]['scalars'][:,0] == 1 for set_name in lh_value}
            _report_num_discarded(mask, mask_name=lh_cut_name)
        mask_exists = True

    # Remove all either converted (rm_conv=True) or unconverted (rm_unconv=True) photons
    if rm_conv is not None or rm_unconv is not None:
        with h5py.File(data_path, 'r') as hf:
            conv_type = load_atlas_data(path=data_path, n_points=n_points, scalar_names=['p_photonConversionType'], verbose=False)

        # Combine if needed
        if mask_exists:
            if rm_conv:
                mask_new = {set_name:conv_type[set_name]['scalars'][:,0] == 0 for set_name in conv_type}
            elif rm_unconv:
                mask_new = {set_name:conv_type[set_name]['scalars'][:,0] > 0 for set_name in conv_type}
            mask = {set_name:mask[set_name] * mask_new[set_name] for set_name in mask}
            _report_num_discarded(mask_new, mask_name=('rm_conv' if rm_conv else 'rm_unconv'))
        else:
            if rm_conv:
                mask = {set_name:conv_type[set_name]['scalars'][:,0] == 0 for set_name in conv_type}
            elif rm_unconv:
                mask = {set_name:conv_type[set_name]['scalars'][:,0] > 0 for set_name in conv_type}
            _report_num_discarded(mask, mask_name=('rm_conv' if rm_conv else 'rm_unconv'))
        mask_exists = True

    _report_num_discarded(mask, mask_name='In total')

    return mask


def apply_mask(data, mask, verbose=True, skip_name=None):
    """Returns data with masked datapoints removed."""

    for set_name in mask:
        if skip_name is not None and set_name==skip_name:
            continue
        for dataset in data[set_name]:
            if dataset=='images':
                for img_set in data[set_name][dataset]:
                    if boolify(data[set_name][dataset]): # Check for truthiness
                        data[set_name][dataset][img_set] = data[set_name][dataset][img_set][mask[set_name]]
            else:
                if boolify(data[set_name][dataset]): # Check for truthiness
                    data[set_name][dataset] = data[set_name][dataset][mask[set_name]]
        if verbose:
            print(f'Removed {np.sum(np.logical_not(mask[set_name]))} data points (out of {len(mask[set_name])}, or {np.sum(np.logical_not(mask[set_name]))/len(mask[set_name])*100:.2f}%) from the {set_name} set.')

    return data


def standardize(data, dataset_name, variable_index=None, scaler_name='Robust', save_path=None):
    """
    Standardize data using an sklearn scaler. Will standardize data[dataset_name]
    in-place if variable_index is None, and data[dataset_name][:,variable_index]
    otherwise.
    """

    # Instantiate scaler
    if scaler_name.lower()=='robust':
        scaler = RobustScaler(copy=False)
    elif scaler_name.lower()=='quantile':
        scaler = QuantileTransformer(output_distribution='normal', copy=False)
    else:
        print('WARNING: Scaler name not recognized. Using RobustScaler.')
        scaler = RobustScaler(copy=False)

    # Fit scaler
    if variable_index is not None:
        print(np.expand_dims(data['train'][dataset_name][:,variable_index],1).shape)
        scaler.fit(np.expand_dims(data['train'][dataset_name][:,variable_index],1))
    else:
        scaler.fit(np.expand_dims(data['train'][dataset_name],1))

    # Transform data
    for set_name in data:
        if variable_index is not None:
            data[set_name][dataset_name][:,variable_index] = np.squeeze(scaler.transform(np.expand_dims(data[set_name][dataset_name][:,variable_index],1)))
        else:
            data[set_name][dataset_name] = np.squeeze(scaler.transform(np.expand_dims(data[set_name][dataset_name],1)))

    # Save scaler
    if save_path is not None:
        with open(save_path, 'wb+') as f:
            joblib.dump(scaler, f)


def set_outliers(array, value=np.median, outlier_criterion=2e2,
                 tol_frac_of_outliers=.0001, verbose=True):
    """Set all outliers to the median (or some value of choice).

    Parameters
    ----------
    array : array-like
        Input array.
    value : callable or float
        If a callable is provided, such as the default `np.median` then it
        is expected to be called value(array). Otherwise, provide a float.
    outlier_criterion : float
        Number of robust standard deviations (absolute deviation from median
        (AD) divided by median(AD)) above which an element in `array` will be
        considered an outlier.
    tol_frac_of_outliers : float
        Maximum allowed fraction of outliers in `array`.
    verbose : bool
        Verbose output.

    Returns
    -------
    Array with outliers set to value.
    """

    def abs_dev(a, c=0.67448975, center=np.median):
        # From https://www.statsmodels.org/dev/generated/statsmodels.robust.scale.mad.html
        """
        The Absolute Deviation along given axis of an array

        Parameters
        ----------
        a : array-like
            Input array.
        c : float, optional
            The normalization constant.  Defined as scipy.stats.norm.ppf(3/4.),
            which is approximately .6745.
        center : callable or float
            If a callable is provided, such as the default `np.median` then it
            is expected to be called center(a). The axis argument will be applied
            via np.apply_over_axes. Otherwise, provide a float.

        Returns
        -------
        Absolute deviation : float
            abs(`a` - center))/`c`
        """
        a = np.asarray(a)
        if callable(center):
            center = np.apply_over_axes(center, a, 0)
        return np.fabs(a-center)/c

    if callable(value):
        value = np.apply_over_axes(value, array, 0)[0]

    # Find outliers
    ad = abs_dev(array)
    mad = np.median(ad)
    outliers = ad/mad > outlier_criterion
    if not np.any(outliers):
        if verbose:
            print(f'Variable had no outliers. Returning array as it was given.')
        return array
    outlier_inds = np.array(np.nonzero(outliers)[0])
    frac_outliers = outlier_inds.shape[0]/array.shape[0]
    if verbose:
        print(f'{frac_outliers*100:.4f}% outliers found.')
    if frac_outliers > tol_frac_of_outliers:
        if verbose:
            print(f'This is above the tolerated {tol_frac_of_outliers*100:.4f}%. '
                  'Returning array as it was given.')
        return array
    else:
        if verbose:
            print(f'Outliers {array[outlier_inds]} have been set to {value:.4f}.')
        array[outlier_inds] = value
        return array


def get_default_params():
    """
    Returns a dictionary containing default parameters to be passed to the model
    container.

    Please see the README for documentation.
    """

    params = {
          # Training
          'use_hls4ml'                 : False,
          'epochs'                     : 1,
          'batch_size'                 : 32,
          'loss'                       : 'mse',
          'metrics'                    : None,
          'optimizer'                  : 'Nadam',
          'lr_finder'                  : {'use':False,
                                          'scan_range':[1e-4, 1e-2],
                                          'epochs':1,
                                          'scale':'linear',
                                          'prompt_for_input':False},
          'lr_schedule'                : {'name':None,
                                          'range':[1e-3,5e-3],
                                          'step_size_factor':2,
                                          'kwargs':{}},
          'auto_lr'                    : False,

          # Misc.
          'use_earlystopping'          : False,
          'restore_best_weights'       : False,
          'pretrained_model'           : {'use':False,
                                          'weights_path':'./weights.h5',
                                          'params_path':None,
                                          'layers_to_load':[],
                                          'freeze_loaded_layers':[]},
          'n_gpus'                     : 0,
          'data_generator'             : {'use':False,
                                          'n_workers':1,
                                          'max_queue_size':10,
                                          'n_points':{'train':None, 'val':None, 'test':None},
                                          'load_kwargs':{'path':'./data.h5',
                                                         'target_name':None,
                                                         'img_names':None,
                                                         'gate_img_prefix':None,
                                                         'scalar_names':None,
                                                         'track_names':None,
                                                         'max_tracks':None,
                                                         'multiply_output_name':None,
                                                         'sample_weight_name':None}},
          'upsampling'                 : {'use':False,
                                          'wanted_size':(100,100),
                                          'interpolation':'nearest'},

          # Submodels
          'top'                        : {'initialization':'orthogonal',
                                          'activation':None,
                                          'normalization':None,
                                          'layer_reg':{},
                                          'dropout':None,
                                          'units':[64,1],
                                          'final_activation':None},
          'cnn'                        : {'initialization':'orthogonal',
                                          'activation':None,
                                          'normalization':None,
                                          'layer_reg':{},
                                          'dropout':None,
                                          'cnn_type':'simple',
                                          'conv_dim':2,
                                          'block_depths':[1],
                                          'n_init_filters':4,
                                          'init_kernel_size':5,
                                          'rest_kernel_size':3,
                                          'init_strides':1,
                                          'rest_strides':1,
                                          'cardinality':1,
                                          'use_squeeze_and_excite':False,
                                          'squeeze_and_excite_ratio':16,
                                          'globalavgpool':False,
                                          'downsampling':None,
                                          'min_size_for_downsampling':2},
          'network_in_network'         : {'initialization':'orthogonal',
                                          'activation':None,
                                          'normalization':None,
                                          'layer_reg':{},
                                          'dropout':None,
                                          'use':False,
                                          'units':[64],
                                          'strides':1,
                                          'globalavgpool':False},
          'scalar_net'                 : {'initialization':'orthogonal',
                                          'activation':None,
                                          'normalization':None,
                                          'layer_reg':{},
                                          'dropout':None,
                                          'units':[],
                                          'connect_to':[]},
          'FiLM_gen'                   : {'initialization':'orthogonal',
                                          'activation':None,
                                          'normalization':None,
                                          'layer_reg':{},
                                          'dropout':None,
                                          'use':False,
                                          'units':[64]},
          'track_net'                  : {'initialization':'orthogonal',
                                          'activation':None,
                                          'normalization':None,
                                          'layer_reg':{},
                                          'dropout':None,
                                          'phi_units':[64],
                                          'rho_units':[64],
                                          'connect_to':[]},
          'gate_net'                   : {'initialization':'orthogonal',
                                          'activation':None,
                                          'normalization':None,
                                          'layer_reg':{},
                                          'dropout':None,
                                          'units':[],
                                          'use_res':False,
                                          'final_activation':None,
                                          'final_activation_init':[1.0],
                                          'merge_method':'concatenate'},
          'quantize'                   : {'use':False,
                                          'q_config':{  "QDense": {
                                                            "kernel_quantizer": "quantized_bits(16,6,1,alpha=1)",
                                                            "bias_quantizer": "quantized_bits(16,6,1,alpha=1)"
                                                        },
                                                    
                                                        "QActivation": { "relu": "quantized_relu(16,8)" },
                                                    
                                                        "QConv2D": {
                                                            "kernel_quantizer": "quantized_bits(16,6,1,alpha=1)",
                                                            "bias_quantizer": "quantized_bits(16,6,1,alpha=1)"},
                                                        
                                                        "activation": "quantized_relu(32,16)",
                                                        #"QBatchNormalization": {}
                                                     }}
          }

    return params


def load_atlas_data(path, n_points={'train':None, 'val':None, 'test':None},
                    target_name=None, # 'p_truth_E' for ER, 'Truth' for PID
                    img_names=None, gate_img_prefix=None, scalar_names=None,
                    track_names=None, max_tracks=None, multiply_output_name=None,
                    sample_weight_name=None, verbose=True, use_hls4ml = False, target_factor = 0.1):
    """
    Function for loading ATLAS data.

    Args:
    -----
        path : *str*
            Path to the data to be loaded.

        n_points : *dict*
            This dictionary defines the names of the subsets of the data loaded
            (by way of the keys of `n_points`), as well as how many datapoints
            should be loaded for each set (by way of the values of `n_points`).

            Its *keys* should be either `'train'` and `'val'`, or `'train'`,
            `'val'` and `'test'`. This means that while both training and
            validation data are required, test data can be left out. If test
            data is left out, no evaluation will be carried out at the end of
            training. It is important that the naming conventions of `'train'`,
            `'val'` and `'test'` are kept, as this is the only way the
            subsequent handling of the data knows which sets are meant to be
            used for what.

            The *values* of `n_points` dictate how many points of each set will
            be loaded. If `None`, all datapoints in the set corresponding to
            the key will be loaded. If instead a number n (*int* or *float*,
            where the latter will be converted into an *int*), only the first n
            datapoints of that set will be loaded. If a *list* of *int*s, this
            will be interpreted as indices, and only datapoints with these
            indices will be loaded (primarily used in conjunction with data
            generators).

        target_name : *str or list of strs or None*
            Name of target. This should correspond to a dataset in the HDF5 file.
            If a list, the list must contain two strings, corresponding to two
            datasets in the HDF5 file; the returned target will then be the first
            divided by the second. E.g., for ER, you may want to divide by the
            accordion energy.

        img_names : *list of strs or None*
            List of image names to be loaded. Each image name will have its own
            entry in the returned data dict (acccessed by its name).
            If `None` (default), no images will be loaded and no CNN will be
            constructed.

        gate_img_prefix : *str or None*
            If not `None`, images with the same names as those given in
            `img_names`, but with `gated` preprended will be added
            to the data dict (accessed by the full name). This will cause
            these images to be processed by the `gate_net` submodel.
            E.g., if `gate_img_prefix` is `'time'`, and `img_names` is
            `['em_barrel']`, then the datasets with name `'time_em_barrel'`
            will be loaded, and can be accessed in the returned data dict by
            `data['train']['images']['gated_em_barrel']`.

        scalar_names : *list of strs or None*
            List of scalar variable names to be loaded into the `'scalars'`
            entry of the returned data dict.
            If `None`, no scalar variables will be loaded.

        track_names : *list of strs or None*
            List of scalar variable names  to be loaded into the `'tracks'`
            entry of the returned data dict.
            If `None`, no track variables will be loaded.

        max_tracks : *int or None*
            Maximum number of tracks. Sequences less than this will be zero-
            padded, while sequences more than this will be truncated, both in
            the end of the sequence.
            If `None`, the maximum length found in the dataset at hand will be
            used.

        multiply_output_name : *str or None*
            Scalar variable name  to be loaded into the `'multiply_output_with'`
            entry of the returned data dict.
            This is used as a variable that the output neuron of the model
            should be multiplied with. This product is then the final output of
            the model. For instance, if our target is an energy, setting
            multiply_output_name to 'p_eAccCluster' (the accordion energy) would
            make the model predict the correction to the accordion energy,
            instead of the energy itself.
            If `None`, no such variable will be loaded.

        sample_weight_name : *str or None*
            Load sample weights to be used in training. Useful if classes are
            unbalanced or if certain marginal signal/background distributions do
            not follow each other nicely.
            The name should correspond to a dataset in the HDF5 file.

        verbose : *bool*
            Verbose output.

    Returns:
    --------
        data : *dict*
            Dictionary of data. This dictionary (in its first level) will
            contain the same keys as those in n_points. Each of these keys point
            to another dict containing the different datasets (such as images
            and scalars).
            For example, if
            `n_points = {'train':None, 'val':2000, 'test':1e3}` and if
            `scalar_names` is not `None`, then we would access our training set
            scalars (which, like all the other datasets, should be a numpy
            array) like so: `data['train']['scalars']`.
            If we instead want to access the test set images, and `img_names`
            is `['img']`, we do `data['test']['images']['img']`.
            The target (or label in the case of classification) datasets are
            named `'targets'`.
            See the documentation in the README for more details.
    """

    if verbose:
        print('Loading data.')

    # Prepare loading only the wanted data points
    # Make copy of n_points to use for loading (changes to n_points will persist
    # outside of this function)
    load_n_points = copy.deepcopy(n_points)

    for set_name in load_n_points:
        # If a list (of specific indices) is not given
        if not hasattr(load_n_points[set_name], '__iter__'):
            # If all data points should be loaded
            if load_n_points[set_name] is None:
                load_n_points[set_name] = slice(None)
            # If only the first int(load_n_points[set_name]) data points should be loaded
            else:
                load_n_points[set_name] = int(load_n_points[set_name])
                n_samples = load_n_points[set_name]
                load_n_points[set_name] = slice(None,load_n_points[set_name])
                if verbose:
                    print(f'Loading only the {n_samples} first data points of the {set_name} set.')
        else:
            load_n_points[set_name] = list(np.sort(load_n_points[set_name]))

    # Load the data
    with h5py.File(path, 'r') as hf:

        # Function for loading and concatenating all scalar-likes
        def expand_and_concat(list_to_iter, set_name, n_samples, axis=1):
            return np.concatenate(tuple(np.expand_dims(hf[f'{set_name}/{it}'][n_samples], axis=axis) for it in list_to_iter), axis=axis)

        # Function for loading and concatenating images
        def load_img(img_name, set_name, lrs, samples):
            # Get shape of images
            lr_shape = hf[f'{set_name}/{img_name}_Lr{lrs[0]}'].shape

            # Get number of samples to be loaded
            if type(samples)==list:
                lr_shape = (len(samples),) + lr_shape[1:]
            elif type(samples)==slice:
                if n_points[set_name] is not None:
                    lr_shape = (int(n_points[set_name]),) + lr_shape[1:]

            # Preallocate
            imgs = np.zeros(lr_shape + (len(lrs),))

            for i,lr in enumerate(lrs):
                imgs[:,:,:,i] = hf[f'{set_name}/{img_name}_Lr{lr}'][samples]

            return imgs

        data = {set_name:{'images':{},
                          'scalars':{},
                          'tracks':{},
                          'targets':{},
                          'multiply_output_with':{},
                          'sample_weights':{}} for set_name in load_n_points}

        # Add targets to data
        if target_name is not None:

            #scale down targets to fraction if use hls4ml later
            if use_hls4ml is True:
                print('Since using hls4ml later, Targets will be scaled down by {}'.format(target_factor))
                
            if type(target_name) in [list, tuple]:
                for set_name in load_n_points:
                    data[set_name]['targets'] = hf[f'{set_name}/{target_name[0]}'][load_n_points[set_name]] * target_factor
                    data[set_name]['targets'] /= hf[f'{set_name}/{target_name[1]}'][load_n_points[set_name]]
            else:
                for set_name in load_n_points:
                    data[set_name]['targets'] = hf[f'{set_name}/{target_name}'][load_n_points[set_name]] * target_factor

        # Add scalars
        for set_name in load_n_points:
            if scalar_names is not None:
                data[set_name]['scalars'] = expand_and_concat(scalar_names,
                                                              set_name,
                                                              load_n_points[set_name])

        # Add images
        if img_names is not None:
            # Add cell images to data
            # Different parts of the detector have a different number of layers
            layers_to_include = {img_name:[0,1,2,3] for img_name in ['em_barrel', 'em_endcap', 'lar_endcap']}
            layers_to_include['tile_barrel'] = [1,2,3]
            layers_to_include['tile_gap'] = [1]

            for set_name in load_n_points:
                for img_name in img_names:
                    data[set_name]['images'][img_name] = load_img(img_name, set_name,
                                                                  layers_to_include[img_name],
                                                                  load_n_points[set_name])

            # Add images to be gated to data
            if gate_img_prefix is not None:
                for set_name in load_n_points:
                    for img_name in img_names:
                        data[set_name]['images'][f'gate_{img_name}'] = load_img(f'{gate_img_prefix}_{img_name}',
                                                                                set_name,
                                                                                layers_to_include[img_name],
                                                                                load_n_points[set_name])

        # Add tracks
        if track_names is not None:
            # Add tracks as ndarrays of ndarrays
            for set_name in load_n_points:
                data[set_name]['tracks'] = expand_and_concat(track_names,
                                                             set_name,
                                                             load_n_points[set_name])

            # Find maximum number of tracks in any of the datasets
            if max_tracks is None:
                max_tracks = max(max(point[0].shape[0] for point in data[set_name]['tracks']) for set_name in load_n_points)

            # Pad with zeros
            for set_name in load_n_points:
                data[set_name]['tracks'] = np.concatenate(
                    tuple(np.expand_dims(pad_sequences(tracks, maxlen=max_tracks, dtype='float32',
                                                       padding='post', truncating='post', value=0.0).T,
                                         axis=0) for tracks in data[set_name]['tracks']), axis=0)

        # Add multiply_output_name
        if multiply_output_name is not None:
            for set_name in load_n_points:
                data[set_name]['multiply_output_with'] = (hf[f'{set_name}/{multiply_output_name}'][load_n_points[set_name]])

        # Add sample weights (from reweighting)
        if sample_weight_name is not None:
            for set_name in load_n_points:
                data[set_name]['sample_weights'] = hf[f'{set_name}/{sample_weight_name}'][load_n_points[set_name]]

    if verbose:
        print('Data loaded.')

    return data
