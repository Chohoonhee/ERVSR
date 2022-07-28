from easydict import EasyDict as edict
from configs.config import get_config as main_config
from configs.config import log_config, print_config
import math
import torch
import numpy as np

def get_config(project = '', mode = '', config = '', data = '', LRS = '', batch_size = 8):

    ### GLOBAL
    config = main_config(project, mode, config, data, LRS, batch_size)

    ### LOCAL
    ## Training
    actual_batch_size = config.batch_size * torch.cuda.device_count()
    config.lr_init = 2e-4
    config.lr_min = 1e-6
    config.wi = None # weight init (xavier)
    config.win = None # weight init (normal)

    config.patch_size = 64
    config.frame_itr_num = 26
    config.frame_num = 13

    config.loss = '1*L1+'
    config.CX_vgg_layer = 'relu3_4'

    ## SR
    config.flag_HD_in = False # whether to SR above original input resolution
    config.scale = 4 # SR scale (2 | 4)
    if config.scale == 2:
        config.matching_ksize = 4 # must be even
    else:
        config.matching_ksize = 2 # must be even

    config.refine_val_lr = 1
    config.refine_val_hr = 1
    if config.flag_HD_in:
        config.matching_ksize *= config.scale

    ## Model specifications
    config.trainer = 'trainer'
    config.network = 'RefVSR'
    config.num_blocks = 30
    config.mid_channels = 48
    #config.reset_branch = None
    config.reset_branch = config.frame_itr_num # enable this if results contain holes

    ## Dataset
    #if config.data == 'RealMCVSR':
    #    total_frame_num = 19426
    #    video_num = 137
    # config.IpE = math.floor((len(list(range(0, total_frame_num - (config.frame_itr_num-1), config.frame_itr_num)))) / actual_batch_size) * config.frame_itr_num
    # max_epoch = math.floor(config.total_itr / IpE)

    ## Triaining
    config.total_itr = 300000
    if config.LRS == 'LD':
        # lr_decay
        config.decay_period = [400000]
        config.decay_rate = 0.25
        config.warmup_itr = -1
    elif config.LRS == 'CA':
        # Cosine Anealing
        config.warmup_itr = -1
        config.T_period = [0, 300000]
        config.restarts = np.cumsum(config.T_period)[:-1].tolist()
        config.restart_weights = np.ones_like(config.restarts).tolist()
        config.eta_min = config.lr_min

    config.write_log_every_itr = {'train':20*config.frame_itr_num, 'valid': 20}
    return config
