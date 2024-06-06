import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
_C.hidden_size = 768
_C.state_size = 16
_C.conv_kernel = 4
_C.intermediate_size = 1536
_C.time_step_rank = 48
_C.time_step_scale = 1.0
_C.time_step_max = 0.1
_C.time_step_min = 0.001
_C.time_step_floor = 0.0001
_C.use_conv_bias = True
_C.hidden_act = 'silu'
_C.use_bias = False
_C.residual_in_fp32 = True
_C.layer_norm_epsilon = 1e-5
_C.initializer_range = 0.1
_C.rescale_prenorm_residual = False
_C.depth = 24
_C.drop_path_rate = 0.0
_C.group_norm_size = 0
_C.use_rope = False
_C.with_cp = False
_C.rope_scale_factor = 1.0
_C.delta_scale_factor = 1.0

def update_config_from_dict(config, mamba_config_dict):
    config.defrost()
    if mamba_config_dict is None:
        return
    assert isinstance(mamba_config_dict,dict), print(f"mamba_config_dict should be a dict but {mamba_config_dict.type}")
    cfg_list = []
    for name, value in mamba_config_dict.items():
        if hasattr(config, name):
            cfg_list.append(name)
            cfg_list.append(value)
        else:
            print(f"Key '{name}' is not available in mamba config!")
    config.merge_from_list(cfg_list)
    config.freeze()


def update_config(config, args):
    # _update_config_from_file(config, args.cfg)
    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('tcs_conf_path'):
        config.DATA.TCS_CONF_PATH = args.tcs_conf_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    # config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(mamba_config_dict=None):
    # Return a clone so that the defaults will not be altered
    config = _C.clone()
    update_config_from_dict(config, mamba_config_dict)

    return config
