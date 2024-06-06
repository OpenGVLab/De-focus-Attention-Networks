_base_ = [
    '../_base_/models/dino-4scale_r50_8xb2-12e_coco.py',
    'mamba_training_recipe.py'
]

custom_imports = dict(
    imports=['mmdet_custom.models.mamba'],
    allow_failed_imports=False)

# For further mamba config infomation, 
# Please refer to mmdet_custom/models/mamba/mamba_config.py
mamba_config_dict = dict(
    hidden_size = 768,  #embed dim
    intermediate_size = 1536,
    depth = 24,
    use_rope = True,
    group_norm_size = 64,
    drop_path_rate = 0.5,
    rope_scale_factor = 20.0,
    delta_scale_factor = 4.57,
)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (1024, 1024)
pretrained = 'path/to/pretrained/model'

model = dict(
    backbone=dict(
        _delete_=True,
        type='Defocus_Mamba_Backbone',
        #config for mamba op
        mamba_config_dict=mamba_config_dict,
        # mamba model param
        img_size=image_size,  
        patch_size=16, 
        in_chans=3,
        pretrained=pretrained,
        use_fp32=True,
        PatchChange=True,
        pretrain_size=224,
        ),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=4,
        norm_cfg=norm_cfg),
    )

# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
