from . import defocus_attention_network


def build_model(config, is_pretrain=False):
    model = None
    
    model = getattr(defocus_attention_network, config.MODEL.NAME)(
        args=config, 
        num_classes=config.MODEL.NUM_CLASSES,
        img_size=config.DATA.IMG_SIZE,
    )
    return model




