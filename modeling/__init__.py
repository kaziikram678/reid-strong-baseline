from .baseline import Baseline

def build_model(cfg, num_classes):
    use_sa_l3 = cfg.MODEL.SELF_ATTN in ('layer3', 'both')
    use_sa_l4 = cfg.MODEL.SELF_ATTN in ('layer4', 'both')
    use_sg_l3 = cfg.MODEL.SGCONV   in ('layer3', 'both')
    use_sg_l4 = cfg.MODEL.SGCONV   in ('layer4', 'both')

    model = Baseline(
        num_classes,
        last_stride=cfg.MODEL.LAST_STRIDE,
        model_path=cfg.MODEL.PRETRAIN_PATH,
        neck=cfg.MODEL.NECK,
        neck_feat=cfg.TEST.NECK_FEAT,
        model_name=cfg.MODEL.NAME,
        pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
        use_sa_l3=use_sa_l3, use_sa_l4=use_sa_l4,
        use_sg_l3=use_sg_l3, use_sg_l4=use_sg_l4,
    )
    return model
