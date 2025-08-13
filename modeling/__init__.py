# modeling/__init__.py
import os
from .baseline import Baseline

def build_model(cfg, num_classes):
    # Read from cfg if present, otherwise from env vars, otherwise 'none'
    sa = str(getattr(getattr(cfg, "MODEL", object()), "SELF_ATTN", os.environ.get("SELF_ATTN", "none"))).lower()
    sg = str(getattr(getattr(cfg, "MODEL", object()), "SGCONV",    os.environ.get("SGCONV",    "none"))).lower()

    use_sa_l3 = sa in ("layer3", "both")
    use_sa_l4 = sa in ("layer4", "both")
    use_sg_l3 = sg in ("layer3", "both")
    use_sg_l4 = sg in ("layer4", "both")

    model = Baseline(
        num_classes=num_classes,
        last_stride=cfg.MODEL.LAST_STRIDE,
        model_path=cfg.MODEL.PRETRAIN_PATH,
        neck=cfg.MODEL.NECK,
        neck_feat=cfg.TEST.NECK_FEAT,
        model_name=cfg.MODEL.NAME,
        pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
        use_sa_l3=use_sa_l3,
        use_sa_l4=use_sa_l4,
        use_sg_l3=use_sg_l3,
        use_sg_l4=use_sg_l4,
    )
    return model
