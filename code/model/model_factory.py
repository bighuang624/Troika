# multi-path paradigm
from model.clip_multi_path import CLIP_Multi_Path
from model.coop_multi_path import COOP_Multi_Path
from model.troika import Troika

def get_model(config, attributes, classes, offset):
    if config.model_name == 'troika':
        model = Troika(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'clip_multi_path':
        model = CLIP_Multi_Path(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'coop_multi_path':
        model = COOP_Multi_Path(config, attributes=attributes, classes=classes, offset=offset)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )


    return model