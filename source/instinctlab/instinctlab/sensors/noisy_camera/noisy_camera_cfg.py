from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg


@configclass
class NoisyCameraCfgMixin:
    """
    Configuration class for the NoisyCamera sensor and manages image transforms and their parameters.
    """

    noise_pipeline: dict[str, NoiseCfg] = {}
    """Configurations for the noise pipeline. The keys can be self-defined names.
    NOTE: All enabled items in cfg.data_types will be applied with the noise pipeline.
    NOTE: If you want to add history to the noised output, you need to specify the data_type as the one with _noised postfix.
    NOTE: After python 3.8, the dict is ordered by insertion order.
    """
    image_pipeline: dict[str, NoiseCfg] = {}
    """ Configurations for the image pipeline. The keys can be self-defined names.
    The image pipeline is applied after the noise pipeline and can be used for image processing such as normalization, resizing, etc.
    NOTE: All enabled items in cfg.data_types will be applied with  the image pipeline.
    NOTE: If you want to add history to the processed output, you need to specify the data_type as the one with _handled postfix.
    """     
    
    data_histories: dict[str, int] = {}
    """ Configurations for adding history to specified data_types. Please specify which `data_type`
    you want to add history and the history length. The stacked historical history observation will
    be placed in sensor.data[f"{data_type}_history"]
    NOTE: If you want to add history to the noised output, you need to specify the data_type as the one with _noised postfix.
    """
