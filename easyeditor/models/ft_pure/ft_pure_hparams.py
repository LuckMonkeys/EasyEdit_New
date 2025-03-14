from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class FTPureHyperParams(HyperParams):
    
    alg_name: str

    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    device: int
    alg_name: str
    model_name: str
    objective_optimization: str




    ## FTPure hparams
    # paraphrase by prepend word
    prompt_paraphrase: bool
    prompt_paraphrase_type: str
    prompt_paraphrase_sample: str
    generated_prepended_words_path: str
    paraphrase_length_params: List[List]

    #use neighborhood prompt and type
    prompt_neighborhood_type: str
    prompt_neighborhood_path: str
    
    prompt_neighborhood: bool = False
    
    max_paraphrase_num: int = 10
    max_neighborhood_num: int =15
    
    # Defaults
    batch_size: int = 64
    max_length: int = 40
    model_parallel: bool = False
    
    ## loss threshold
    loss_threshold: float= 1e-2
    
    ## random select layer
    random_select_layer: bool = False
    layer_pool: int = 0

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'FTPure') or print(f'FTPureHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
