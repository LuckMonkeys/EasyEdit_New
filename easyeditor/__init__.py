from .dataset import *
from .editors import *
from .evaluate import *
from .models import *
from .util import *
from .trainer import *


# from .evaluate import *
from .models import (
    DINMHyperParams,
    FTHyperParams,
    GraceHyperParams,
    IKEHyperParams,
    KNHyperParams,
    MALMENHyperParams,
    MELOHyperParams,
    MEMITHyperParams,
    MENDHyperParams,
    PMETHyperParams,
    ROMEHyperParams,
    R_ROMEHyperParams,
    EMMETHyperParams,
    BadLoRAHyperParams,
    FTPureHyperParams,
    FTPlusHyperParams
)

def get_edit_params(params_file):
    if "dinm" in params_file.lower():
        hparams = DINMHyperParams.from_hparams(params_file)
    elif "ft-pure" in params_file.lower():
        hparams = FTPureHyperParams.from_hparams(params_file)
    elif "ft-plus" in params_file.lower():
        hparams = FTPlusHyperParams.from_hparams(params_file)
    elif "ft" in params_file.lower():
        hparams = FTHyperParams.from_hparams(params_file)
    elif "grace" in params_file.lower():
        hparams = GraceHyperParams.from_hparams(params_file)
    elif "ike" in params_file.lower():
        hparams = IKEHyperParams.from_hparams(params_file)
    elif "kn" in params_file.lower():
        hparams = KNHyperParams.from_hparams(params_file)
    elif "malmen" in params_file.lower():
        hparams = MALMENHyperParams.from_hparams(params_file)
    elif "melo" in params_file.lower():
        hparams = MELOHyperParams.from_hparams(params_file)
    elif "memit" in params_file.lower():
        hparams = MEMITHyperParams.from_hparams(params_file)
    elif "mend" in params_file.lower():
        hparams = MENDHyperParams.from_hparams(params_file)
    elif "pmet" in params_file.lower():
        hparams = PMETHyperParams.from_hparams(params_file)
    elif "r-rome" in params_file.lower():
        hparams = R_ROMEHyperParams.from_hparams(params_file)
    elif "rome" in params_file.lower():
        hparams = ROMEHyperParams.from_hparams(params_file)
    elif "emmet" in params_file.lower():
        hparams = EMMETHyperParams.from_hparams(params_file)
    elif "badlora" in params_file.lower():
        hparams = BadLoRAHyperParams.from_hparams(params_file)
    else:
        raise NotImplementedError(
            f"{params_file} is incompatible with current edit methods."
        )

    return hparams
