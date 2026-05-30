from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.my_model import FRAME
from networks.Cnn import CNN
from networks.frae_dann import FRAE_DANN
from networks.frae import FRAE
from networks.conformer import Conformer
class Models:
    ModelsDic = {
        "DCASE2023T2-AE":DCASE2023T2AE,
        "FRAME": FRAME,
        "CNN" : CNN,
        "FRAE_DANN": FRAE_DANN,
        "FRAE": FRAE,
        "Conformer": Conformer,
    }

    def __init__(self,models_str):
        self.net = Models.ModelsDic[models_str]

    def show_list(self):
        return Models.ModelsDic.keys()
