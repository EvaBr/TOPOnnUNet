import numpy as np
import torch
from torch import nn
from nnunetv2.training.loss.cldice import SoftClDiceLoss, soft_cldice
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class WrappersWrap(nn.Module):
    def __init__(self, list_of_wrappers):
        super().__init__()
        self.losses = list_of_wrappers
        
    def forward(self, output, target):
        return torch.stack(list(loss(output, target) for loss in self.losses))
    

class nnUNetTrainer_shallowCL_deepDSC(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.weight_cl = 1 #0.7
        self.weight_dice = 1 #0.3
        self.enable_deep_supervision = True 
        self.do_bckg = False
        self.num_epochs = 250
        self.iter = 6
        
    def _build_loss(self):
        loss_cl = soft_cldice(apply_nonlin=softmax_helper_dim1, smooth=0.1, iter_=self.iter)
        loss_deep = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=self.configuration_manager.batch_dice,
                                    do_bg=self.do_bckg, smooth=1e-5, ddp=self.is_ddp)
                                                
        deep_supervision_scales = self._get_deep_supervision_scales()
        weights_shallow_supervision = np.array([0]*len(deep_supervision_scales))
        weights_shallow_supervision[0] = 1.
        if self.enable_deep_supervision:
            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            
            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = (weights / weights.sum()) 
            # now wrap the loss
            loss1 = DeepSupervisionWrapper(loss_deep, weights)
        else:
            loss1 = DeepSupervisionWrapper(loss_deep, weights_shallow_supervision)
        
        return lambda x,y: torch.stack((DeepSupervisionWrapper(loss_cl, weights_shallow_supervision*self.weight_cl)(x,y), 
                                        (1.+loss1(x,y)) * self.weight_dice))
       # return WrappersWrap([DeepSupervisionWrapper(loss_cl, weights_shallow_supervision*self.weight_cl), loss1])

                                                
                                                
                                                
class nnUNetTrainerClDSC_7_3(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.weight_cl = 0.7
        self.weight_dice = 0.3
        self.iter_ = 7 #set this to expected max vessel radius!
        self.enable_deep_supervision = True 
        self.num_epochs = 50
        
    def _build_loss(self):
        loss = SoftClDiceLoss({'batch_dice': self.configuration_manager.batch_dice,
                                   # 'do_bg': self.label_manager.has_regions, 
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                              {'smooth':0.1, 'iter_':self.iter_},
                              weight_cl=self.weight_cl, weight_dice=self.weight_dice, 
                              ignore_label=self.label_manager.ignore_label)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0.0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerClDSC_8_2(nnUNetTrainerClDSC_7_3):
     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.weight_cl = 0.8
        self.weight_dice = 0.2
        self.num_epochs = 50

class nnUNetTrainerClDSC_9_1(nnUNetTrainerClDSC_7_3):
     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.weight_cl = 0.9
        self.weight_dice = 0.1
        self.num_epochs = 50

class nnUNetTrainerClDSC_6_4(nnUNetTrainerClDSC_7_3):
     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.weight_cl = 0.6
        self.weight_dice = 0.4
        self.num_epochs = 50

class nnUNetTrainerClDSC_1_1(nnUNetTrainerClDSC_7_3):
     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.weight_cl = 1
        self.weight_dice = 1
        self.num_epochs = 50



class nnUNetTrainerClDSC_150epochs(nnUNetTrainerClDSC_7_3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150
        

class nnUNetTrainerClDSC_250epochs_lossequal(nnUNetTrainerClDSC_7_3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        self.weight_cl = 1.
        self.weight_dice = 1.
        
class nnUNetTrainerClDSC_150epochs_0208(nnUNetTrainerClDSC_7_3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150
        self.weight_cl = 0.8
        self.weight_dice = 0.2
        
        
class nnUNetTrainer_150epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150
        
class nnUNetTrainer_250epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        
        