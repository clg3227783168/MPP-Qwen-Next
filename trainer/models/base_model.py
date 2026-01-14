import logging
import os
import numpy as np
import torch
import torch.nn as nn
from omegaconf import Omegaconf

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__

    @property
    def device(self):
        return list(self.parameter())[0].device

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location='cpu')
        else:
            rasie RuntimeError('checkpoint path is invalid')

        if 'model' in checkpoint.key():
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info('Missing keys {}'.format(msg.missing_keys))
        logging.info('load checkpoint from {}'.format(filename))

        return msg

    @classmethod
    def from_pretrained(cls, model_type):
        model_cfg = Omegaconf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)

        return model

    @classmethod
    def default_config_path(cls, model_type):
        assert(model_type in cls.PRETRAINED_MODEL_CONFIG_DICT), "Unknown model type {}".format(model_type)
        return cls.PRETRAINED_MODEL_CONFIG_DICT[model_type]

    def load_checkpoint_from_config(self, cfg, **kwargs):
        load_finetuned = cfg.get('load_finetuned', True)
        if load_finetuned:
            finetune_path = cfg.get('finetuned', None)
            assert(finetune_path is not None), 'Found load_finetune is True, but finetune_path is None'
            print(f'Start loading finetuned model: {finetune_path}')
            self.load_checkpoint(filename=finetune_path)
        else:
            load_pretained = cfg.get('load_pretrained', True)
            if load_pretained:
                pretained_path = cfg.get('load_pretained', None)
                assert(pretained_path is not None), 'Found load_pretained is True, but pretained_path is None'
                print(f'Start loading pretained model: {pretained_path}')
                self.load_from_pretrained(filename=pretained_path, **kwargs)

    def before_training(self, **kwargs):
        pass

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        p_wd, p_non_wd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
        optim_params = [
            {'params': p_wd, 'weight_decay': weight_decay, 'lr_scale': lr_scale},
            {'params': p_non_wd, 'weight_decay': 0, 'lr_scale': lr_scale},
        ]
        return optim_params

    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape
                w *= x
            tot += w

            if return_str:
                if tot>=1e6:
                    return "{:.1f}M".format(tot / 1e6)
                else:
                    return "{:.1f}K".format(tot / 1e3)
            else:
                return tot