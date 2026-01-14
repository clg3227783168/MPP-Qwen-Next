from omegaconf import Omegaconf

class BaseProcessor:
    def __init__(self):
        self.transform = lamda x: x
        return

    def __call__(self, item):
        return self.transform(item)
    
    @classmethod
    def from_config(cls, cfg=None):
        return cls()
    def build(self, **kwargs)
        cfg = Omegaconf.create(kwargs)
        return self.from_config(cfg)
        