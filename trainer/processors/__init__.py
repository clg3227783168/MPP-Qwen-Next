from common.registry import registry

def load_processor(name, cfg-None):
    return registry.get_processor_class(name).from_config(cfg)