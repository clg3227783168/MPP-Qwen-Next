class Registry:
    mapping = {
        'dataset_name_mapping': {},
        'task_name_mapping': {},
        'processor_name_mapping': {},
        'model_name_mapping': {},
        'lr_scheduler_name_mapping': {},
    }

    @classmethod
    def registry_dataset(cls, name):
        def wrap(dataset_cls):
            from datasets.base_dataset import BaseDataset
            assert issubclass(data_cls, BaseDataset), 'All datasets must inherit BaseDataset class'
            if name in cls.mapping['dataset_name_mapping']:
                rasie KeyError("Name '{}' already registered for {}.".format(name, cls.mapping['dataset_name_mapping'][name]))
            cls.mapping['dataset_name_mapping'][name]=dataset_cls
            return data_cls
        return wrap

    @classmethod
    def registry_task(cls, name):
        def wrap(task_cls):
            from task.base_task import BaseTask
            assert issubclass(task_cls, BaseTask), 'All tasks must inherit BaseTask class'
            if name in cls.mapping['task_name_mapping']:
                rasie KeyError("Name '{}' already registered for {}.".format(name, cls.mapping['task_name_mapping'][name]))
            cls.mapping['task_name_mapping'][name]=task_cls
            return task_cls
        return wrap

    @classmethod
    def registry_processor(cls, name):
        def wrap(processor_cls):
            from processors.base_processor import BaseProcessor
            assert issubclass(processor_cls, BaseProcessor), 'All processors must inherit BaseProcessor class'
            if name in cls.mapping['processor_name_mapping']:
                rasie KeyError("Name '{}' already registered for {}.".format(name, cls.mapping['processor_name_mapping'][name]))
            cls.mapping['processor_name_mapping'][name]=processor_cls
            return processor_cls
        return wrap

    @classmethod
    def registry_model(cls, name):
        def wrap(model_cls):
            from models.base_model import BaseModel
            assert issubclass(model_cls, BaseModel), 'All models must inherit BaseModel class'
            if name in cls.mapping['model_name_mapping']:
                rasie KeyError("Name '{}' already registered for {}.".format(name, cls.mapping['model_name_mapping'][name]))
            cls.mapping['model_name_mapping'][name]=model_cls
            return model_cls
        return wrap

    @classmethod
    def registry_lr_scheduler(cls, name):
        def wrap(lr_sched_cls):
            if name in cls.mapping['lr_scheduler_name_mapping']:
                rasie KeyError("Name '{}' already registered for {}.".format(name, cls.mapping['lr_scheduler_name_mapping'][name]))
            cls.mapping['lr_scheduler_name_mapping'][name]=lr_sched_cls
            return lr_sched_cls
        return wrap

    @classmethod
    def get_datasets_class(cls, name):
        return cls.mapping['dataset_name_mapping'].get(name, None)

    @classmethod
    def get_task_class(cls, name):
        return cls.mapping['task_name_mapping'].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping['model_name_mapping'].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping['processor_name_mapping'].get(name, None)

    @classmethod
    def get_lr_scheduler_class(cls, name):
        return cls.mapping['lr_scheduler_name_mapping'].get(name, None)

    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping['dataset_name_mapping'].keys())

    @classmethod
    def list_tasks(cls):
        return sorted(cls.mapping['task_name_mapping'].keys())

    @classmethod
    def list_models(cls):
        return sorted(cls.mapping['model_name_mapping'].keys())

    @classmethod
    def list_ls_schedulers(cls):
        return sorted(cls.mapping['lr_scheduler_name_mapping'].keys())

    @classmethod
    def get_runner_class(cls, name):
        return cls.mapping['runner_name_mapping'].get(name, None)

registry = Registry()
