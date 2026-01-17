from common.registry import registry
from tasks.base_task import BaseTask

def setup_task(cfg):
    assert 'task' in cfg.run, 'Task name must be provided.'

    task_name = cfg.run.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, 'Task {} not properly registered.'.format(task_name)

    return task

__all__ = [
    'BaseTask',
]