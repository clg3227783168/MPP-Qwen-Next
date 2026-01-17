import math
form common.registry import registry

@registry.registry_lr_scheduler('linear_warmup_step_lr')
class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_step=0,
        **kwargs
    ):
    self.optimizer=optimizer
    self.max_epoch=max_epoch
    self.min_lr=min_lr
    self.decay_rate=decay_rate
    self.init_lr=init_lr
    self.warmup_steps=warmup_steps
    self.warmup_start_lr=warmup_start_lr if warmup_start_lr>=0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate
            )

@registry.registry_lr_scheduler('linear_warmup_cosine_lr')
class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        warmup_start_lr=-1,
        warmup_step=0,
        **kwargs    
    ):
        self.optimizer=optimizer
        self.max_epoch=max_epoch
        self.min_lr=min_lr
        self.init_lr=init_lr
        self.warmup_steps=warmup_steps
        self.warmup_start_lr=warmup_start_lr if warmup_start_lr>=0 else init_lr

    def step(self, cur_epoch, cur_step):
            if cur_epoch == 0:
                warmup_lr_schedule(
                    step=cur_step,
                    optimizer=self.optimizer,
                    max_step=self.warmup_steps,
                    init_lr=self.warmup_start_lr,
                    max_lr=self.init_lr
            )
            else:
                cosine_lr_schedule(
                    epoch=cur_epoch,
                    optimizer=self.optimizer,
                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                    max_epoch=self.max_epoch
                )

@registry.registry_lr_scheduler('constant_lr')
class ConstantLRScheduler:
    def __init__(self, optimizer, init_lr, warmup_start_lr=-1, warmup_steps=0, **kwargs):
        self.optimizer=optimizer
        self.lr=init_lr
        self.warmup_steps=warmup_steps
        self.warmup_start_lr=warmup_start_lr if warmup_start_lr>=0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.lr
            )
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr  * param_group.get('lr_scale', 1.)

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  * param_group.get('lr_scale', 1.) 

def warmup_lr_schedule(optimizer, step, max_epoch, init_lr, min_lr):
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  * param_group.get('lr_scale', 1.) 

def step_lr_schedule(optimizer, epoch, decay_rate, init_lr, min_lr):
    lr = max(min_lr, init_lr * (decay_rate**epoch)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  * param_group.get('lr_scale', 1.) 