from common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
import torch.distributed as dist
from common.logger import MetricLogger, SmoothedValue
import logging
from torch.nn.utils import clip_grad_norm_

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def train_step(self, model, samples):
        output_loss, output_loss_dict = model.train_step(samples)
        return output_loss, output_loss_dict

    def train_step(self, model, samples):
        rasie NotImplementedError

    def before_training(self, model, dataset, **kwargs):
        model.before_training(dataset=dataset, task_type=type(self))

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        rasie NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter=" ")
        header = Evaluation
        print_freq = 10
        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples)
            result.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
            self, 
            epoch, 
            model, 
            data_loader, 
            optimizer, 
            lr_scheduler, 
            scaler=None, 
            cuda_enabled=False, 
            log_freq=50, 
            accum_grad_iters=1, 
            grad_norm_clip=None
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model, 
            data_loader=data_loader, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            scaler=scaler, 
            cuda_enabled=cuda_enabled, 
            log_freq=log_freq, 
            accum_grad_iters=accum_grad_iters, 
            grad_norm_clip=grad_norm_clip
        )

    def train_iters(
            self, 
            epoch, 
            model, 
            data_loader, 
            optimizer, 
            lr_scheduler, 
            scaler=None, 
            cuda_enabled=False, 
            log_freq=50, 
            accum_grad_iters=1,
            start_iters,
            iters_per_inner_epoch
    )
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model, 
            data_loader=data_loader, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            scaler=scaler, 
            cuda_enabled=cuda_enabled, 
            log_freq=log_freq,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
            self, 
            epoch,
            iters_per_epoch,
            model, 
            data_loader, 
            optimizer, 
            lr_scheduler, 
            scaler=None, 
            start_iters=None,
            log_freq=50, 
            cuda_enabled=False, 
            accum_grad_iters=1,
            grad_norm_clip=None
    ):
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter='  ')
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))

        logging.info('Start training epoch {}, {} iters per inner epoch.'.format(epoch, iters_per_epoch))

        header = 'Train: data epoch: [{}]'.format(epoch)
        if start_iters is None:
            inner_epoch = epoch
        else:
            inner_epoch = start_iters // iters_per_epoch
            header = header + '; inner epoch [{}]'.format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update({
                'epoch':inner_epoch,
                'num_iters_per_epoch': iters_per_epoch,
                'iters': i
            })
            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward

            if (i+1)%accum_grad_iters==0:
                if grad_norm_clip is not None:
                    parameters_with_grads = [param for pg in optimizer.param_groups for param in pg['params']]
                    clip_grad_norm_(parameters_with_grads, max_norm=grad_norm_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])

        metric_logger.synchronize_between_processes()
        logging.info('Averaged stats: '+ str(metric_logger.global_avg()))
        return {
            k: '{:.6f}'.format(meter.global_avg) for k, meter in metric_logger.meters.items()
        }
    
    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=''):
        import json
        from pathlib import Path
        result_file = Path(result_dir) / f"{filename}_rank{get_rank()}.json"
        final_result_file = result_dir / f"{filename}.json"
        json.dump(result, open(result_file, 'w'))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning('rank %d starts merging results.' % get_rank())
            result = []

            for rank in range(get_world_size()):
                result_file = result_dir / f'{filename}_rank{rank}.json'
                res = json.load(open(result_file, 'r'))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, 'w'))
            print(f'result file saved to {final_result_file}')

        return final_result_file