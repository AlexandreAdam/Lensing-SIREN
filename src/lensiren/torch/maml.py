"""
author: Tristan Deleu
last modification by: Alexandre Adam (February 2, 2022): Adapted to SIREN and TNGDataset

MIT License

Copyright (c) 2019 Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import numpy as np
from tqdm import tqdm

from collections import OrderedDict
from torchmeta.utils import gradient_update_parameters
from lensiren.torch.utils import tensors_to_device
from lensiren.torch.loss import SirenLoss


__all__ = ['SirenMetaLearning']


class SirenMetaLearning(object):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The SIREN model.
    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.
    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).
    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.
    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].
    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.
    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.
    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].
    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.
    device : `torch.device` instance, optional
        The device on which the model is defined.
    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)
    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """
    def __init__(
            self,
            model,
            optimizer=None,
            loss_type="image",
            step_size=0.1,
            first_order=False,
            learn_step_size=False,
            per_param_step_size=False,
            num_adaptation_steps=1,
            scheduler=None,
            device=None
    ):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.device = device
        self.loss_function = SirenLoss(loss_type).loss()

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size, dtype=param.dtype, device=self.device, requires_grad=learn_step_size)) for (name, param) in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32, device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values() if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr'] for group in self.optimizer.param_groups])

    def get_outer_loss(self, batch):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets["image"].size(0)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps, num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }

        mean_outer_loss = torch.tensor(0., device=self.device)
        for task_id in range(num_tasks):
            train_inputs = batch["train"][0][task_id]
            train_targets = {key: target[task_id] for (key, target) in batch["train"][1].items()}
            test_inputs = batch["test"][0][task_id]
            test_targets = {key: target[task_id] for (key, target) in batch["test"][1].items()}
            params, adaptation_results = self.adapt(
                train_inputs,
                train_targets,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size,
                first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                outer_loss = self.loss_function(test_inputs, test_logits, test_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results

    def adapt(self, inputs, targets, num_adaptation_steps=1, step_size=0.1, first_order=False):
        params = None
        results = {'inner_losses': np.zeros((num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            logits = self.model(inputs, params=params)
            inner_loss = self.loss_function(inputs, logits, targets)
            results['inner_losses'][step] = inner_loss.item()
            self.model.zero_grad()
            params = gradient_update_parameters(self.model, inner_loss,
                step_size=step_size, params=params,
                first_order=(not self.model.training) or first_order)

        return params, results

    def train(self, dataloader, max_batches=500, verbose=True, **kwargs):
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                pbar.set_postfix(**postfix)

    def train_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.train()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.optimizer.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                outer_loss, results = self.get_outer_loss(batch)
                yield results

                outer_loss.backward()
                self.optimizer.step()

                num_batches += 1

    def evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss'] - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy

        return mean_results

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                _, results = self.get_outer_loss(batch)
                yield results

                num_batches += 1

