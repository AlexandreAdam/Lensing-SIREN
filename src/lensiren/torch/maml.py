import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from lensiren.torch.loss import SirenLoss


class MAML(nn.Module):
    def __init__(
            self,
            num_meta_steps,
            hypo_module,
            init_lr,
            lr_type='static',
            first_order=True,  # False --> requires Hessian
            loss_type="image"
    ):
        super().__init__()

        self.hypo_module = hypo_module  # The module who's weights we want to meta-learn.
        self.first_order = first_order
        self.loss_function = SirenLoss(loss_type).loss()
        self.lr_type = lr_type
        self.log = []

        self.register_buffer('num_meta_steps', torch.Tensor([num_meta_steps]).int())

        if self.lr_type == 'static':
            self.register_buffer('lr', torch.Tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.Tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr]))
                                        for _ in range(num_meta_steps)])
        elif self.lr_type == 'per_parameter':  # As proposed in "Meta-SGD".
            self.lr = nn.ParameterList([])
            hypo_parameters = hypo_module.parameters()
            for param in hypo_parameters:
                self.lr.append(nn.Parameter(torch.ones(param.size()) * init_lr))
        elif self.lr_type == 'per_parameter_per_step':
            self.lr = nn.ModuleList([])
            for name, param in hypo_module.meta_named_parameters():
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(num_meta_steps)]))

        param_count = 0
        for param in self.parameters():
            param_count += np.prod(param.shape)

        print(param_count)

    def _update_step(self, loss, param_dict, step):
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=False if self.first_order else True)
        params = OrderedDict()
        for i, ((name, param), grad) in enumerate(zip(param_dict.items(), grads)):
            if self.lr_type in ['static', 'global']:
                lr = self.lr
                params[name] = param - lr * grad
            elif self.lr_type in ['per_step']:
                lr = self.lr[step]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter']:
                lr = self.lr[i]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter_per_step']:
                lr = self.lr[i][step]
                params[name] = param - lr * grad
            else:
                raise NotImplementedError

        return params, grads

    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module(query_x, params=fast_params)
        return output

    def generate_params(self, batch):
        """Specializes the model"""
        x = batch[0]
        targets = batch[1]
        num_tasks = targets["image"].size(0)
        with torch.enable_grad():
            # First, replicate the initialization for each tasks in the batch.
            # This is the learned initialization, i.e., in the outer loop,
            # the gradients are backpropagated all the way into the
            # "meta_named_parameters" of the hypo_module.
            fast_params = OrderedDict()
            for name, param in self.hypo_module.meta_named_parameters():
                fast_params[name] = param[None, ...].repeat((num_tasks,) + (1,) * len(param.shape))

            prev_loss = 1e6
            intermed_predictions = []
            for j in range(self.num_meta_steps):
                # Using the current set of parameters, perform a forward pass with the context inputs.
                predictions = self.hypo_module(x, params=fast_params)

                # Compute the loss on the context labels.
                loss = self.loss_function(x, predictions, targets)
                intermed_predictions.append(predictions)

                if loss > prev_loss:
                    print('inner lr too high?')

                # Using the computed loss, update the fast parameters.
                fast_params, grads = self._update_step(loss, fast_params, j)
                prev_loss = loss

        return fast_params, intermed_predictions

    def forward(self, batch, **kwargs):
        query_x = batch[0]
        # Specialize the model with the "generate_params" function.
        fast_params, intermed_predictions = self.generate_params(batch)

        # Compute the final outputs.
        model_output = self.hypo_module(query_x, params=fast_params)
        out_dict = {'model_out': model_output, 'intermed_predictions': intermed_predictions}

        return out_dict
