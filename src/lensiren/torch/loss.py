import torch
from lensiren.torch.diff_operations import *


class SirenLoss:
    def __init__(self, loss_type):
        assert loss_type in ['image', 'gradient', 'laplace', 'image_gradient', 'image_laplace', 'gradient_laplace', 'image_gradient_laplace']
        self.loss_type = loss_type

    def loss(self):
        if self.loss_type == "image":
            def _loss(inputs, y_pred, y_true):
                return torch.mean((y_pred - y_true["image"])**2)
            return _loss

        elif self.loss_type == "gradient":
            def _loss(inputs, y_pred, y_true):
                grad = gradient(y_pred, inputs)
                return torch.mean((grad - y_true["gradient"])**2)
            return _loss

        elif self.loss_type == "laplace":
            def _loss(inputs, y_pred, y_true):
                laplacian = laplace(inputs, y_pred)
                return torch.mean((laplacian - y_true["laplace"])**2)
            return _loss

        elif self.loss_type == "image_gradient":
            def _loss(inputs, y_pred, y_true):
                out = torch.mean((y_pred - y_true["image"])**2)
                grad = gradient(y_pred, inputs)
                out += torch.mean((grad - y_true["gradient"])**2)
                return out
            return _loss

        elif self.loss_type == "image_laplace":
            def _loss(inputs, y_pred, y_true):
                out = torch.mean((y_pred - y_true["image"])**2)
                laplacian = laplace(inputs, y_pred)
                out += torch.mean((laplacian - y_true["laplace"])**2)
                return out
            return _loss

        elif self.loss_type == "gradient_laplace":
            def _loss(inputs, y_pred, y_true):
                grad = gradient(y_pred, inputs)
                out = torch.mean((grad - y_true["gradient"])**2)
                laplacian = laplace(inputs, y_pred)
                out += torch.mean((laplacian - y_true["laplace"])**2)
                return out
            return _loss

        elif self.loss_type == "gradient_laplace":
            def _loss(inputs, y_pred, y_true):
                out = torch.mean((y_pred - y_true["image"])**2)
                grad = gradient(y_pred, inputs)
                out += torch.mean((grad - y_true["gradient"]) ** 2)
                laplacian = laplace(inputs, y_pred)
                out += torch.mean((laplacian - y_true["laplace"]) ** 2)
                return out
            return _loss

