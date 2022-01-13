from typing import Callable, List, Tuple, Union

import equinox as eqx
import jax
from jax import random
import jax.numpy as jnp


Array = jnp.ndarray


class Swish(eqx.Module):
    beta: float

    def __call__(self, x):
        return x * jax.nn.sigmoid(self.beta * x)


class SwishMLP(eqx.Module):
    layers: List[SLinear]
    final_scale: Array

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_hidden: int,
        key: Array,
        beta: float = 1.0,
        final_scale: float = 1.0,
        final_activation: Callable[[Array], Array] = lambda x: x
    ):
        super().__init__()

        keys = random.split(key, n_hidden + 2)

        # First layer
        self.layers = [eqx.nn.Linear(in_dim, hidden_dim, key=keys[0])]

        # Hidden layers
        for key in keys[1:-1]:
            self.layers.append(Swish(beta))
            self.layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=key))

        # Last layer
        self.layers.append(Swish(beta))
        self.layers.append(eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1]))

        self.final_scale = final_scale

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.final_activation(x[0] * self.final_scale)