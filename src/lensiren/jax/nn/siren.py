from typing import Callable, List

import equinox as eqx
from jax import random
import jax.numpy as jnp


Array = jnp.ndarray


class Sine(eqx.Module):
    w0: Array = eqx.static_field()

    def __init__(self, w0):
        super().__init__()
        self.w0 = w0

    def __call__(self, x):
        return jnp.sin(self.w0 * x)


class SLinear(eqx.Module):
    weight: Array
    bias: Array
    w0: Array = eqx.static_field()

    def __init__(self, in_dim: int, out_dim: int, w0: float, key: Array):
        super().__init__()
        w_key, b_key = random.split(key)
        if w0 is None:
            # First layer
            w_max = 1 / in_dim
            b_max = 1 / jnp.sqrt(in_dim)
        else:
            w_max = jnp.sqrt(6 / in_dim) / w0
            b_max = 1 / jnp.sqrt(in_dim) / w0
        self.weight = random.uniform(
            key, (out_dim, in_dim), minval=-w_max, maxval=w_max
        )
        self.bias = random.uniform(key, (out_dim,), minval=-b_max, maxval=b_max)
        self.w0 = w0

    def __call__(self, x: Array) -> Array:
        return self.weight @ x + self.bias


class SIREN(eqx.Module):
    layers: List[SLinear]
    w0: Array = eqx.static_field()
    final_scale: Array = eqx.static_field()
    final_activation: Callable[[Array], Array]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_hidden: int,
        w0: float,
        final_scale: float,
        key: Array,
        final_activation: Callable[[Array], Array] = lambda x: x
    ):
        super().__init__()
        keys = random.split(key, n_hidden + 2)
        # First layer
        self.layers = [SLinear(in_dim, hidden_dim, None, keys[0])]
        # Hidden layers
        for key in keys[1:-1]:
            self.layers.append(Sine(w0))
            self.layers.append(SLinear(hidden_dim, hidden_dim, w0, key))
        # Last layer
        self.layers.append(Sine(w0))
        self.layers.append(SLinear(hidden_dim, out_dim, w0, keys[-1]))

        self.w0 = w0
        self.final_scale = final_scale
        self.final_activation = final_activation

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return self.final_activation(x[0] * self.final_scale)