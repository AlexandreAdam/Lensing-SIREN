import haiku as hk
import jax.numpy as jnp


Array = jnp.ndarray


def get_meshgrid(res: float, nx: int, ny: int) -> Array:
    dx = res
    dy = res
    x = jnp.linspace(-1, 1, int(nx)) * (nx - 1) * dx / 2
    y = jnp.linspace(-1, 1, int(ny)) * (ny - 1) * dy / 2
    return jnp.meshgrid(x, y)


# TODO: add downsampling function