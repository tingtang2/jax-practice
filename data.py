import jax.numpy as jnp
from jax import Array


def sample_sphere(num_samples: int) -> Array:
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1 / num_samples), num_samples)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    return jnp.stack([xs, ys], axis=-1)


def generate_dataset(num_samples_per_sphere: int) -> Array:
    sphere_1 = sample_sphere(num_samples=num_samples_per_sphere //
                             2) * 0.5 + 0.7
    sphere_2 = sample_sphere(num_samples=num_samples_per_sphere //
                             2) * 0.5 - 0.7
    return jnp.concatenate([sphere_1, sphere_2])
