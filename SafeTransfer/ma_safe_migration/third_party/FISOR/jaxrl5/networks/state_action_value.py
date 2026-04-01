import flax.linen as nn
import jax.numpy as jnp

from jaxrl5.networks import default_init


class StateActionValue(nn.Module):
    base_cls: nn.Module
    output_dim: int = 1

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(self.output_dim, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1) if self.output_dim == 1 else value

class Relu_StateActionValue(nn.Module):
    base_cls: nn.Module
    output_dim: int = 1

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(self.output_dim, kernel_init=default_init())(outputs)

        value = nn.softplus(value)

        return jnp.squeeze(value, -1) if self.output_dim == 1 else value
