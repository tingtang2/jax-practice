from jax import numpy as jnp
from jax import vmap, jit
from matplotlib import pyplot as plt


def heatmap_data(positions, area_min=-2, area_max=2):

    def small_kernel(z, area_min, area_max):
        a = jnp.linspace(area_min, area_max, 512)
        x, y = jnp.meshgrid(a, a)
        dist = (x - z[0])**2 + (y - z[1])**2
        hm = jnp.exp(-350 * dist)
        return hm

    @jit
    def produce_heatmap(positions, area_min, area_max):
        return jnp.sum(vmap(small_kernel,
                            in_axes=(0, None, None))(positions, area_min,
                                                     area_max),
                       axis=0)

    hm = produce_heatmap(positions, area_min, area_max)
    return hm


def plot_heatmap(positions, area_min=-2, area_max=2):
    '''
    positions: locations of all particles in R^2, array (J, 2)
    area_min: lowest x and y coordinate
    area_max: highest x and y coordinate
    
    will plot a heatmap of all particles in the area [area_min, area_max] x [area_min, area_max]
    '''
    hm = heatmap_data(positions, area_min, area_max)
    extent = [area_min, area_max, area_max, area_min]
    im = plt.imshow(hm, interpolation='nearest', extent=extent)
    ax = plt.gca()
    ax.invert_yaxis()

    plt.savefig('heatmap_fig.png')
