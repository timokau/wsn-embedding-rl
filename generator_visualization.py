"""Some visual testing"""

from math import floor

import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from infrastructure import random_infrastructure, draw_infra
from overlay import random_overlay, draw_overlay
from embedding import PartialEmbedding, draw_embedding

def random_embedding(
        max_embedding_nodes=32,
        rand=np.random,
):
    """Generate matching random infrastructure + overlay + embedding"""
    infra = random_infrastructure(
        rand,
        min_nodes=3,
        max_nodes=max_embedding_nodes / 3,
        num_sources=2,
    )

    num_nodes = len(infra.graph.nodes())
    max_blocks = min([num_nodes, floor(max_embedding_nodes / num_nodes)])
    overlay = random_overlay(
        rand,
        min_blocks=3,
        max_blocks=max_blocks,
        num_sources=len(infra.sources),
    )
    source_mapping = dict()
    source_blocks = list(overlay.sources)
    source_nodes = list(infra.sources)
    rand.shuffle(source_nodes)
    source_mapping = list(zip(source_blocks, source_nodes))
    embedding = PartialEmbedding(
        infra,
        overlay,
        source_mapping,
    )
    return embedding

class Visualization():
    """Visualize embedding with its infrastructure and overlay"""
    def __init__(
            self,
            embedding: PartialEmbedding,
    ):
        self.colors = {
            'sources_color': 'red',
            'sink_color': 'yellow',
            'intermediates_color': 'green',
        }

        self.embedding = embedding

        self.infra_ax = plt.subplot2grid((2, 3), (0, 0))
        self.overlay_ax = plt.subplot2grid((2, 3), (1, 0))
        self.embedding_ax = plt.subplot2grid(
            shape=(2, 3),
            loc=(0, 1),
            rowspan=2,
            colspan=2,
        )

        self.update_infra()
        self.update_overlay()
        self.update_embedding()

        pa = mpatches.Patch
        plt.gcf().legend(handles=[
            pa(color=self.colors['sources_color'], label='source'),
            pa(color=self.colors['sink_color'], label='sink'),
            pa(color=self.colors['intermediates_color'], label='intermediate'),
        ])

    def update_infra(self):
        """Redraws the infrastructure"""
        plt.sca(self.infra_ax)
        plt.cla()
        self.infra_ax.set_title("Infrastructure")
        draw_infra(self.embedding.infra, **self.colors)

    def update_overlay(self):
        """Redraws the overlay"""
        plt.sca(self.overlay_ax)
        plt.cla()
        self.overlay_ax.set_title("Overlay")
        draw_overlay(self.embedding.overlay, **self.colors)

    def update_embedding(self):
        """Redraws the embedding"""
        plt.sca(self.embedding_ax)
        plt.cla()
        self.embedding_ax.set_title("Embedding")
        draw_embedding(self.embedding, **self.colors)


def random_action(
        embedding: PartialEmbedding,
        rand=np.random,
):
    """Take a random action on the given partial embedding"""
    possibilities = embedding.possibilities()
    if len(possibilities) == 0:
        print('No action possible')
        return
    choice = rand.randint(0, len(possibilities))
    action = possibilities[choice]
    print(f'Action is {action}')
    embedding.take_action(*action)

def _main():
    embedding = random_embedding()
    viz = Visualization(embedding)
    ax = plt.gca()

    def step(_):
        random_action(embedding)
        viz.update_embedding()
        plt.draw()

    btn = Button(ax, "Action")
    btn.on_clicked(step)

    plt.show()

if __name__ == "__main__":
    _main()
