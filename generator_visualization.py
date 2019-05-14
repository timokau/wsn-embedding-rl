"""Some visual testing"""

import re
from math import floor

import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, TextBox

from infrastructure import random_infrastructure, draw_infra
from overlay import random_overlay, draw_overlay
from embedding import ENode, PartialEmbedding, draw_embedding

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

def _parse_embedding_str(embedding_str):
    if '-' not in embedding_str:
        return ENode(None, embedding_str)
    return ENode(*embedding_str.split('-'))

COLORS = {
    'sources_color': 'red',
    'sink_color': 'yellow',
    'intermediates_color': 'green',
}

class Visualization():
    """Visualize embedding with its infrastructure and overlay"""
    def __init__(
            self,
            embedding: PartialEmbedding,
    ):
        self.embedding = embedding

        shape = (2, 3)
        self.infra_ax = plt.subplot2grid(
            shape=shape,
            loc=(0, 0),
        )
        self.overlay_ax = plt.subplot2grid(
            shape=shape,
            loc=(1, 0),
        )
        self.embedding_ax = plt.subplot2grid(
            shape=shape,
            loc=(0, 1),
            rowspan=2,
            colspan=2,
        )
        plt.subplots_adjust(bottom=0.2)
        input_text_ax = plt.axes([
            0.1, # left
            0.05, # bottom
            0.6, # width
            0.075 # height
        ])
        input_btn_ax = plt.axes([
            0.7, # left
            0.05, # bottom
            0.2, # width
            0.075 # height
        ])

        self.update_infra()
        self.update_overlay()
        self.update_embedding()

        pa = mpatches.Patch
        plt.gcf().legend(handles=[
            pa(color=COLORS['sources_color'], label='source'),
            pa(color=COLORS['sink_color'], label='sink'),
            pa(color=COLORS['intermediates_color'], label='intermediate'),
        ])


        random = get_random_action(self.embedding)
        self.text_box_val = str(random)
        self.text_box = TextBox(
            input_text_ax,
            'Action',
            initial=self.text_box_val,
        )


        def _update_textbox_val(new_val):
            self.text_box_val = new_val
        self.text_box.on_text_change(_update_textbox_val)

        self.submit_btn = Button(input_btn_ax, "Take action")

        def _on_clicked(_):
            self._take_action(self._parse_textbox())
        self.submit_btn.on_clicked(_on_clicked)

    def _parse_textbox(self):
        action = self.text_box_val
        pattern = r'\(([^,]+), ([^,]+), ([^,]+)\)'
        match = re.match(pattern, action)
        if match is None:
            print('Action could not be parsed')
            return None
        source_embedding = _parse_embedding_str(match.group(1))
        target_embedding = _parse_embedding_str(match.group(2))
        timeslot = int(match.group(3))
        return (source_embedding, target_embedding, timeslot)

    def _update_textbox(self):
        next_random = get_random_action(self.embedding)
        self.text_box.set_val(
            str(next_random) if next_random is not None else ''
        )

    def _take_action(self, action):
        if action is None:
            print('Action could not be parsed')
            return
        print(f'Taking action: {action}')
        success = self.embedding.take_action(*action)
        if not success:
            print('Action is not valid. The possibilities are:')
        self.update_embedding()
        self._update_textbox()
        plt.draw()

    def update_infra(self):
        """Redraws the infrastructure"""
        plt.sca(self.infra_ax)
        plt.cla()
        self.infra_ax.set_title("Infrastructure")
        draw_infra(self.embedding.infra, **COLORS)

    def update_overlay(self):
        """Redraws the overlay"""
        plt.sca(self.overlay_ax)
        plt.cla()
        self.overlay_ax.set_title("Overlay")
        draw_overlay(self.embedding.overlay, **COLORS)

    def update_embedding(self):
        """Redraws the embedding"""
        plt.sca(self.embedding_ax)
        plt.cla()
        self.embedding_ax.set_title("Embedding")
        draw_embedding(self.embedding, **COLORS)


def get_random_action(
        embedding: PartialEmbedding,
        rand=np.random,
):
    """Take a random action on the given partial embedding"""
    possibilities = embedding.possibilities()
    if len(possibilities) == 0:
        print('No action possible')
        return None
    choice = rand.randint(0, len(possibilities))
    action = possibilities[choice]
    print(f'Picking random action {action} from')
    for possibility in possibilities:
        print('\t' + str(possibility))
    return action

def _main():
    embedding = random_embedding()
    Visualization(embedding)
    plt.show()

if __name__ == "__main__":
    _main()
