"""Tests the feature extraction"""

import numpy as np
from infrastructure import InfrastructureNetwork
from overlay import OverlayNetwork
from embedding import PartialEmbedding, ENode
from features import features_by_name


# easiest to do everything in one function, although it isn't pretty
def test_features():
    """Tests feature extractions on some hand-verified examples"""
    infra = InfrastructureNetwork()

    nso1 = infra.add_source(
        name="nso1", pos=(0, 0), transmit_power_dbm=30, capacity=1.5
    )
    nso2 = infra.add_source(
        name="nso2", pos=(0, 1), transmit_power_dbm=30, capacity=2.8
    )
    nsi = infra.set_sink(
        name="nsi", pos=(2, 0), transmit_power_dbm=30, capacity=np.infty
    )
    ninterm = infra.add_intermediate(
        name="ninterm", pos=(0, 1), transmit_power_dbm=30, capacity=0
    )

    overlay = OverlayNetwork()
    bso1 = overlay.add_source(name="bso1", datarate=5, requirement=1)
    bso2 = overlay.add_source(name="bso2", datarate=5, requirement=2)
    bin1 = overlay.add_intermediate(name="bin1", datarate=5, requirement=3)
    bin2 = overlay.add_intermediate(name="bin2", datarate=5, requirement=0)
    bsi = overlay.set_sink(name="bsi", datarate=5, requirement=4)
    overlay.add_link(bso1, bin1)
    overlay.add_link(bso1, bin2)
    overlay.add_link(bin1, bsi)
    overlay.add_link(bin2, bsi)
    overlay.add_link(bso2, bsi)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(bso1, nso1), (bso2, nso2)]
    )

    eso1 = ENode(bso1, nso1)
    eso2 = ENode(bso2, nso2)
    esi = ENode(bsi, nsi)
    erelay = ENode(bso1, ninterm, bin1)
    ein = ENode(bin1, nsi)
    erelay_unchosen = ENode(bso2, ninterm, bsi)

    assert embedding.take_action(eso1, erelay, 0)
    assert embedding.take_action(erelay, ein, 1)

    feature_dict = features_by_name()

    def node_feature(name, node):
        return tuple(
            feature_dict["node_" + name].process_node(embedding, node)
        )

    assert node_feature("pos", ein) == (2, 0)  # pos of nsi

    assert node_feature("relay", erelay) == (1.0,)
    assert node_feature("relay", erelay_unchosen) == (1.0,)
    assert node_feature("relay", eso1) == (0.0,)
    assert node_feature("relay", ein) == (0.0,)

    assert node_feature("sink", eso2) == (0.0,)
    assert node_feature("sink", ein) == (0.0,)
    assert node_feature("sink", esi) == (1.0,)
    num_sinks = 0
    for node in embedding.nodes():
        if node_feature("sink", node)[0] == 1.0:
            num_sinks += 1
    assert num_sinks == 1
