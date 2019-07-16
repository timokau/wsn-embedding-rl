"""Tests the feature extraction"""

# Tests are verbose.
# pylint: disable=too-many-statements

import numpy as np
from pytest import approx

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
    bin3 = overlay.add_intermediate(name="bin3", datarate=42, requirement=0.7)
    bin4 = overlay.add_intermediate(name="bin4", datarate=0, requirement=0.2)
    bsi = overlay.set_sink(name="bsi", datarate=5, requirement=4)
    overlay.add_link(bso1, bin1)
    overlay.add_link(bso1, bin2)
    overlay.add_link(bin1, bsi)
    overlay.add_link(bin2, bsi)
    overlay.add_link(bso2, bsi)
    overlay.add_link(bso2, bin3)
    overlay.add_link(bin3, bin4)
    overlay.add_link(bin4, bsi)

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

    # this is always pretending the node isn't already chosen, so the
    # requirement of the block in question is always exempt
    assert node_feature("remaining_capacity", eso1)[0] == approx(1.5)
    assert node_feature("remaining_capacity", eso2)[0] == approx(2.8)
    assert node_feature("remaining_capacity", esi)[0] == np.infty
    assert node_feature("remaining_capacity", erelay)[0] == approx(0)
    assert node_feature("remaining_capacity", ein)[0] == np.infty
    assert node_feature("remaining_capacity", erelay_unchosen)[0] == approx(0)

    assert node_feature("weight", eso1)[0] == approx(1)
    assert node_feature("weight", esi)[0] == approx(4)
    assert node_feature("weight", erelay)[0] == approx(0)

    # the block itself should not be counted if it is already embedded
    # therefore, nso1 has a remaining capacity of 1.5 (instead of 0.5)
    assert node_feature("compute_fraction", eso1)[0] == approx(1 / 1.5)
    assert node_feature("compute_fraction", esi)[0] == 0  # /infty
    assert node_feature("compute_fraction", erelay)[0] == approx(0)

    # capacity inf can always embed everything
    assert node_feature("options_lost", esi)[0] == 0
    # has remaining capacity .5 after, can embed bin2 or bin4; before
    # also bin4
    assert node_feature("options_lost", eso1)[0] == 1
    # has remaining capacity .8 after, can embed bin2 or bin3 (just as
    # before)
    assert node_feature("options_lost", eso2)[0] == 0
    # remaining capacity of .1 after, can only embed bin2 (before also
    # bin4)
    assert node_feature("options_lost", ENode(bin3, nso2))[0] == 1

    def edge_feature(name, u, v, t):
        return tuple(
            feature_dict["edge_" + name].process_edge(embedding, u, v, t)
        )

    assert edge_feature("timeslot", eso1, erelay, 0)[0] == 0
    assert edge_feature("timeslot", eso2, esi, 2)[0] == 2

    assert edge_feature("chosen", eso1, erelay, 0)[0] == 1
    assert edge_feature("chosen", eso2, esi, 2)[0] == 0

    assert edge_feature("additional_timeslot", eso1, erelay, 0)[0] == 0
    assert edge_feature("additional_timeslot", eso2, esi, 2)[0] == 1

    assert edge_feature("datarate_requirement", eso1, erelay, 0)[0] == approx(
        5
    )
    assert edge_feature("datarate_requirement", erelay, ein, 1)[0] == approx(5)
