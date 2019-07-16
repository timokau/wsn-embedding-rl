"""Tests the feature extraction"""

from infrastructure import InfrastructureNetwork
from overlay import OverlayNetwork
from embedding import PartialEmbedding, ENode
from features import features_by_name


# easiest to do everything in one function, although it isn't pretty
def test_features():
    """Tests feature extractions on some hand-verified examples"""
    infra = InfrastructureNetwork()

    nso1 = infra.add_source(name="nso1", pos=(0, 0), transmit_power_dbm=30)
    nso2 = infra.add_source(name="nso2", pos=(0, 1), transmit_power_dbm=30)
    nsi = infra.set_sink(name="nsi", pos=(2, 0), transmit_power_dbm=30)

    overlay = OverlayNetwork()
    bso1 = overlay.add_source(name="bso1", datarate=5, requirement=0)
    bso2 = overlay.add_source(name="bso2", datarate=5, requirement=0)
    bin_ = overlay.add_intermediate(name="bin", datarate=5, requirement=0)
    bsi = overlay.set_sink(name="bsi", datarate=5, requirement=0)
    overlay.add_link(bso1, bin_)
    overlay.add_link(bin_, bsi)
    overlay.add_link(bso2, bsi)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(bso1, nso1), (bso2, nso2)]
    )

    _eso1 = ENode(bso1, nso1)
    _eso2 = ENode(bso2, nso2)
    ein = ENode(bin_, nsi)
    _esi = ENode(bsi, nsi)
    _erelay = ENode(bso1, nso2, bin_)

    feature_dict = features_by_name()

    def node_feature(name, node):
        return tuple(
            feature_dict["node_" + name].process_node(embedding, node)
        )

    assert node_feature("pos", ein) == (2, 0)  # pos of nsi
