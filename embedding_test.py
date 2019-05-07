"""Tests the encoding of domain information into the embedding"""

from pytest import approx

from infrastructure import (
        InfrastructureNetwork,
)
from overlay import (
        OverlayNetwork,
)
from embedding import (
        PartialEmbedding,
        ENode,
)

def test_path_loss():
    """
    Tests that an embedding over impossible distances is recognized as
    invalid.
    """
    infra = InfrastructureNetwork()

    # Two nodes, 1km apart. The transmitting node has a transmission
    # power of 1dBm (=1.26mW). With a path loss over 1km of *at least*
    # 30dBm, less than ~-30dBm (approx. 10^-3 = 0.001mW = 1uW) arrives
    # at the target. That is a very optimistic approximation and is not
    # nearly enough to send any reasonable signal.
    source_node = infra.add_source(
        pos=(0, 0),
        transmit_power_dbm=1,
    )
    infra.set_sink(
        pos=(1000, 0),
        transmit_power_dbm=0, # does not matter
    )

    overlay = OverlayNetwork()
    source_block = overlay.add_source()
    sink_block = overlay.set_sink()

    overlay.add_link(source_block, sink_block)

    embedding = PartialEmbedding(
        infra,
        overlay,
        source_mapping=[(source_block, source_node)],
        timeslots=1,
        sinrth=2.0,
    )
    assert len(embedding.possibilities()) == 0

def test_trivial_possibilities():
    """
    Tests that a single reasonable option is correctly generated in a
    trivial case.
    """
    infra = InfrastructureNetwork()

    # Two nodes, 1m apart. The transmitting node has a
    # transmit_power_dbm
    # power of 30dBm (similar to a regular router) which should easily
    # cover the distance of 1m without any noise.
    source_node = infra.add_source(
        pos=(0, 0),
        transmit_power_dbm=.1,
    )
    infra.set_sink(
        pos=(1, 0),
        transmit_power_dbm=0, # does not matter
    )

    overlay = OverlayNetwork()
    source_block = overlay.add_source()
    sink_block = overlay.set_sink()

    overlay.add_link(source_block, sink_block)

    embedding = PartialEmbedding(
        infra,
        overlay,
        source_mapping=[(source_block, source_node)],
        timeslots=1,
        sinrth=2.0,
    )
    # either go to N2 as a relay (doesn't make sense but is a viable
    # option), or embed B2 into N2 and go there
    assert len(embedding.possibilities()) == 2

def test_manually_verified_sinr():
    """
    Tests that the SINR calculation agrees with a manually verified
    example.
    """
    infra = InfrastructureNetwork()

    # 2 sources, 2 intermediaries, 1 sink
    n_source1 = infra.add_source(pos=(0, 0), transmit_power_dbm=30)
    n_interm1 = infra.add_intermediate(pos=(1, 0), transmit_power_dbm=30)
    n_source2 = infra.add_source(pos=(0, 2), transmit_power_dbm=30)
    n_interm2 = infra.add_intermediate(pos=(1, 2), transmit_power_dbm=30)
    n_sink = infra.set_sink(pos=(3, 1), transmit_power_dbm=0)

    overlay = OverlayNetwork()
    b_source1 = overlay.add_source()
    b_interm1 = overlay.add_intermediate()
    b_source2 = overlay.add_source()
    b_interm2 = overlay.add_intermediate()
    b_sink = overlay.set_sink()

    overlay.add_link(b_source1, b_interm1)
    overlay.add_link(b_source2, b_interm2)

    # just to make the embedding complete
    overlay.add_link(b_interm1, b_sink)
    overlay.add_link(b_interm2, b_sink)

    embedding = PartialEmbedding(
        infra,
        overlay,
        timeslots=3,
        source_mapping=[(b_source1, n_source1), (b_source2, n_source2)],
    )

    # this doesn't actually do anything, just makes the next step more
    # convenient
    e_source1 = ENode(b_source1, n_source1)
    e_source2 = ENode(b_source2, n_source2)
    e_interm1 = ENode(b_interm1, n_interm1)
    e_interm2 = ENode(b_interm2, n_interm2)
    e_sink = ENode(b_sink, n_sink)

    # Here is the important part: Two signals in parallel
    embedding.take_action(e_source1, e_interm1, 0)
    embedding.take_action(e_source2, e_interm2, 0)

    # We don't really care what is going on in other timeslots, this is
    # just to make the embedding valid.
    embedding.take_action(e_interm1, e_sink, 1)
    embedding.take_action(e_interm2, e_sink, 2)

    # Now we have a clean model to work with in timeslot 1: Two parallel
    # communications, one signal and one noise.
    # Let's calculate the SINR or signal1.

    # source1 sends with 30dBm. There is a distance of 1m to interm1.
    # According to the log path loss model with a loss exponent of 4
    # (appropriate for a building), the signal will incur a loss of
    # 4 * distance_decibel dBm
    # Where distance_decibel is the distance in relation to 1m, i.e. 0
    # in this case. That means there is *no loss*, at least according to
    # the model.

    # It follows that interm1 receives a signal of 30dBm. Now on to the
    # received noise. source2 also sends with 30dBm and has a distance
    # of sqrt(1^2 + 2^2) ~= 2.24m to interm1. According to the log path
    # loss model:
    # distance_decibel = 10 * lg(2.24) ~= 3.50
    # => path_loss = 4 * 3.50 ~= 14 dBm

    # So interm1 receives roughly 16 dBm of noise. Lets assume a base
    # noise of 15dB. We have to add those two. Care must be taken here
    # because of the logarithmic scale. Naive addition would result in
    # multiplication of the actual power in watts. So we need to convert
    # back to watts first, then add, then convert back:
    # base_noise_milliwatts = 10^(1.5) ~= 31.62 mW
    # com_noise_milliwatts = 10^(1.6) ~= 39.81 mW
    # => total_noise = 31.62 + 39.81 = 71.43 mW
    # The total noise is 71.43 mW, which equals
    # 10*lg(71.43) ~= 18.54 dB
    # That is way less than the naively calculated 16 + 15 = 31 dB.

    # That means the SINR should be
    # sinr = 30dBm - 18.54dBm = 11.46dB
    # Here the subtraction actually *should* represent a division of the
    # powers.

    sinr = embedding.known_sinr(n_source1, n_interm1, 0, noise_floor_dbm=15)
    assert sinr == approx(11.46, abs=0.1)

def test_invalidating_earlier_choice_impossible():
    """
    Tests that an action that would invalidate an earlier action is
    impossible.
    """
    infra = InfrastructureNetwork()

    # Two sources, one sink. Equal distance from both sources to sink.
    # One source with moderate transmit power (but enough to cover the
    # distance, one source with excessive transmit power.
    # transmit_power_dbm
    # power of 30dBm (similar to a regular router) which should easily
    # cover the distance of 1m without any noise.
    source_node_silent = infra.add_source(
        pos=(0, 0),
        transmit_power_dbm=20,
        name="Silent",
    )
    source_node_screamer = infra.add_source(
        pos=(3, 0),
        transmit_power_dbm=100,
        name="Screamer",
    )
    node_sink = infra.set_sink(
        pos=(1, 3),
        transmit_power_dbm=0, # does not matter
        name="Sink",
    )

    overlay = OverlayNetwork()

    esource_silent = ENode(overlay.add_source(), source_node_silent)
    esource_screamer = ENode(overlay.add_source(), source_node_screamer)
    esink = ENode(overlay.set_sink(), node_sink)

    overlay.add_link(esource_silent.block, esink.block)
    overlay.add_link(esource_screamer.block, esink.block)

    embedding = PartialEmbedding(
        infra,
        overlay,
        source_mapping=[
            (esource_silent.block, esource_silent.node),
            (esource_screamer.block, esource_screamer.node),
        ],
        timeslots=1,
        sinrth=2.0,
    )

    # embed the link from the silent node to the sink
    embedding.take_action(esource_silent, esink, 0)

    # first assert that action would be valid by itself
    screamer_sinr = embedding.known_sinr(source_node_screamer, node_sink, 0)
    assert screamer_sinr > embedding.sinrth

    # but since the action would make the first embedding invalid (a
    # node cannot receive two signals at the same time), it should still
    # not be possible. That means there are no remaining possibilities.
    assert len(embedding.possibilities()) == 0

def test_no_unnecessary_options():
    """
    Tests that no unnecessary connections are offered.
    """
    infra = InfrastructureNetwork()

    # Two sources, one sink. Equal distance from both sources to sink.
    # One source with moderate transmit power (but enough to cover the
    # distance, one source with excessive transmit power.
    # transmit_power_dbm
    # power of 30dBm (similar to a regular router) which should easily
    # cover the distance of 1m without any noise.
    source_node = infra.add_source(
        pos=(0, 0),
        transmit_power_dbm=30,
        name="Source",
    )
    sink_node = infra.set_sink(
        pos=(1, 3),
        transmit_power_dbm=0,
        name="Sink",
    )

    overlay = OverlayNetwork()

    esource = ENode(overlay.add_source(), source_node)
    esink = ENode(overlay.set_sink(), sink_node)

    overlay.add_link(esource.block, esink.block)

    embedding = PartialEmbedding(
        infra,
        overlay,
        source_mapping=[
            (esource.block, esource.node),
        ],
        timeslots=2,
        sinrth=2.0,
    )

    # Currently, it would be possible to either embed the sink and
    # directly link to it, or to use the sink as a relay (although that
    # wouldn't make much sense). Each of those options could be taken at
    # either timeslot.
    assert len(embedding.possibilities()) == 4

    # embed the sink
    embedding.take_action(esource, esink, 0)

    # Now it would still be *feasible* according to add a connection to
    # the relay in the other timeslot. It shouldn't be possible however,
    # since all outgoing connections are already embedded.
    assert len(embedding.possibilities()) == 0

def test_all_viable_options_offered():
    """
    Tests that all manually verified options are offered in a concrete
    example.
    """
    infra = InfrastructureNetwork()

    # Two sources, one sink, one intermediate, one relay
    # Enough transmit power so that it doesn't need to be taken into account
    nsource1 = infra.add_source(
        pos=(0, 0),
        # transmit power should not block anything in this example
        transmit_power_dbm=100,
        name="N1",
    )
    nsource2 = infra.add_source(
        pos=(1, 0),
        transmit_power_dbm=100,
        name="N2",
    )
    _nrelay = infra.add_intermediate(
        pos=(0, 1),
        transmit_power_dbm=100,
        name="N3",
    )
    ninterm = infra.add_intermediate(
        pos=(2, 0),
        transmit_power_dbm=100,
        name="N4",
    )
    nsink = infra.set_sink(
        pos=(1, 1),
        transmit_power_dbm=100,
        name="N5",
    )

    overlay = OverlayNetwork()

    esource1 = ENode(overlay.add_source(), nsource1)
    esource2 = ENode(overlay.add_source(), nsource2)
    einterm = ENode(overlay.add_intermediate(), ninterm)
    esink = ENode(overlay.set_sink(), nsink)

    # source1 connects to the sink over the intermediate source2
    # connects both to the sink and to source1.
    overlay.add_link(esource1.block, einterm.block)
    overlay.add_link(einterm.block, esink.block)
    overlay.add_link(esource2.block, esink.block)
    overlay.add_link(esource2.block, esource1.block)

    embedding = PartialEmbedding(
        infra,
        overlay,
        source_mapping=[
            (esource1.block, esource1.node),
            (esource2.block, esource2.node),
        ],
        timeslots=2,
        sinrth=2.0,
    )

    # source1 can connect to the intermediate, which could be embedded
    # in either of two nodes (2). It could also connect to any other
    # node as a relay (4) -> 6
    # source2 can connect to the sink (1) or the other source (1). It
    # could also connect to any other node as a relay (4) -> 6
    # All of this could happen at either timeslot (*2)
    assert len(embedding.possibilities()) == (6 + 6) * 2

def test_completion_detection():
    """
    Tests that the completeness of an embedding is accurately detected
    in a simple example.
    """
    infra = InfrastructureNetwork()

    # One source, one sink, one relay.
    # Enough transmit power so that it doesn't need to be taken into account
    nsource = infra.add_source(
        pos=(0, 0),
        # transmit power should not block anything in this example
        transmit_power_dbm=100,
    )
    _nrelay = infra.add_intermediate(
        pos=(0, 1),
        transmit_power_dbm=100,
    )
    nsink = infra.set_sink(
        pos=(1, 1),
        transmit_power_dbm=100,
    )

    overlay = OverlayNetwork()

    esource = ENode(overlay.add_source(), nsource)
    esink = ENode(overlay.set_sink(), nsink)

    overlay.add_link(esource.block, esink.block)

    embedding = PartialEmbedding(
        infra,
        overlay,
        source_mapping=[
            (esource.block, esource.node),
        ],
        timeslots=1,
        sinrth=2.0,
    )

    assert not embedding.is_complete()

    embedding.take_action(esource, esink, 0)

    assert embedding.is_complete()
