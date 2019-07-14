"""Tests the encoding of domain information into the embedding"""

# The tests are verbose, but they are not intended to read exhaustively
# anyway. When reading particular failing examples, verbosity is good.
# pylint:disable=too-many-lines

from pytest import approx

from infrastructure import InfrastructureNetwork
from overlay import OverlayNetwork
from embedding import PartialEmbedding, ENode

# Sometimes its nicer to start variable names with uppercase letters
# pylint: disable=invalid-name


def take_action(embedding, action, expect_success=True):
    """Takes an action by text, to ease testing"""
    possibilities = embedding.possibilities()
    for possibility in possibilities:
        if str(possibility) == action:
            embedding.take_action(*possibility)
            return True
    if expect_success:
        raise Exception(
            f"Action {action} not in possibilities: {possibilities}"
        )
    return False


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
    source_node = infra.add_source(pos=(0, 0), transmit_power_dbm=1)
    infra.set_sink(pos=(1000, 0), transmit_power_dbm=0)

    overlay = OverlayNetwork()
    source_block = overlay.add_source()
    sink_block = overlay.set_sink()

    overlay.add_link(source_block, sink_block)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(source_block, source_node)]
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
    source_node = infra.add_source(pos=(0, 0), transmit_power_dbm=0.1)
    infra.set_sink(pos=(1, 0), transmit_power_dbm=0)

    overlay = OverlayNetwork()
    source_block = overlay.add_source()
    sink_block = overlay.set_sink()

    overlay.add_link(source_block, sink_block)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(source_block, source_node)]
    )
    # can only embed B2 into N2
    assert len(embedding.possibilities()) == 1


def test_manually_verified_sinr():
    """
    Tests that the SINR calculation agrees with a manually verified
    example.
    """
    infra = InfrastructureNetwork(noise_floor_dbm=15)

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
    # According to the log path loss model with a loss exponent of 2
    # (appropriate for a building), the signal will incur a loss of
    # 2 * distance_decibel dBm
    # Where distance_decibel is the distance in relation to 1m, i.e. 0
    # in this case. That means there is *no loss*, at least according to
    # the model.

    # It follows that interm1 receives a signal of 30dBm. Now on to the
    # received noise. source2 also sends with 30dBm and has a distance
    # of sqrt(1^2 + 2^2) ~= 2.24m to interm1. According to the log path
    # loss model:
    # distance_decibel = 10 * lg(2.24) ~= 3.50
    # => path_loss = 2 * 3.50 ~= 7 dBm

    # So interm1 receives roughly 30 - 7 = 23 dBm of noise. Lets assume
    # a base noise of 15dB. We have to add those two. Care must be taken
    # here because of the logarithmic scale. Naive addition would result
    # in multiplication of the actual power in watts. So we need to
    # convert back to watts first, then add, then convert back:
    # base_noise_milliwatts = 10^(1.5) ~= 31.62 mW
    # com_noise_milliwatts = 10^(2.3) ~= 199.53 mW
    # => total_noise = 31.62 + 199.53 = 231.15 mW
    # The total noise is 231.15 mW, which equals
    # 10*lg(231.15) ~= 23.64 dB
    # That is way less than the naively calculated 16 + 15 = 31 dB.

    # That means the SINR should be
    # sinr = 30dBm - 23.64 = 6.36dB
    # Here the subtraction actually *should* represent a division of the
    # powers.

    sinr = embedding.known_sinr(n_source1, n_interm1, 0)
    assert sinr == approx(6.36, abs=0.1)


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
        pos=(0, 0), transmit_power_dbm=20, name="Silent"
    )
    source_node_screamer = infra.add_source(
        pos=(3, 0), transmit_power_dbm=100, name="Screamer"
    )
    node_sink = infra.set_sink(pos=(1, 3), transmit_power_dbm=0, name="Sink")

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
    )

    action_to_be_invalidated = (esource_screamer, esink, 0)
    # make sure the action is an option in the first place
    assert action_to_be_invalidated in embedding.possibilities()

    # embed the link from the silent node to the sink
    embedding.take_action(esource_silent, esink, 0)

    # first assert that action would be valid by itself
    screamer_sinr = embedding.known_sinr(source_node_screamer, node_sink, 0)
    assert screamer_sinr > 2.0

    new_possibilities = embedding.possibilities()
    # but since the action would make the first embedding invalid (a
    # node cannot receive two signals at the same time), it should still
    # not be possible
    assert action_to_be_invalidated not in new_possibilities

    # since there are no options left in the first timeslot, there are
    # now exactly 2 (screamer -> silent as relay, screamer -> sink
    # embedded) options left in the newly created second timeslot
    assert len(new_possibilities) == 2


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
        pos=(0, 0), transmit_power_dbm=30, name="Source"
    )
    sink_node = infra.set_sink(pos=(1, 3), transmit_power_dbm=0, name="Sink")

    overlay = OverlayNetwork()

    esource = ENode(overlay.add_source(), source_node)
    esink = ENode(overlay.set_sink(), sink_node)

    overlay.add_link(esource.block, esink.block)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(esource.block, esource.node)]
    )

    assert len(embedding.possibilities()) == 1

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
    nso1 = infra.add_source(
        pos=(0, 0),
        # transmit power should not block anything in this example
        transmit_power_dbm=100,
        name="nso1",
    )
    nso2 = infra.add_source(pos=(1, 0), transmit_power_dbm=100, name="nso2")
    _nrelay = infra.add_intermediate(
        pos=(0, 1), transmit_power_dbm=100, name="nr"
    )
    _ninterm = infra.add_intermediate(
        pos=(2, 0), transmit_power_dbm=100, name="ni"
    )
    _nsink = infra.set_sink(pos=(1, 1), transmit_power_dbm=100, name="nsi")

    overlay = OverlayNetwork()

    bso1 = overlay.add_source(name="bso1")
    bso2 = overlay.add_source(name="bso2")
    bsi = overlay.set_sink(name="bsi")
    bin_ = overlay.add_intermediate(name="bin")

    eso1 = ENode(bso1, nso1)
    eso2 = ENode(bso2, nso2)

    # source1 connects to the sink over the intermediate source2
    # connects both to the sink and to source1.
    overlay.add_link(bso1, bin_)
    overlay.add_link(bin_, bsi)
    overlay.add_link(bso2, bsi)
    overlay.add_link(bso2, bso1)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(bso1, eso1.node), (bso2, eso2.node)]
    )

    # source1 can connect to the intermediate, which could be embedded
    # in any node (5). It could also connect to any other node as a
    # relay (4) -> 9. source2 can connect to the sink (1) or the other
    # source (1). It could also connect to any other node as a relay for
    # either of its two links (2 * 3) -> 8 No timeslot is used yet, so
    # there is just one timeslot option.
    assert len(embedding.possibilities()) == 9 + 8


def test_timeslots_dynamically_created():
    """Tests the dynamic creation of new timeslots as needed"""
    infra = InfrastructureNetwork()

    nso1 = infra.add_source(
        name="nso1",
        pos=(0, 0),
        # transmits so loudly that no other node can realistically
        # transmit in the same timeslot
        transmit_power_dbm=1000,
    )
    nso2 = infra.add_source(name="nso2", pos=(1, 0), transmit_power_dbm=1000)
    nsi = infra.set_sink(name="nsi", pos=(1, 1), transmit_power_dbm=1000)

    overlay = OverlayNetwork()
    bso1 = overlay.add_source(name="bso1")
    bso2 = overlay.add_source(name="bso2")
    bsi = overlay.set_sink(name="bsi")

    eso1 = ENode(bso1, nso1)
    esi = ENode(bsi, nsi)

    overlay.add_link(bso1, bsi)
    overlay.add_link(bso2, bsi)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(bso1, nso1), (bso2, nso2)]
    )

    # nothing used yet
    assert embedding.used_timeslots == 0

    # it would be possible to create a new timeslot and embed either
    # link in it (2) or go to a relay from either source (2)
    assert len(embedding.possibilities()) == 4

    # Take an action. nosurce1 will transmit so strongly that nso2
    # cannot send at the same timelot
    assert embedding.take_action(eso1, esi, 0)

    # timeslot 0 is now used
    assert embedding.used_timeslots == 1

    # New options (for creating timeslot 1) were created accordingly.
    # The second source could now still send to the other source as a
    # relay or to to the sink directly, it will just have to do it in a
    # new timeslot.
    assert len(embedding.possibilities()) == 2


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
    _nrelay = infra.add_intermediate(pos=(0, 1), transmit_power_dbm=100)
    nsink = infra.set_sink(pos=(1, 1), transmit_power_dbm=100)

    overlay = OverlayNetwork()

    esource = ENode(overlay.add_source(), nsource)
    esink = ENode(overlay.set_sink(), nsink)

    overlay.add_link(esource.block, esink.block)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(esource.block, esource.node)]
    )

    assert not embedding.is_complete()

    embedding.take_action(esource, esink, 0)

    assert embedding.is_complete()


def test_parallel_receive_impossible():
    """
    Tests that receiving from two sender nodes at the same time is
    impossible
    """
    infra = InfrastructureNetwork()

    nsource1 = infra.add_source(pos=(0, 0), transmit_power_dbm=30)
    nsource2 = infra.add_source(pos=(3, 0), transmit_power_dbm=30)
    nsink = infra.set_sink(pos=(2, 0), transmit_power_dbm=30)

    overlay = OverlayNetwork()

    esource1 = ENode(overlay.add_source(), nsource1)
    esource2 = ENode(overlay.add_source(), nsource2)
    esink = ENode(overlay.set_sink(), nsink)

    # two incoming connections to sink
    overlay.add_link(esource1.block, esink.block)
    overlay.add_link(esource2.block, esink.block)

    embedding = PartialEmbedding(
        infra,
        overlay,
        source_mapping=[
            (esource1.block, esource1.node),
            (esource2.block, esource2.node),
        ],
    )

    # Try to send two signals to sink at the same timeslot. This should
    # fail, as either one signal should overshadow the other.
    embedding.take_action(esource1, esink, 0)
    assert not embedding.take_action(esource2, esink, 0)


def test_broadcast_possible():
    """Tests that broadcast is possible despite SINR constraints"""
    infra = InfrastructureNetwork()

    # One source, one sink, one intermediate
    nsource = infra.add_source(pos=(0, 0), transmit_power_dbm=30)
    ninterm = infra.add_intermediate(pos=(1, 2), transmit_power_dbm=30)
    nsink = infra.set_sink(pos=(2, 0), transmit_power_dbm=30)

    overlay = OverlayNetwork()

    esource = ENode(overlay.add_source(), nsource)
    einterm = ENode(overlay.add_intermediate(), ninterm)
    esink = ENode(overlay.set_sink(), nsink)

    # fork
    overlay.add_link(esource.block, einterm.block)
    overlay.add_link(esource.block, esink.block)

    # make complete
    overlay.add_link(einterm.block, esink.block)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(esource.block, esource.node)]
    )

    # Broadcast from source to sink and intermediate
    sinr_before = embedding.known_sinr(esource.node, esink.node, timeslot=0)
    assert embedding.take_action(esource, esink, 0)
    # Easiest way to test this, easy to change if internals change.
    # pylint: disable=protected-access
    power_at_sink = embedding.infra.power_at_node(
        esink.node, frozenset(embedding._nodes_sending_in[0])
    )
    assert embedding.take_action(esource, einterm, 0)

    # Make sure the broadcasting isn't counted twice
    new_power = embedding.infra.power_at_node(
        esink.node, frozenset(embedding._nodes_sending_in[0])
    )
    assert new_power == power_at_sink

    # Make sure the broadcasts do not interfere with each other
    assert sinr_before == embedding.known_sinr(
        esource.node, esink.node, timeslot=0
    )


def test_count_timeslots_multiple_sources():
    """Tests correct counting behaviour with multiple sources"""
    infra = InfrastructureNetwork()

    nsource1 = infra.add_source(pos=(0, -1), transmit_power_dbm=30)
    nsource2 = infra.add_source(pos=(0, 1), transmit_power_dbm=30)
    nsink = infra.set_sink(pos=(1, 0), transmit_power_dbm=30)

    overlay = OverlayNetwork()

    esource1 = ENode(overlay.add_source(), nsource1)
    esource2 = ENode(overlay.add_source(), nsource2)
    esink = ENode(overlay.set_sink(), nsink)

    overlay.add_link(esource1.block, esink.block)
    overlay.add_link(esource2.block, esink.block)

    embedding = PartialEmbedding(
        infra,
        overlay,
        source_mapping=[
            (esource1.block, esource1.node),
            (esource2.block, esource2.node),
        ],
    )

    assert not embedding.is_complete()
    assert embedding.used_timeslots == 0

    assert embedding.take_action(esource1, esink, 0)

    assert not embedding.is_complete()
    assert embedding.used_timeslots == 1

    assert embedding.take_action(esource2, esink, 1)

    assert embedding.is_complete()
    assert embedding.used_timeslots == 2


def test_count_timeslots_parallel():
    """Tests correct counting behaviour with parallel connections"""
    infra = InfrastructureNetwork()

    # One source, one sink, two intermediates
    nsource = infra.add_source(
        pos=(0, 0), transmit_power_dbm=30, name="nsource"
    )
    ninterm1 = infra.add_intermediate(
        pos=(1, 2), transmit_power_dbm=30, name="ninterm1"
    )
    ninterm2 = infra.add_intermediate(
        pos=(1, -2), transmit_power_dbm=30, name="ninterm2"
    )
    nsink = infra.set_sink(pos=(2, 0), transmit_power_dbm=30, name="nsink")

    overlay = OverlayNetwork()

    esource = ENode(overlay.add_source(name="bsource"), nsource)
    einterm1 = ENode(overlay.add_intermediate(name="binterm1"), ninterm1)
    einterm2 = ENode(overlay.add_intermediate(name="binterm2"), ninterm2)
    esink = ENode(overlay.set_sink(name="bsink"), nsink)

    # fork
    overlay.add_link(esource.block, einterm1.block)
    overlay.add_link(esource.block, einterm2.block)

    overlay.add_link(einterm1.block, esink.block)
    overlay.add_link(einterm2.block, esink.block)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(esource.block, esource.node)]
    )

    assert not embedding.is_complete()
    assert embedding.used_timeslots == 0

    assert embedding.take_action(esource, einterm1, 0)
    assert embedding.take_action(esource, einterm2, 0)

    assert not embedding.is_complete()
    assert embedding.used_timeslots == 1

    assert embedding.take_action(einterm1, esink, 1)

    assert not embedding.is_complete()
    assert embedding.used_timeslots == 2

    assert embedding.take_action(einterm2, esink, 2)

    assert embedding.is_complete()
    assert embedding.used_timeslots == 3


def test_count_timeslots_loop():
    """Tests reasonable counting behaviour with loops"""
    infra = InfrastructureNetwork()

    # One source, one sink, two intermediates
    nsource = infra.add_source(pos=(0, 0), transmit_power_dbm=30, name="nso")
    ninterm1 = infra.add_intermediate(
        pos=(2, 1), transmit_power_dbm=5, name="ni1"
    )
    ninterm2 = infra.add_intermediate(
        pos=(0, -1), transmit_power_dbm=5, name="ni2"
    )
    nsink = infra.set_sink(pos=(2, 0), transmit_power_dbm=30, name="nsi")

    overlay = OverlayNetwork()

    esource = ENode(overlay.add_source(name="bso"), nsource)
    einterm1 = ENode(overlay.add_intermediate(name="bi1"), ninterm1)
    einterm2 = ENode(overlay.add_intermediate(name="bi2"), ninterm2)
    esink = ENode(overlay.set_sink(name="bsi"), nsink)

    overlay.add_link(esource.block, einterm1.block)
    overlay.add_link(einterm1.block, esink.block)
    overlay.add_link(esink.block, einterm2.block)
    overlay.add_link(einterm2.block, esource.block)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(esource.block, esource.node)]
    )

    assert not embedding.is_complete()
    assert embedding.used_timeslots == 0

    assert embedding.take_action(esource, einterm1, 0)

    assert not embedding.is_complete()
    assert embedding.used_timeslots == 1

    assert embedding.take_action(einterm1, esink, 1)

    assert not embedding.is_complete()
    assert embedding.used_timeslots == 2

    assert embedding.take_action(esink, einterm2, 2)

    assert not embedding.is_complete()
    assert embedding.used_timeslots == 3

    assert embedding.take_action(einterm2, esource, 1)

    assert embedding.is_complete()
    assert embedding.used_timeslots == 3


def test_outlinks_limited():
    """
    Tests that the number of possible outlinks is limited by the number
    of outlinks to embed for that block.
    """
    infra = InfrastructureNetwork()

    nsource = infra.add_source(pos=(0, 0), transmit_power_dbm=1, name="nso")
    nrelay = infra.add_intermediate(
        pos=(1, 0), transmit_power_dbm=1, name="nr"
    )
    # The sink is way out of reach, embedding is not possible
    _nsink = infra.set_sink(pos=(1, 1), transmit_power_dbm=1, name="nsi")

    overlay = OverlayNetwork()
    bsource = overlay.add_source(name="bso")
    bsink = overlay.set_sink(name="bsi")

    esource = ENode(bsource, nsource)

    overlay.add_link(bsource, bsink)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(esource.block, esource.node)]
    )

    erelay = ENode(bsource, nrelay, bsink)
    assert embedding.take_action(esource, erelay, 0)

    possibilities_from_source = [
        (source, target, timeslot)
        for (source, target, timeslot) in embedding.possibilities()
        if source == esource
    ]
    # the source block has one outgoing edge, one outlink is already
    # embedded (although the link is not embedded completely)
    assert len(possibilities_from_source) == 0

    possibilities_from_relay = [
        (source, target, timeslot)
        for (source, target, timeslot) in embedding.possibilities()
        if source == erelay
    ]
    # yet the link can be continued from the relay
    assert len(possibilities_from_relay) > 0


def test_loop_within_infra_possible():
    """
    Tests that a loop within the infrastructure is always possible and
    does not interfere with other connections. This can be used to embed
    multiple consecutive blocks within one node.
    """
    infra = InfrastructureNetwork()

    nsource = infra.add_source(pos=(0, 0), transmit_power_dbm=30, name="nso")
    nsink = infra.set_sink(pos=(1, 0), transmit_power_dbm=30, name="nsi")

    overlay = OverlayNetwork()

    esource = ENode(overlay.add_source(name="bso"), nsource)
    einterm = ENode(overlay.add_intermediate(name="bin"), nsource)
    esink = ENode(overlay.set_sink(name="bsi"), nsink)

    overlay.add_link(esource.block, einterm.block)
    overlay.add_link(einterm.block, esink.block)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(esource.block, esource.node)]
    )

    sinr_before = embedding.known_sinr(nsource, nsink, 0)
    assert embedding.take_action(esource, einterm, 0)
    sinr_after = embedding.known_sinr(nsource, nsink, 0)
    assert sinr_before == sinr_after

    assert embedding.take_action(einterm, esink, 0)
    assert embedding.is_complete()


def test_link_edges_cannot_be_embedded_twice():
    """Tests that edges completing a link that is already embedded are
    removed or not even added when creating a new timestep"""
    infra = InfrastructureNetwork()
    nso = infra.add_source(pos=(0, 0), transmit_power_dbm=30, name="nso")
    nsi = infra.set_sink(pos=(2, 0), transmit_power_dbm=30, name="nsi")
    _nint = infra.add_intermediate(
        pos=(1, -1), transmit_power_dbm=30, name="nint"
    )

    overlay = OverlayNetwork()
    bso = overlay.add_source(name="bso")
    bsi = overlay.set_sink(name="bsi")
    bint = overlay.add_intermediate(name="bint")

    overlay.add_link(bso, bsi)
    overlay.add_link(bso, bint)
    overlay.add_link(bint, bsi)

    embedding = PartialEmbedding(infra, overlay, source_mapping=[(bso, nso)])

    eso = ENode(bso, nso)
    esi = ENode(bsi, nsi)
    # now the link from source to sink is already embedded, only the one
    # from source to intermediate should be left
    assert embedding.take_action(eso, esi, 0)

    # so embedding it again should not be possible
    assert not embedding.take_action(ENode(bso, nso), ENode(bsi, nsi), 1)


def test_unnecessary_links_removed_in_other_timeslots():
    """
    Tests that links in other timeslots are removed if they are embedded
    in one timeslot.
    """
    infra = InfrastructureNetwork()

    nfaraway_1 = infra.add_source(
        pos=(999999998, 99999999), transmit_power_dbm=5, name="nfaraway_1"
    )
    nfaraway_2 = infra.add_intermediate(
        pos=(999999999, 99999999), transmit_power_dbm=5, name="nfaraway_2"
    )

    nsi = infra.set_sink(pos=(9, 5), transmit_power_dbm=12, name="nsi")
    nso = infra.add_source(pos=(8, 3), transmit_power_dbm=3, name="nso")

    overlay = OverlayNetwork()

    bsi = overlay.set_sink(name="bsi")
    bso = overlay.add_source(name="bso")
    bfaraway_1 = overlay.add_source(name="bfaraway_1")
    bfaraway_2 = overlay.add_intermediate(name="bfaraway_2", datarate=0)

    overlay.add_link(bso, bsi)
    overlay.add_link(bfaraway_1, bfaraway_2)
    # just to make it correct
    overlay.add_link(bfaraway_2, bsi)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(bso, nso), (bfaraway_1, nfaraway_1)]
    )

    esi = ENode(bsi, nsi)
    eso = ENode(bso, nso)
    efaraway_1 = ENode(bfaraway_1, nfaraway_1)
    efaraway_2 = ENode(bfaraway_2, nfaraway_2)

    # make sure a second timeslot is created
    assert embedding.take_action(efaraway_1, efaraway_2, 0)

    # make sure embedding is possible in ts1
    assert (eso, esi, 1) in embedding.possibilities()

    # embed the link in ts 0
    assert embedding.take_action(eso, esi, 0)

    # now no embedding in another timeslot should be possible anymore
    possible_outlinks_from_eso = [
        pos for pos in embedding.possibilities() if pos[0] == eso
    ]
    assert len(possible_outlinks_from_eso) == 0


def test_block_embedding_is_unique():
    """Tests that other embedding options are removed once one of them
    is chosen"""
    infra = InfrastructureNetwork()

    nso1 = infra.add_source(pos=(0, 0), transmit_power_dbm=26, name="nso1")
    nso2 = infra.add_source(pos=(2, 0), transmit_power_dbm=26, name="nso2")
    _nsi = infra.set_sink(pos=(0, 1), transmit_power_dbm=16, name="nsi")
    n1 = infra.add_intermediate(pos=(1, 0), transmit_power_dbm=16, name="n1")
    _n2 = infra.add_intermediate(pos=(1, 1), transmit_power_dbm=16, name="n2")

    overlay = OverlayNetwork()

    bso1 = overlay.add_source(name="bso1")
    bso2 = overlay.add_source(name="bso2")
    binterm = overlay.add_intermediate(name="binterm")
    bsi = overlay.set_sink(name="bsi")

    overlay.add_link(bso1, binterm)
    # there are multiple in-edges to binterm, which could lead to
    # multiple different embeddings
    overlay.add_link(bso2, binterm)
    overlay.add_link(binterm, bsi)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(bso1, nso1), (bso2, nso2)]
    )

    eso1 = ENode(bso1, nso1)

    def embeddings_for_block(block):
        count = 0
        for node in embedding.graph.nodes():
            if node.block == block:
                count += 1
        return count

    # could embed binterm in multiple blocks
    assert embeddings_for_block(binterm) > 1

    # decide for one embedding
    assert embedding.take_action(eso1, ENode(binterm, n1), 0)

    # other options are removed
    assert embeddings_for_block(binterm) == 1


def test_block_capacity():
    """Tests that per-node capacity is respected for each timeslot"""
    infra = InfrastructureNetwork()

    nso = infra.add_source(pos=(0, 0), transmit_power_dbm=30, name="nso")
    nin1 = infra.add_intermediate(
        pos=(-1, 1), transmit_power_dbm=30, capacity=42, name="nin1"
    )
    nin2 = infra.add_intermediate(
        pos=(1, 1), transmit_power_dbm=30, capacity=5, name="nin2"
    )
    _nsi = infra.set_sink(pos=(0, 1), transmit_power_dbm=30, name="nsi")

    overlay = OverlayNetwork()

    # ignore sinr constraints -> 0 datarate requirements
    bso = overlay.add_source(name="bso", datarate=0)
    bin1 = overlay.add_intermediate(requirement=40, name="bin1", datarate=0)
    bin2 = overlay.add_intermediate(requirement=5, name="bin2", datarate=0)
    bsi = overlay.set_sink(name="bsi", datarate=0)

    overlay.add_link(bso, bin1)
    overlay.add_link(bso, bin2)
    overlay.add_link(bin1, bsi)
    overlay.add_link(bin2, bsi)

    embedding = PartialEmbedding(infra, overlay, source_mapping=[(bso, nso)])

    eso = ENode(bso, nso)
    possibilities = embedding.possibilities()
    # bin1 can be embedded in nin1, because 42>=40
    assert (eso, ENode(bin1, nin1), 0) in possibilities
    # but not in nin2 because it does not have enough capacity
    assert (eso, ENode(bin1, nin2), 0) not in possibilities

    # bin2 has less requirements and can be embedded in either one
    assert (eso, ENode(bin2, nin1), 0) in possibilities
    assert (eso, ENode(bin2, nin2), 0) in possibilities

    # embed bin1 in nin1
    assert embedding.take_action(ENode(bso, nso), ENode(bin1, nin1), 0)
    possibilities = embedding.possibilities()

    # pylint:disable=protected-access
    # The easiest way to test this, not too hard to adjust when
    # internals change.
    assert embedding._capacity_used[nin1] == 40

    # which means bin2 can no longer be embedded in it
    assert (eso, ENode(bin2, nin1), 0) not in possibilities
    # while it can still be embedded in nin2
    assert (eso, ENode(bin2, nin2), 0) in possibilities


def test_source_and_sink_capacity_check():
    """Tests that an embedding with invalid source or sink capacity
    cannot be created"""

    infra = InfrastructureNetwork()

    nso = infra.add_source(
        pos=(0, 0), transmit_power_dbm=30, capacity=0, name="nso"
    )
    _nsi = infra.set_sink(
        pos=(1, 0), transmit_power_dbm=30, capacity=0, name="nsi"
    )

    def embedding_fails(overlay):
        source_block = list(overlay.sources)[0]
        print(f"bso is {source_block}")
        failed = False
        try:
            _embedding = PartialEmbedding(
                infra, overlay, source_mapping=[(source_block, nso)]
            )
        except AssertionError as _:
            failed = True
        return failed

    # this is fine
    overlay = OverlayNetwork()
    bso = overlay.add_source(name="bso", requirement=0)
    bsi = overlay.set_sink(name="bin", requirement=0)
    overlay.add_link(bso, bsi)
    assert not embedding_fails(overlay)

    # source requirement not met
    overlay = OverlayNetwork()
    bso = overlay.add_source(name="bso", requirement=1)
    bsi = overlay.set_sink(name="bin", requirement=0)
    overlay.add_link(bso, bsi)
    assert embedding_fails(overlay)

    # sink requirement not met
    overlay = OverlayNetwork()
    bso = overlay.add_source(name="bso", requirement=0)
    bsi = overlay.set_sink(name="bin", requirement=1)
    overlay.add_link(bso, bsi)
    assert embedding_fails(overlay)


def test_non_broadcast_parallel_communications_impossible():
    """Tests that non-broadcast parallel communications *do* affect the
    SINR."""
    infra = InfrastructureNetwork()

    nso1 = infra.add_source(pos=(1, 0), transmit_power_dbm=30, name="nso1")
    nso2 = infra.add_source(pos=(-1, 0), transmit_power_dbm=30, name="nso2")
    nin = infra.add_intermediate(pos=(1, 0), transmit_power_dbm=30, name="nin")
    nsi = infra.set_sink(pos=(2, 0), transmit_power_dbm=30, name="nsi")

    overlay = OverlayNetwork()
    bso1 = overlay.add_source(name="bso1")
    bso2 = overlay.add_source(name="bso2")
    bsi = overlay.set_sink(name="bsi")

    overlay.add_link(bso1, bsi)
    overlay.add_link(bso2, bsi)

    embedding = PartialEmbedding(
        infra, overlay, source_mapping=[(bso1, nso1), (bso2, nso2)]
    )

    # both sources use nin as a relay
    eso1 = ENode(bso1, nso1)
    eso2 = ENode(bso2, nso2)
    esi = ENode(bsi, nsi)
    ein1 = ENode(bso1, nin, bsi)
    ein2 = ENode(bso2, nin, bsi)
    assert embedding.take_action(eso1, ein1, 0)
    assert embedding.take_action(eso2, ein2, 1)

    assert embedding.take_action(ein1, esi, 2)
    assert (ein2, esi, 2) not in embedding.possibilities()


def test_relay_circles_impossible():
    """Tests that each relay node can be taken at most once in a
    path"""
    infra = InfrastructureNetwork()

    N2 = infra.add_source(name="N2", pos=(4.8, 5.7), transmit_power_dbm=29.7)
    N4 = infra.add_intermediate(
        name="N4", pos=(4.7, 8.8), transmit_power_dbm=13.4
    )
    _N1 = infra.set_sink(name="N1", pos=(7.7, 5.2), transmit_power_dbm=22.9)

    overlay = OverlayNetwork()
    B2 = overlay.add_source(name="B2", requirement=0, datarate=0)
    _B5 = overlay.add_intermediate(name="B5", requirement=0, datarate=0)
    B4 = overlay.add_intermediate(name="B4", requirement=0, datarate=0)
    B1 = overlay.set_sink(name="B1", requirement=0, datarate=0)
    overlay.add_link(B2, B4)
    overlay.add_link(B4, B1)

    embedding = PartialEmbedding(infra, overlay, source_mapping=[(B2, N2)])

    eso = ENode(B2, N2)
    ein1 = ENode(B2, N4, B4)
    ein2 = ENode(B2, N2, B4)
    assert embedding.take_action(eso, ein1, 0)
    # n2 was already visited, circle
    assert not embedding.take_action(ein1, ein2, 1)


def test_same_connection_not_possible_twice():
    """Tests that the same connection cannot be taken twice"""
    infra = InfrastructureNetwork()

    N2 = infra.add_source(name="N2", pos=(2.3, 2.2), transmit_power_dbm=26.9)
    N3 = infra.add_intermediate(name="N3", pos=(0, 4), transmit_power_dbm=11)
    _N1 = infra.set_sink(name="N1", pos=(9.4, 9.5), transmit_power_dbm=26.1)

    overlay = OverlayNetwork()
    B2 = overlay.add_source(name="B2")
    B3 = overlay.add_intermediate(name="B3")
    B1 = overlay.set_sink(name="B1")
    overlay.add_link(B2, B1)
    overlay.add_link(B2, B3)
    overlay.add_link(B3, B1)

    embedding = PartialEmbedding(infra, overlay, source_mapping=[(B2, N2)])
    eso = ENode(B2, N2)
    ein = ENode(B2, N3, B1)

    assert embedding.take_action(eso, ein, 0)
    # this connection has already been taken
    assert not embedding.take_action(eso, ein, 1)


def test_half_duplex():
    """Tests that a node cannot send and receive at the same time"""
    infra = InfrastructureNetwork()

    nso = infra.add_source(name="nso", pos=(0, 0), transmit_power_dbm=30)
    ni = infra.add_intermediate(name="ni", pos=(1, 0), transmit_power_dbm=30)
    nsi = infra.set_sink(name="nsi", pos=(2, 0), transmit_power_dbm=30)

    overlay = OverlayNetwork()
    # links have no datarate requirements, so SINR concerns don't apply
    bso = overlay.add_source(name="bso", datarate=0)
    bsi = overlay.set_sink(name="bsi", datarate=0)
    overlay.add_link(bso, bsi)

    embedding = PartialEmbedding(infra, overlay, source_mapping=[(bso, nso)])

    eso = ENode(bso, nso)
    esi = ENode(bsi, nsi)
    ein = ENode(bso, ni, bsi)
    print(embedding.possibilities())
    assert embedding.take_action(eso, ein, 0)
    assert not embedding.take_action(ein, esi, 0)


def test_connection_within_node_always_possible():
    """Tests that a node cannot send and receive at the same time"""
    infra = InfrastructureNetwork()

    nso = infra.add_source(name="nso", pos=(0, 0), transmit_power_dbm=30)
    nsi = infra.set_sink(name="nsi", pos=(2, 0), transmit_power_dbm=30)

    overlay = OverlayNetwork()
    bso = overlay.add_source(name="bso", datarate=0, requirement=0)
    bin_ = overlay.add_intermediate(name="bin", datarate=0, requirement=0)
    bsi = overlay.set_sink(name="bsi", datarate=0, requirement=0)
    overlay.add_link(bso, bin_)
    overlay.add_link(bin_, bsi)

    embedding = PartialEmbedding(infra, overlay, source_mapping=[(bso, nso)])

    eso = ENode(bso, nso)
    ein = ENode(bin_, nsi)
    esi = ENode(bsi, nsi)
    assert embedding.take_action(eso, ein, 0)
    # even though nsi is already receiving in ts 0
    assert embedding.take_action(ein, esi, 0)
