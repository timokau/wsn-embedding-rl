"""Imports marvelo problems and results from csv"""

import os
import re
import csv
import numpy as np

from infrastructure import InfrastructureNetwork
from overlay import OverlayNetwork
from embedding import PartialEmbedding


def csv_to_list(csvfile, sep=","):
    """Parses a csv file into a 2d array, ignoring the header"""
    with open(csvfile, "r") as f:
        lines = list(csv.reader(f, delimiter=sep))
        # skip header
        return lines[1:]


def parse_overlay(blocks_file, links_file, datarate):
    """Reads an overlay in MARVELOs format from file"""
    # read the files
    names = []
    requirements = []
    links = []
    for (_id, srcapp, _port, demand) in csv_to_list(blocks_file):
        names.append(srcapp)
        requirements.append(float(demand))
    for (_id, srcapp, _srcport, dstapp) in csv_to_list(links_file):
        links.append((srcapp, dstapp))

    block_specifications = list(zip(names, requirements))

    # construct the overlay from the gathered info
    overlay = OverlayNetwork()
    for (name, requirement) in block_specifications[:1]:
        overlay.add_source(requirement, datarate, name)
    for (name, requirement) in block_specifications[1:-1]:
        overlay.add_intermediate(requirement, datarate, name)
    for (name, requirement) in block_specifications[-1:]:
        overlay.set_sink(requirement, datarate, name)

    for (source, target) in links:
        overlay.add_link(source, target)

    return overlay


def parse_infra(
    nodes_file,
    sink_source_mapping,
    positions_file,
    source_seed,
    transmit_power,
):
    """Reads an infrastructure definition in MARVELO format from csvs"""
    # read the files
    names = []
    capacities = []
    positions = []
    for (_id, name, capacity) in csv_to_list(nodes_file):
        names.append(name)
        capacities.append(float(capacity))
    for csvline in csv_to_list(positions_file, sep=";"):
        positions.extend([float(pos) for pos in csvline[1:]])

    # the positions are saved in a weird format, probably a reshape
    # mistake when saving
    positions = np.reshape(positions, (-1, 2))
    specs = list(zip(names, capacities, positions))

    # unfortunately the source and sink are not indicated in the csv
    # files, so we have to re-generate them
    (sink_idx, source_idx) = sink_source_mapping[(source_seed, len(specs))]
    if source_idx == sink_idx:
        return None

    # make sure source is always first, sink always last
    specs[0], specs[source_idx] = specs[source_idx], specs[0]
    specs[-1], specs[sink_idx] = specs[sink_idx], specs[-1]

    # construct the infrastructure from the gathered info
    infra = InfrastructureNetwork()
    for (name, capacity, pos) in specs[:1]:
        infra.add_source(pos, transmit_power, capacity, name)
    for (name, capacity, pos) in specs[1:-1]:
        infra.add_intermediate(pos, transmit_power, capacity, name)
    for (name, capacity, pos) in specs[-1:]:
        infra.set_sink(pos, transmit_power, capacity, name)

    return infra


# pylint: disable=too-many-arguments
def parse_embedding(
    nodes_file,
    sink_source_mapping,
    positions_file,
    source_seed,
    blocks_file,
    links_file,
    transmit_power,
    datarate,
):
    """Reads a problem instance in MARVELO format from csv files"""
    infra = parse_infra(
        nodes_file,
        sink_source_mapping,
        positions_file,
        source_seed,
        transmit_power,
    )
    if infra is None:
        return None
    overlay = parse_overlay(blocks_file, links_file, datarate)

    # otherwise the mapping wouldn't be specified
    assert len(overlay.sources) == 1
    assert len(infra.sources) == 1
    overlay_source = list(overlay.sources)[0]
    infra_source = list(infra.sources)[0]

    source_mapping = [(overlay_source, infra_source)]
    return PartialEmbedding(infra, overlay, source_mapping)


def load_from_dir(basedir):
    """Loads a set of results from a directory with an assumed format"""
    datarate = 20  # at bitrate 1 equivalent to SINRth 20 linear
    # can't fix this before I get a reply
    # pylint: disable=fixme
    transmit_power = 30  # FIXME get correct value from Haitham

    results_dir = f"{basedir}/resultsTxt"
    param_dir = f"{basedir}/param"

    result_name_pat = re.compile(r"n(\d+)b(\d+)s(\d+).txt")

    # pre-generated with the python2 RNG by simply setting the seed and
    # then taking random.choice(range(nr_nodes)) twice
    sink_source_mapping = dict()
    sink_source_file = f"{basedir}/sink_source_mapping.csv"
    for (seed, nodes, sink_idx, source_idx) in csv_to_list(sink_source_file):
        sink_source_mapping[(int(seed), int(nodes))] = (
            int(sink_idx),
            int(source_idx),
        )

    result = []
    for result_file in os.listdir(results_dir):
        match = result_name_pat.match(result_file)
        nodes = int(match.group(1))
        blocks = int(match.group(2))
        seed = int(match.group(3))
        marvelo_result = int(
            float(open(f"{results_dir}/{result_file}").read())
        )
        info = (nodes, blocks, seed)

        nodes_file = f"{param_dir}/n{nodes}{seed}.csv"
        # arcsfile = f'{param_dir}/a{nodes}{seed}.csv'
        links_file = f"{param_dir}/chain_linear_{blocks}.csv"
        blocks_file = f"{param_dir}/b{blocks}{seed}.csv"
        positions_file = f"{param_dir}/pos{nodes}{seed}.csv"

        embedding = parse_embedding(
            nodes_file,
            sink_source_mapping,
            positions_file,
            seed,
            blocks_file,
            links_file,
            transmit_power,
            datarate,
        )
        result.append((embedding, marvelo_result, info))
    return result


def main():
    """Some testing"""
    count = 0
    other = 0
    for (embedding, result, _info) in load_from_dir("marvelo_data"):
        if embedding is not None:
            count += 1
            print(result)
        else:
            other += 1
    # 1047, 152
    print(count, other)


if __name__ == "__main__":
    main()
