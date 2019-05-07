"""
Some helper function for dealing with wireless signal transmission.
"""

from math import inf, log, sqrt

def subtract_dbm(
        dbm1: float,
        dbm2: float,
):
    """Adds two decibel values"""
    watt1 = dbm_to_watt(dbm1)
    watt2 = dbm_to_watt(dbm2)
    return watt_to_dbm(watt1 - watt2)

def dbm_to_watt(dbm: float):
    """
    Bell is a logarithmic scale to express the ratio of two
    quantities. It is defined as the logarithm of the ratio to base
    10.

    Q = lg P_1/P_2 Bell = 10 * lg P_1/P_2 deciBell

    So if we have a power of 50 dBm (= decibel with respect to 1 mW),
    that means

    50 dBm = 10 * lg P/1mW
    => 5 dBm = lg P - lg 1mW
    => 5 dBm + lg 1mW = lg P
    => 10^(5 dBm + lg 1mW) = P
    => 10^(5 dBm) * 10^(lg 1mW) = P
    => 10^(5 dBm) * 1mW = P
    => P = 10^5 mW = 10^5 * 10^(-3) W
    """

    # 0 cannot be represented as a ratio to 1mW
    if dbm == -inf:
        return 0

    bell = dbm / 10.0
    milliwatts = 10.0 ** bell
    watts = milliwatts / 1000.0
    return watts

def watt_to_dbm(watts: float):
    """
    Converts an amount in watt to a ratio to 1mW, represented in dBm.
    """
    # 0 cannot be represented as a ratio to 1mW
    if watts <= 0:
        return -inf
    milliwatts = watts * 1000.0
    bell = log(milliwatts, 10.0)
    decibel = bell * 10.0
    return decibel

def log_path_loss(
        distance_meters: float,
        loss_exponent: int = 4,
        system_loss: float = 0
    ):
    """
    Returns the approximated path loss (in dB) over a certain distance
    using the simple log-distance model. This approximation is dependent
    on the "loss exponent", which depends on the environment and usually
    is a value between 2 (indicating free space) and 4 (indicating a
    noisy environment like a building). Wikipedia has a more detailed
    description:

    https://en.wikipedia.org/wiki/Path_loss#Loss_exponent
    """
    # decibels relative to 1m
    if distance_meters != 0:
        distance_decibel = 10 * log(distance_meters, 10)
    else:
        # it is mathematically convenient here to define log(0) = -inf
        distance_decibel = -inf
    # loss_exponent = loss increase (decibels) per increase in order of
    # magnitude. E.g. loss_exponent = 3dB means that the signal loses half
    # of its strength for each order of magnitude increase in distance.
    loss = loss_exponent * distance_decibel + system_loss

    # The approximation breaks down once the nodes get too close. Loss
    # can never be negative.
    return max([loss, 0])

def distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
):
    """
    Calculates the euclidean distance between two points in 2d space.
    """
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def power_received(
        distance_meter: float,
        transmit_power_dbm: float,
):
    """
    Calculates the power received at target when source sends at
    full transmission power (based on log path loss model).
    """
    path_loss = log_path_loss(distance_meter)
    # path loss is in dB, so this subtraction is "actually" a division
    return transmit_power_dbm - path_loss

def sinr(
        received_signal_dbm: float,
        received_interference_dbm: float,
        noise_floor_dbm: float,
):
    """
    Calculate the Signal Interference Noise Ratio in decibels.
    That means that

    $10^{sinr/10} . (noise+interference) = signal$

    Should always hold.
    """
    # We need to convert to watts for addition (log scale can only multiply)
    received_noise_watt = dbm_to_watt(received_interference_dbm) \
            + dbm_to_watt(noise_floor_dbm)
    received_noise_dbm = watt_to_dbm(received_noise_watt)

    # Even though it is called Signal Interference Noise *Ratio*, here we have
    # to subtract (not divide) since we are calculating in dBm, which is a
    # logarithmic scale.
    return received_signal_dbm - received_noise_dbm
