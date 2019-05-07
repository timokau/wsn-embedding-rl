"""Test wireless signal utility functions"""

from pytest import approx
from wsignal import (
        dbm_to_watt,
        watt_to_dbm,
)

def test_known_dbm_to_watt_results():
    """
    Tests dbm to watt conversion with some manually verified examples.
    """
    # by definition, 0dbm = 1mW
    assert dbm_to_watt(0) == approx(1e-3)
    assert dbm_to_watt(10) == approx(1e-2)
    assert dbm_to_watt(20) == approx(1e-1)
    assert dbm_to_watt(30) == approx(1)
    assert dbm_to_watt(40) == approx(10)

def test_known_watt_to_dbm_results():
    """
    Tests watt to dbm conversion with some manually verified examples.
    """
    # by definition, 0dbm = 1mW
    assert watt_to_dbm(1e-3) == approx(0)
    assert watt_to_dbm(1e-4) == approx(-10)
    assert watt_to_dbm(1) == approx(30)

def test_db_watt_conversion_reversible():
    """
    Tests that converting from db to watt and back results in the
    original db value.
    """
    db = 10
    assert watt_to_dbm(dbm_to_watt(db)) == approx(db)
