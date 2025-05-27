import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from entrax.pcap_processing.flow import Flow, KeyComparison, read_from_csv, get_key_by_file_path
from entrax.utils.constants import FlowData, TlsInfo, Features

def test_key_comparison_equal_source_and_dest():
    key1 = ("a","b","c","d","TCP")
    key2 = ("a","b","c","d","TCP")
    key3 = ("b","a","d","c","TCP")
    f = Flow(key1)
    assert f.key_is_equal(key2) == KeyComparison.EQUAL_AND_SOURCE
    assert f.key_is_equal(key3) == KeyComparison.EQUAL_AND_DESTINATION
    assert f.key_is_equal(("x","y","z","w","UDP")) == KeyComparison.NOT_EQUAL

def test_add_and_internal_dataframe():
    key = ("1","2","3","4","P")
    f = Flow(key)
    f.add(1.0, 10, True, "tls-0301-xx")
    f.add(2.0, 20, False, "0-0-0")
    df = f._Flow__data_frame
    assert df.shape[0] == 2
    assert df.iloc[0][FlowData.TIMESTAMP.value] == 0.0
    assert df.iloc[1][FlowData.TIMESTAMP.value] == pytest.approx(1.0)

def test_add_if_key_is_equal_returns_and_adds():
    key1 = ("1","2","3","4","P")
    f = Flow(key1)
    # matching key
    assert f.add_if_key_is_equal(key1, 5.0, 15, "tls-0301-yy")
    assert f._Flow__data_frame.shape[0] == 1
    # non-matching key
    assert not f.add_if_key_is_equal(("x","y","z","w","P"), 6.0, 30, "tls-0302-zz")

def test_filter_behavior():
    key = ("1","2","3","4","P")
    f = Flow(key)
    # no packets
    assert not f.filter()
    # only source
    f.add(0.0,1,True,"0-0-0")
    assert not f.filter()
    # only dest
    f2 = Flow(key)
    f2.add(0.0,1,False,"0-0-0")
    assert not f2.filter()
    # both
    f.add(1.0,1,False,"0-0-0")
    assert f.filter()

def test_get_tls_info_counts_and_error():
    key = ("1","2","3","4","P")
    f = Flow(key)
    # normal entries with various versions
    f.add(0.0,1,True,"tls-0-0")
    f.add(1.0,1,False,"tls-0301-00")
    f.add(2.0,1,True,"tls-0302-00")
    f.add(3.0,1,False,"tls-0303-00")
    f.add(4.0,1,True,"0-0-0")
    info = f.get_tls_info()
    assert info[TlsInfo.TLS] == 1
    assert info[TlsInfo.TLS1] == 1
    assert info[TlsInfo.TLS2] == 1
    assert info[TlsInfo.TLS3] == 1
    assert info[TlsInfo.NOTLS] == 1
    # invalid string format
    f2 = Flow(key)
    f2.add(0.0,1,True,"invalid-string")
    with pytest.raises(ValueError):
        f2.get_tls_info()

def test_save_and_read_csv(tmp_path):
    key = ("1","2","3","4","P")
    f = Flow(key)
    f.add(0.0, 5, True, "0-0-0")
    f.add(1.0, 10, False, "tls-0303-00")
    # save and read
    f.save_to_csv(tmp_path, overwrite=True)
    file = tmp_path / f.get_file_name()
    assert file.exists()
    f2 = read_from_csv(file)
    pd.testing.assert_frame_equal(f._Flow__data_frame, f2._Flow__data_frame)

def test_get_key_by_file_path():
    key = ("a","b","c","d","P")
    path = Path("some/path/a_b_c_d_P.csv")
    assert get_key_by_file_path(path) == key

def test_calculate_features_basic():
    key = ("1","2","3","4","TCP")
    f = Flow(key)
    # add packets
    f.add(0.0, 10, True, "0-0-0")
    f.add(1.0, 20, True, "0-0-0")
    f.add(2.5, 30, False, "0-0-0")
    res = f.calculate_features()
    # duration
    assert res[Features.DURATION.value] == pytest.approx(2.5)
    # packet counts
    assert res[Features.FWDPNTOTAL.value] == 2
    assert res[Features.BWDPNTOTAL.value] == 1
    # payload totals
    assert res[Features.FWDPLTOTAL.value] == 30
    assert res[Features.BWDPLTOTAL.value] == 30
    # packets per second
    assert res[Features.PS.value] == pytest.approx(3/2.5)
