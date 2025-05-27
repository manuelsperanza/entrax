import pytest
import logging
import time
from pathlib import Path
from entrax.utils.log import *
from _pytest.logging import LogCaptureFixture



def test_setup_logger_level_0()->None:
    log_filename = "test_log.log"
    module_name = "test_module"
    level = 0

    logger = setup_logger(module_name=module_name, log_filename=log_filename, level=level)
    assert logger.name == module_name
    assert not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
    assert not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)

def test_setup_logger_level_2()->None:
    log_filename = "test_log.log"
    module_name = "test_module"
    level = 2

    logger = setup_logger(module_name=module_name, log_filename=log_filename, level=level)
    assert logger.name == module_name
    assert any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
    assert any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)


def test_setup_logger_invalid_level()->None:
    log_filename = "test_log.log"
    module_name = "test_module"

    # Test invalid log level
    with pytest.raises(ValueError):
        setup_logger(module_name=module_name, log_filename=log_filename, level=-1)


@pytest.mark.parametrize("seconds, expected", [
    (3665, "Elapsed time: 1h 1m 5.00s"),
    (59, "Elapsed time: 59.00s"),
    (120, "Elapsed time: 2m 0.00s"),
    (7325, "Elapsed time: 2h 2m 5.00s"),
    (0, "Elapsed time: 0.00s")
])
def test_format_elapsed_time(seconds:int, expected:str)->None:
    assert format_elapsed_time(seconds) == expected


@pytest.mark.parametrize("seconds, expected", [
    (0.001, "Elapsed time: 0.00s"),  # Very small time
    (3600, "Elapsed time: 1h 0m 0.00s"),  # Exactly 1 hour
    (86400, "Elapsed time: 24h 0m 0.00s"),  # Exactly 1 day
    (123456.789, "Elapsed time: 34h 17m 36.79s")  # Large time
])
def test_format_elapsed_time_edge_cases(seconds:int, expected:str)->None:
    assert format_elapsed_time(seconds) == expected


def test_write_pcap_log(tmp_path:Path)->None:
    log_path = tmp_path / "pcap_log.txt"
    pcap_index = 1
    pcap_name = "test.pcap"
    start_time = time.perf_counter()
    packet_count = 100
    flows_count = 10

    logger_name = "test_logger"
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(log_path)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    write_pcap_log(logger_name, pcap_index, pcap_name, start_time, packet_count, flows_count)

    with open(log_path, "r") as file:
        content = file.read()

    assert f"{pcap_index} file: {pcap_name}" in content
    assert "Elapsed time:" in content
    assert f"Number of packet analysed: {packet_count}" in content
    assert f"Number of flows found: {flows_count}" in content
    if flows_count != 0:
        assert f"Mean lenght of flows: {packet_count / flows_count}" in content

    logger.removeHandler(handler)
    handler.close()


def test_write_pcap_log_no_flows(tmp_path:Path)->None:
    log_path = tmp_path / "pcap_log.txt"
    pcap_index = 1
    pcap_name = "test.pcap"
    start_time = time.perf_counter()
    packet_count = 100
    flows_count = 0  # No flows

    logger_name = "test_logger"
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(log_path)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    write_pcap_log(logger_name, pcap_index, pcap_name, start_time, packet_count, flows_count)

    with open(log_path, "r") as file:
        content = file.read()

    assert f"{pcap_index} file: {pcap_name}" in content
    assert "Elapsed time:" in content
    assert f"Number of packet analysed: {packet_count}" in content
    assert f"Number of flows found: {flows_count}" in content
    assert "Mean lenght of flows:" not in content  # No mean length if no flows

    logger.removeHandler(handler)
    handler.close()


def test_write_pcap_console(caplog:LogCaptureFixture)->None:
    module_name = "test_logger"
    pcap_path = "test.pcap"
    packet_count = 100
    flows_count = 10

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        write_pcap_console(module_name, pcap_path, packet_count, flows_count)

    assert f"For: {pcap_path}  Number of packets: {packet_count} Number of flows: {flows_count}" in caplog.text


def test_write_pcap_console_empty(caplog:LogCaptureFixture)->None:
    module_name = "test_logger"
    pcap_path = ""
    packet_count = 0
    flows_count = 0

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        write_pcap_console(module_name, pcap_path, packet_count, flows_count)

    assert f"For: {pcap_path}  Number of packets: {packet_count} Number of flows: {flows_count}" in caplog.text


def test_write_pcap_error(caplog:LogCaptureFixture)->None:
    module_name = "test_logger"
    error_message = "Test error message"

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.ERROR)

    with caplog.at_level(logging.ERROR):
        write_error(module_name, error_message)

    assert error_message in caplog.text


def test_write_pcap_error_empty_message(caplog:LogCaptureFixture)->None:
    module_name = "test_logger"
    error_message = ""

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.ERROR)

    with caplog.at_level(logging.ERROR):
        write_error(module_name, error_message)

    assert error_message in caplog.text  # Should handle empty error messages gracefully


def test_write_set_info(caplog:LogCaptureFixture)->None:
    module_name = "test_logger"
    info_message = "Test info message"
    debug_message = "Test debug message"

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        write_set_info(module_name, info_message, debug_message)

    assert info_message in caplog.text
    assert debug_message in caplog.text


def test_write_set_info_empty_messages(caplog:LogCaptureFixture)->None:
    module_name = "test_logger"
    info_message = ""
    debug_message = ""

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        write_set_info(module_name, info_message, debug_message)

    assert info_message in caplog.text
    assert debug_message in caplog.text