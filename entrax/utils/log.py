import logging
import time
from entrax.utils.IO import DefaultPaths, get_available_filepath
from entrax.utils.general import format_elapsed_time

def get_owner(owner: object) -> str:
    # Returns the class name if a class is provided, otherwise returns the string value.
    # Returns a string in the form "ClassName_id"
    return f"{owner.__class__.__name__}_{id(owner)}"

def write_error(owner: object, error: str) -> None:
    #check if the owner has a logger
    logger = logging.getLogger(get_owner(owner))
    if not logger.hasHandlers():
        raise ValueError("Logger not initialized")
    logger.error(error)

def write_info(owner: object, info: str) -> None:
    logger = logging.getLogger(get_owner(owner))
    if not logger.hasHandlers():
        raise ValueError("Logger not initialized")
    logger.info(info)

def write_debug_message(owner: object, debug_message: str) -> None:
    logger = logging.getLogger(get_owner(owner))
    if not logger.hasHandlers():
        raise ValueError("Logger not initialized")
    logger.debug(debug_message)

def setup_logger(owner: object, log_filename: str, level: int) -> logging.Logger:
    """Creates and configures a logger with separate file logs per class."""
    if level > 2 or level < 0:
        raise ValueError("level not correct")
    log_file = DefaultPaths.LOG.value / log_filename
    log_file = get_available_filepath(log_file)
    if level > 0:
        if not log_file.parent.exists():
            log_file.parent.mkdir(parents=True)
    
    logger = logging.getLogger(get_owner(owner))
    if getattr(logger, "_configured", False):
        return logger
    
    logger.setLevel([logging.WARNING, logging.INFO, logging.DEBUG][level])

    if level > 1:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%H:%M:%S")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    if level > 0:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    elif level == 0:
        logger.addHandler(logging.NullHandler())

    setattr(logger, "_configured", True)
    return logger

def write_pcap_log(owner: object, pcap_index: int, pcap_name: str, start_time: float, packet_count: int, flows_count: int) -> None:
    logger = logging.getLogger(get_owner(owner))
    if not logger.hasHandlers():
        raise ValueError("Logger not initialized")
    elapsed_time = time.perf_counter() - start_time
    message = f"\n{pcap_index} file: {pcap_name}\n"
    message += f"Elapsed time: {format_elapsed_time(elapsed_time)}\n"
    message += f"Number of packet analysed: {packet_count}\n"
    message += f"Number of flows found: {flows_count}\n"
    if flows_count != 0:
        message += f"Mean lenght of flows: {packet_count/flows_count}\n"
    message += "\n"
    logger.info(message)

def write_pcap_console(owner: object, pcap_path: str, packet_count: int, flows_count: int) -> None:
    logger = logging.getLogger(get_owner(owner))
    if not logger.hasHandlers():
        raise ValueError("Logger not initialized")
    message = f"For: {pcap_path}  Number of packets: {packet_count} Number of flows: {flows_count}\n"
    logger.debug(message)

def write_set_info(owner: object, info: str, head: str) -> None:
    logger = logging.getLogger(get_owner(owner))
    if not logger.hasHandlers():
        raise ValueError("Logger not initialized")
    logger.info(info)
    logger.debug(head)
