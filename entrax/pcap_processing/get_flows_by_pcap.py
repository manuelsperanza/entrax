from pathlib import Path
import time
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional,Any

import pyshark # type: ignore

from entrax.pcap_processing.flow import Flow, read_from_csv
from entrax.utils.log import setup_logger, write_pcap_log, write_pcap_console, write_error
from entrax.utils.IO import get_files_recursively
from entrax.utils.general import get_class
from entrax.utils.constants import Classes

class GetFlowsByPCAP:
    def __init__(self, pcap_path:Path, save_flows_folder_path:Optional[Path]=None, get_classes:bool=False, resume:bool=False, info:int=0):
        self.__pcap_path = pcap_path
        self.__save_folder_path = save_flows_folder_path
        self.__get_classes = get_classes
        self.__resume = resume #############
        now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        self.__log_filename = f"log_{self.__pcap_path.stem}_{now}.txt"
        self.__log_pcap_index = 0
        self.__log_time = time.perf_counter()
        self.__log_packet_count = 0
        self.__log_flows_count = 0

        setup_logger(owner=self, log_filename=self.__log_filename, level=info)

        self.__flows = defaultdict(list)
        for pcap in get_files_recursively(path=self.__pcap_path, suffix_list=['.pcap', '.pcapng']):
            self.__log_pcap_index +=1
            if self.__resume and self.__save_folder_path is not None:
                save_path = self.__save_folder_path / pcap.stem

                if save_path.exists() and save_path.is_dir():
                    log_time = time.perf_counter()
                    label = Classes.NOCLASS
                    if self.__get_classes:
                        label = get_class(path=pcap)

                    flows_paths = get_files_recursively(path=save_path, suffix_list=['.csv'])
                    log_flows_lengt = 0
                    for flow_path in flows_paths:
                        flow = read_from_csv(file_path=flow_path)
                        #if flow.filter():
                        log_flows_lengt+=1
                        self.__flows[label].append(flow)
                    write_pcap_log(owner=self, pcap_index=self.__log_pcap_index, pcap_name=f"Loaded flows from {save_path} corresponding to file {pcap}", start_time=log_time, packet_count=0, flows_count=log_flows_lengt)
                    continue
            try: 
                self.analyse_pcap(pcap_path=pcap, use_ek=True)
            except Exception as e:
                write_error(owner=self, error=f"{str(type(e))}  {e}")
                self.analyse_pcap(pcap_path=pcap, use_ek=False)

        #after the analysis write the total infos
        write_pcap_log(owner=self, pcap_index=self.__log_pcap_index+1, pcap_name="TOTAL", start_time=self.__log_time, packet_count=self.__log_packet_count, flows_count=self.__log_flows_count)
        
    def analyse_pcap(self, pcap_path: Path, use_ek:bool)->None:
        flows:list[Flow] = [] # flows of the current pcap file
        packet_count = 0
        log_time = time.perf_counter()
                
        
        capture = pyshark.FileCapture(pcap_path, use_ek=use_ek, include_raw=False, keep_packets=False, display_filter="ip and not dns", override_prefs={"tcp.check_checksum": "FALSE"})
        #capture = pyshark.FileCapture(pcap_path, include_raw=False, keep_packets=False, display_filter="ip and not dns and (tcp or udp)", override_prefs={"tcp.check_checksum": "FALSE"})
        try:
            for pkt in capture:
                try:
                    packet_count+=1
                    if packet_count % 50000 == 0:
                        write_pcap_console(owner=self, pcap_path=str(pcap_path), packet_count=packet_count, flows_count=len(flows))
                    
                    #if "DNS" in pkt:
                    #    continue
                    #if "IP" not in pkt:
                    #    continue"""
                    if use_ek:
                        ip1 = pkt.ip.src.value #for EK
                        ip2 = pkt.ip.dst.value #for EK
                    else:
                        ip1 = str(pkt.ip.src)
                        ip2 = str(pkt.ip.dst)

                    if "TCP" in pkt:
                        port1 = str(pkt.tcp.srcport)
                        port2 = str(pkt.tcp.dstport)
                        protocol = "TCP"
                    elif "UDP" in pkt:
                        port1 = str(pkt.udp.srcport)
                        port2 = str(pkt.udp.dstport)
                        protocol = "UDP"
                    else:
                        continue
                    key = (ip1, ip2, port1, port2, protocol)
                    
                    if not all(isinstance(x, str) for x in key):
                        bad_types = {type(x).__name__ for x in key if not isinstance(x, str)}
                        raise TypeError(f"All elements must be strings. Found types: {bad_types}")
                    
                    tls_info = self.analyse_encryption(pkt, use_ek)
                    
                    if use_ek:
                        timestamp_str = pkt.frame_info.time.epoch  # Converts EkMultiField to string
                        dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f000Z")
                        timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
                    else:
                        timestamp = pkt.sniff_time.timestamp()
                    added = False
                    for f in flows:
                        if f.add_if_key_is_equal(key=key, timestamp=float(timestamp), lenght=int(pkt.length), tls_info=tls_info):
                            added = True
                            break
                    if not added:
                        new_flow = Flow(key)
                        new_flow.add(timestamp=float(timestamp), lenght=int(pkt.length), is_source=True, tls_info=tls_info)
                        flows.append(new_flow)
                except Exception as e:
                    write_error(owner=self, error=f"{str(type(e))} problem with packet {packet_count}  {e}")
        except pyshark.capture.capture.TSharkCrashException as e:
            write_error(owner=self, error=f"{str(type(e))},TShark error:  {e}")
        capture.close()
        flows = [flow for flow in flows if flow.filter()] # remove the flow for which is not possible to calculate features
        
        write_pcap_log(owner=self, pcap_index=self.__log_pcap_index, pcap_name=str(pcap_path), start_time=log_time, packet_count=packet_count, flows_count=len(flows))
        self.__log_packet_count += packet_count
        self.__log_flows_count += len(flows)
        
        if self.__get_classes:
            label = get_class(path=pcap_path)
        else:
            label = Classes.NOCLASS
        self.__flows[label].extend(flows)     

        if self.__save_folder_path is not None:
            for flow in flows:
                flow.save_to_csv(folder_path=self.__save_folder_path / pcap_path.stem, overwrite=True)

    
    def analyse_encryption(self, pkt:Any, use_ek:bool)->str:
        if use_ek:
            if hasattr(pkt, 'tls'):
                tls_info = "tls"
                if hasattr(pkt.tls, 'record'):
                #if hasattr(pkt.tls, 'content'):
                    record = pkt.tls.record
                    if hasattr(record, 'version'):
                        version = pkt.tls.record.version
                        if isinstance(version, list):
                            v = version[0]
                            if version[1] < v:
                                v = version[1]
                            tls_version = f"{v:04x}"
                        else:
                            tls_version = f"{version:04x}"
                    else:
                        tls_version = "0"
                    if hasattr(record, 'content_type'):
                        content_type = record.content_type
                        if isinstance(content_type, list):
                            content_type = content_type[0]
                        tls_content_type = str(content_type)
                    else:
                        tls_content_type = "0"
                else:
                    tls_version = "0"
                    tls_content_type = "0"
            else:
                tls_info = "0"
                tls_version = "0"
                tls_content_type = "0"
        else:
            if hasattr(pkt, 'tls'):
                tls_info = "tls"
                if hasattr(pkt.tls, 'record_version'):
                    tls_version = pkt.tls.record_version
                else:
                    tls_version = "0"
                
                if hasattr(pkt.tls, 'record_content_type'):
                    tls_content_type = pkt.tls.record_content_type
                else:
                    tls_content_type = "0"
            else:
                tls_info = "0"
                tls_version = "0"
                tls_content_type = "0"
        tls_info = f"{tls_info}-{tls_version}-{tls_content_type}"
        return tls_info
    
    def get_flows(self)-> defaultdict[Classes, list[Flow]]:
        return self.__flows
    


"""for field in dir(pkt.tls):
    if not field.startswith('_'):
        print(f"{field}: {getattr(pkt.tls, field)}")"""