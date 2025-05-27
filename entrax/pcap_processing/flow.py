from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from enum import Enum

from entrax.utils.constants import FlowData, Features, TlsInfo

class KeyComparison(Enum):
    NOT_EQUAL = 0 #keys not equals to flow key
    EQUAL_AND_DESTINATION = 1 #key equals to flow key but not source
    EQUAL_AND_SOURCE = 2 # key equal to flow key and source

#FLOWDTYPE = {FlowData.TIMESTAMP.value: pd.Series(dtype="float64"), FlowData.SIZE.value:pd.Series(dtype="int32"), FlowData.IS_SOURCE.value:pd.Series(dtype="bool")}
FLOWDTYPE = {FlowData.TIMESTAMP.value:float, FlowData.SIZE.value:int, FlowData.IS_SOURCE.value:bool, FlowData.TLS_INFO.value:str}

class Flow:
    def __init__(self, key:tuple[str,str,str,str,str], data_frame:Optional[pd.DataFrame]=None):
        self.__key = key
        self.__first_timestamp = 0.0
        if data_frame is None:
            #self.__data_frame = pd.DataFrame({FlowData.TIMESTAMP.value: pd.Series(dtype="float64"), FlowData.SIZE.value:pd.Series(dtype="int32"), FlowData.IS_SOURCE.value:pd.Series(dtype="bool")})
            self.__data_frame = pd.DataFrame({col: pd.Series(dtype=FLOWDTYPE[col]) for col in FLOWDTYPE})
        else:
            self.__data_frame = data_frame
    
    def key_is_equal(self, key:tuple[str,str,str,str,str]) -> KeyComparison: # compares keys
        #key[0] #IP src
        #key[1] #IP dst
        #key[2] #port src
        #key[3] #port dst
        #key[4] #Protocol
        if self.__key[4] == key[4]: #Protocol
            if self.__key[0] == key[0] and self.__key[1] == key[1] and self.__key[2] == key[2] and self.__key[3] == key[3]:
                return KeyComparison.EQUAL_AND_SOURCE
            elif self.__key[0] == key[1] and self.__key[1] == key[0] and self.__key[2] == key[3] and self.__key[3] == key[2]:
                return KeyComparison.EQUAL_AND_DESTINATION
            else:
                return KeyComparison.NOT_EQUAL
        else:
            return KeyComparison.NOT_EQUAL

    def add(self, timestamp: float, lenght: int, is_source:bool, tls_info:str)->None: # add the data of a new packet
        if len(self.__data_frame) == 0:
            self.__first_timestamp = timestamp
        #if tls_info == "0-0-0":
        #    tls_info = np.nan
        new_row = {FlowData.TIMESTAMP.value: (timestamp - self.__first_timestamp), FlowData.SIZE.value: lenght, FlowData.IS_SOURCE.value:is_source, FlowData.TLS_INFO.value:tls_info}
        self.__data_frame = pd.concat([self.__data_frame, pd.DataFrame([new_row])], ignore_index=True)  

    def add_if_key_is_equal(self, key:tuple[str,str,str,str,str], timestamp:float, lenght:int, tls_info:str) -> bool :
        """verify if the key is equal to the features key
        if yes it add the packet data to the flow and return True
        else not add and return False"""

        key_res = self.key_is_equal(key)
        if key_res == KeyComparison.EQUAL_AND_SOURCE:
            source = True
        elif key_res == KeyComparison.EQUAL_AND_DESTINATION:
            source = False
        else:
            return False
        self.add(timestamp=timestamp, lenght=lenght, is_source=source, tls_info=tls_info)
        return True
        
    def save_to_csv(self, folder_path:Path, overwrite:bool=False)->None:
        """save the flows data in csv format"""
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        
        filename = self.get_file_name()
        output_file_path = folder_path / filename
        if not output_file_path.exists():
            self.__data_frame.to_csv(output_file_path, index=False, float_format="%.17g")
        else:
            if overwrite:
                self.__data_frame.to_csv(output_file_path, index=False, float_format="%.17g")
    
    def get_file_name(self)-> Path:
        """return the file name of the flow using the key"""
        return Path(f"{self.__key[0]}_{self.__key[1]}_{self.__key[2]}_{self.__key[3]}_{self.__key[4]}.csv")

    def filter(self) -> bool:
        if self.__data_frame.loc[self.__data_frame[FlowData.IS_SOURCE.value] == True].shape[0] < 1:
            return False
        if self.__data_frame.loc[self.__data_frame[FlowData.IS_SOURCE.value] == False].shape[0] < 1:
            return False
        return True

    def get_tls_info(self)->dict[TlsInfo, int]:

        info_dict = {TlsInfo.NOTLS : 0, TlsInfo.TLS:0, TlsInfo.TLS1:0 , TlsInfo.TLS2:0, TlsInfo.TLS3:0}
        infos_str = self.__data_frame[FlowData.TLS_INFO.value]
        for str in infos_str:
            parts = str.split('-')
            if len(parts) != 3:
                raise ValueError(f"Expected 3 parts after parsing, got {len(parts)}. {str}")
            if parts[0] == "0":
                info_dict[TlsInfo.NOTLS] += 1
            elif parts[0] == "tls":
                if parts[1] == "0":
                    info_dict[TlsInfo.TLS] += 1
                elif parts[1] == "0301":
                    info_dict[TlsInfo.TLS1] += 1
                elif parts[1] == "0302":
                    info_dict[TlsInfo.TLS2] += 1
                elif parts[1] == "0303":
                    info_dict[TlsInfo.TLS3] += 1
        return info_dict
    
    def calculate_features(self)->dict[str,float]:        
        flow_times = self.__data_frame[FlowData.TIMESTAMP.value].to_numpy()
        fwd = self.__data_frame.loc[self.__data_frame[FlowData.IS_SOURCE.value] == True] #forward packets        
        bwd = self.__data_frame.loc[self.__data_frame[FlowData.IS_SOURCE.value] == False] #backward packets

        duration = flow_times[-1] #self.__data_frame[FlowData.TIMESTAMP.value].iloc[-1] # The duration of the flow.
        fwdPL = fwd[FlowData.SIZE.value].to_numpy()
        if fwdPL.shape[0] == 0:
            fwdPLMean = 0
            fwdPLMax = 0
            fwdPLMin = 0
            fwdPLStd = 0
            fwdPLTotal = 0
        else:
            fwdPLMean = fwdPL.mean()
            fwdPLMax = fwdPL.max()
            fwdPLMin = fwdPL.min()
            fwdPLStd = fwdPL.std()
            fwdPLTotal = fwdPL.sum()
        bwdPL = bwd[FlowData.SIZE.value].to_numpy()
        if bwdPL.shape[0] == 0:
            bwdPLMean = 0
            bwdPLMax = 0
            bwdPLMin = 0
            bwdPLStd = 0
            bwdPLTotal = 0
        else:
            bwdPLMean = bwdPL.mean()
            bwdPLMax = bwdPL.max()
            bwdPLMin = bwdPL.min()
            bwdPLStd = bwdPL.std()
            bwdPLTotal = bwdPL.sum()

        PL = self.__data_frame[FlowData.SIZE.value].to_numpy()
        PLMean = (fwdPL.shape[0] * fwdPLMean + bwdPL.shape[0] * bwdPLMean) / (fwdPL.shape[0]+bwdPL.shape[0])
        PLMax =  max(fwdPLMax, bwdPLMax)
        PLMin = min(fwdPLMin, bwdPLMin)
        PLStd = PL.std()
        PLTotal = fwdPLTotal + bwdPLTotal

        Bs =  (PLTotal/duration) if duration != 0 else 0  # Flow Bytes per second
        Ps =  (len(PL)/duration) if duration != 0 else 0

        flowiat = np.diff(flow_times)# flow_time[1:] - flow_time[:-1] # Flow Inter Arrival Time, the time between two packets sent in either direction (mean, min, max, std).
        if flowiat.shape[0] == 0:
            flowiatMean = 0
            flowiatMax = 0
            flowiatMin = 0
            flowiatStd = 0
        else:
            flowiatMean = flowiat.mean()
            flowiatMax = flowiat.max()
            flowiatMin = flowiat.min()
            flowiatStd = flowiat.std()

        fwd_time = fwd[FlowData.TIMESTAMP.value].to_numpy()
        fiat = np.diff(fwd_time) #fwd_time[1:] - fwd_time[:-1] # Forward Inter Arrival Time, the time between two packets sent forward direction.
        if fiat.shape[0] == 0:
            fiatMean = 0
            fiatMax = 0
            fiatMin = 0
            fiatStd = 0
            fiatTotal = 0
        else:
            fiatMean = fiat.mean()
            fiatMax = fiat.max()
            fiatMin = fiat.min()
            fiatStd = fiat.std()
            fiatTotal = fiat.sum()

        bwd_time = bwd[FlowData.TIMESTAMP.value].to_numpy()
        biat = np.diff(bwd_time) #bwd_time[1:] - bwd_time[:-1] # Backward Inter Arrival Time, the time between two packets sent backwards (mean, min, max, std).
        if biat.shape[0] == 0:
            biatMean = 0
            biatMax = 0
            biatMin = 0
            biatStd = 0
            biatTotal = 0
        else:
            biatMean = biat.mean()
            biatMax = biat.max()
            biatMin = biat.min()
            biatStd = biat.std()
            biatTotal = biat.sum()

        timeout = 15

        if self.__key[-1] == 'TCP':
            #protocol = "TCP"
            # 1) find the indices where IS_SOURCE flips
            flags = self.__data_frame[FlowData.IS_SOURCE.value].values
            # compare shifted arrays to detect flips
            flip_points = np.nonzero(flags[1:] != flags[:-1])[0] + 1
            # always include the very first packet
            index = np.concatenate(([0], flip_points, [len(flow_times) - 1]))

            # 2) compute the inter-flip intervals
            times = np.diff(flow_times[index])

            # 3) compute the idle flags per-interval with an EWMA-based RTT
            alpha = 7/8
            rtt = timeout
            isIdleIndex = np.empty_like(times, dtype=bool)
            for i, dt in enumerate(times):
                isIdleIndex[i] = (dt > rtt)
                rtt = alpha * rtt + (1 - alpha) * dt

            # 4) expand those per-interval flags to per-packet flags
            #    for each segment, repeat its isIdleIndex value for as many packets are in that segment
            #    Note: segment lengths are the differences between consecutive indices
            segment_lengths = np.diff(index)
            # now tile the boolean values out to each packet in the segment
            isIdle = np.repeat(isIdleIndex, segment_lengths)

        elif self.__key[-1] == 'UDP':
            #protocol = "UDP"
            isIdle = np.where(flowiat > timeout, True, False)

        
        idles = flowiat[isIdle] # The amount of time time a flow was idle before becoming active (mean, min, max, std).
        runs = np.diff(np.concatenate(([False], ~isIdle, [False])).astype(int))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        # Step 2: Sum the corresponding slices
        actives = np.array([flowiat[start:end].sum() for start, end in zip(starts, ends)])

        if idles.shape[0] == 0:
            idleMean = 0
            idleMin = 0
            idleMax = 0
            idleStd = 0
            idleTotal = 0
        else:
            idleMean = idles.mean() 
            idleMin = idles.min()
            idleMax = idles.max()
            idleStd = idles.std()
            idleTotal = idles.sum()

        if actives.shape[0] == 0:
            activeMean = duration
            activeMin = duration
            activeMax = duration
            activeStd = duration
            activeTotal = duration
        else:
            activeMean = actives.mean() # The amount of time time a flow was active before going idle (mean, min, max, std).
            activeMin = actives.min()
            activeMax = actives.max()
            activeStd = actives.std()
            activeTotal = actives.sum()

        return {
            Features.DURATION.value: duration,

            Features.FLOWIATMEAN.value: flowiatMean,
            Features.FLOWIATMAX.value: flowiatMax,
            Features.FLOWIATMIN.value: flowiatMin,
            Features.FLOWIATSTD.value: flowiatStd,

            Features.FIATMEAN.value: fiatMean,
            Features.FIATMAX.value: fiatMax,
            Features.FIATMIN.value: fiatMin,
            Features.FIATSTD.value: fiatStd,
            Features.FIATTOTAL.value: fiatTotal,

            Features.BIATMEAN.value: biatMean,
            Features.BIATMAX.value: biatMax,
            Features.BIATMIN.value: biatMin,
            Features.BIATSTD.value: biatStd,
            Features.BIATTOTAL.value: biatTotal,

            Features.ACTIVEMEAN.value: activeMean,
            Features.ACTIVEMAX.value: activeMax,
            Features.ACTIVEMIN.value: activeMin,
            Features.ACTIVESTD.value: activeStd,
            Features.ACTIVETOTAL.value: activeTotal,

            Features.IDLEMEAN.value: idleMean,
            Features.IDLEMAX.value: idleMax,
            Features.IDLEMIN.value: idleMin,
            Features.IDLESTD.value: idleStd,
            Features.IDLETOTAL.value: idleTotal,

            Features.BS.value: Bs,  # Flow Bytes per second
            Features.PS.value: Ps, # Flow packets per second

            Features.FLOWPN.value: flow_times.shape[0],
            Features.FWDPN.value: fwd.shape[0],
            Features.BWDPN.value: bwd.shape[0],
            
            Features.FWDPLMEAN.value: fwdPLMean,
            Features.FWDPLMAX.value: fwdPLMax,
            Features.FWDPLMIN.value:  fwdPLMin,
            Features.FWDPLSTD.value: fwdPLStd,
            Features.FWDPLTOTAL.value: fwdPLTotal,
            Features.BWDPLMEAN.value: bwdPLMean,
            Features.BWDPLMAX.value: bwdPLMax,
            Features.BWDPLMIN.value: bwdPLMin,
            Features.BWDPLSTD.value: bwdPLStd,
            Features.BWDPLTOTAL.value: bwdPLTotal,

            Features.FLOWPLMEAN.value: PLMean,
            Features.FLOWPLMAX.value: PLMax,
            Features.FLOWPLMIN.value: PLMin,
            Features.FLOWPLSTD.value: PLStd,
            Features.FLOWPLTOTAL.value: PLTotal,
            #Features.PROTOCOL.value : protocol
        }



    def __len__(self)->int:
        return len(self.__data_frame)

    def __str__(self)->str:
        return str(self.__data_frame)
    
def get_key_by_file_path(file_path:Path) -> tuple[str,str,str,str,str]:
        key = tuple(file_path.stem.split('_'))
        if not isinstance(key, tuple):
            raise TypeError(f"Expected tuple, got {type(key).__name__}")
    
        if len(key) != 5:
            raise ValueError(f"Expected 5 elements, got {len(key)}")
    
        if not all(isinstance(x, str) for x in key):
            bad_types = {type(x).__name__ for x in key if not isinstance(x, str)}
            raise TypeError(f"All elements must be strings. Found types: {bad_types}")
        else:
            return key

def read_from_csv(file_path: Path) -> Flow:
    """
    Read a file and create Flow object.
    
    Parameters:
        path (str): The path of the file being read.
    """
    return Flow(get_key_by_file_path(file_path), pd.read_csv(file_path, dtype=FLOWDTYPE))