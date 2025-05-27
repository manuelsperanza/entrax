from enum import Enum

#Classes
class Classes(Enum):
    CLASS = "Class" #for the dataframe column

    BROWSING = "BROWSING"
    CHAT = "CHAT"
    AUDIO_STREAMING = "AUDIO_STREAMING"
    VIDEO_STREAMING = "VIDEO_STREAMING"
    MAIL = "MAIL"
    VOIP = "VOIP"
    VIDEOCALL = "VIDEO_CALL"
    P2P = "P2P"
    FT = "FT"
    C2 = "C2" #Command and control
    VPN_BROWSING = "VPN-BROWSING"
    VPN_CHAT = "VPN-CHAT"
    VPN_AUDIO_STREAMING = "VPN-AUDIO_STREAMING"
    VPN_VIDEO_STREAMING = "VPN-VIDEO_STREAMING"
    VPN_FT = "VPN-FT"
    VPN_VOIP = "VPN-VOIP"
    VPN_VIDEOCALL = "VPN-VIDEO_CALL"
    VPN_P2P = "VPN-P2P"
    VPN_MAIL = "VPN-MAIL"
    VPNC2 = "VPN-C2"

    VPN = "VPN"
    NONVPN = "NONVPN"

    NOCLASS = "NOCLASS"

class TlsInfo(Enum):
    NOTLS = "noTLS"
    TLS = "TLS"
    TLS1 = "TLSv1.1"
    TLS2 = "TLSv1.2"
    TLS3 = "TLSv1.3"
#Flow Data
class FlowData(Enum):
    TIMESTAMP = "timestamp"
    SIZE = "size"
    IS_SOURCE = "is_source"
    TLS_INFO = "tls_info"

class Features(Enum):
    #Features (calculated on each flow)

    #time based features
    DURATION = 'duration'
    #FLOW IAT time between two packet 
    FLOWIATMEAN = 'flowiatMean'
    FLOWIATMAX = 'flowiatMax'
    FLOWIATMIN = 'flowiatMin'
    FLOWIATSTD = 'flowiatStd'

    # FIAT time between two packet sent forward direction
    FIATMEAN = 'fiatMean' 
    FIATMAX = 'fiatMax'
    FIATMIN = 'fiatMin'
    FIATSTD = 'fiatStd'
    FIATTOTAL = 'fiatTotal'

    # BIAT time between two packet sent backward direction
    BIATMEAN = 'biatMean' 
    BIATMAX = 'biatMax'
    BIATMIN = 'biatMin'
    BIATSTD = 'biatStd'
    BIATTOTAL = 'biatTotal'
    
    ACTIVEMEAN = 'activeMean'
    ACTIVEMAX = 'activeMax'
    ACTIVEMIN = 'activeMin'
    ACTIVESTD = 'activeStd'
    ACTIVETOTAL = 'activeTotal'

    IDLEMEAN = 'idleMean'
    IDLEMAX = 'idleMax'
    IDLEMIN = 'idleMin'
    IDLESTD = 'idleStd'
    IDLETOTAL = 'idleTotal'
    
    BS = 'bs' # number of flow bytes per second
    PS = 'ps' # number of flow packets per second
    #lenght based features
    FLOWPN = 'flowPN' #number of flow Packet
    FWDPN = 'fwdPN' #number of forward Packet
    BWDPN = 'bwdPN' #number of backward Packet

    #FWDPL lenght of forward packets
    FWDPLMEAN = 'fwdPLMean'
    FWDPLMAX = 'fwdPLMax'
    FWDPLMIN = 'fwdPLMin'
    FWDPLSTD = 'fwdPLStd'
    FWDPLTOTAL = 'fwdPLTotal'

    #BWDPL lenght of backward packets
    BWDPLMEAN = 'bwdPLMean'
    BWDPLMAX = 'bwdPLMax'
    BWDPLMIN = 'bwdPLMin'
    BWDPLSTD = 'bwdPLStd'
    BWDPLTOTAL = 'bwdPLTotal'

    #FLOWPL lenght of packets
    FLOWPLMEAN = 'flowPLMean'
    FLOWPLMAX = 'flowPLMax'
    FLOWPLMIN = 'flowPLMin'
    FLOWPLSTD = 'flowPLStd'
    FLOWPLTOTAL = 'flowPLTotal'


    

    #PROTOCOL = "protocol"