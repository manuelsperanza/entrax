from pathlib import Path

from entrax.utils.constants import Classes

def format_elapsed_time(seconds:float)-> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours:
        return f"Elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes:
        return f"Elapsed time: {int(minutes)}m {seconds:.2f}s"
    else:
        return f"Elapsed time: {seconds:.2f}s"



def get_class(path: Path) -> Classes:
    filename = path.stem.lower()

    if ('nonvpn' in filename) or ('vpn' not in filename):
        if 'chat' in filename:
            return Classes.CHAT
        elif 'email' in filename:
            return Classes.MAIL
        elif ('voip' in filename) or ('audio' in filename):
            return Classes.VOIP
        elif ('netflix' in filename) or ('vimeo' in filename) or ('youtube' in filename):
            return Classes.VIDEO_STREAMING
        elif 'video' in filename:
            return Classes.VIDEOCALL
        elif ('ftps' in filename) or ('sftp' in filename) or ('scp' in filename) or ('rsync' in filename) or ('file' in filename):
            return Classes.FT
        elif 'spotify' in filename:
            return Classes.AUDIO_STREAMING
        elif ('ssh' in filename) or ('rdp' in filename):
            return Classes.C2
        else:
            if ('nonvpn' in filename):
                return Classes.NONVPN
            else:
                return Classes.NOCLASS

    elif 'vpn' in filename:
        if 'chat' in filename:
            return Classes.VPN_CHAT
        elif 'email' in filename:
            return Classes.VPN_MAIL
        elif ('voip' in filename) or ('audio' in filename):
            return Classes.VPN_VOIP
        elif 'video' in filename:
            return Classes.VPN_VIDEOCALL
        elif ('ftps' in filename) or ('sftp' in filename) or ('scp' in filename) or ('rsync' in filename) or ('file' in filename):
            return Classes.VPN_FT
        elif ('netflix' in filename) or ('vimeo' in filename) or ('youtube' in filename):
            return Classes.VPN_VIDEO_STREAMING
        elif 'spotify' in filename:
            return Classes.VPN_AUDIO_STREAMING
        elif 'bittorrent' in filename:
            return Classes.VPN_P2P
        elif ('ssh' in filename) or ('rdp' in filename):
            return Classes.C2
        else:
            return Classes.VPN
    else: 
        return Classes.NOCLASS