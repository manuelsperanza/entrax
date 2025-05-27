import pytest
from pathlib import Path
from entrax.utils.general import get_class
from entrax.utils.constants import Classes

@pytest.mark.parametrize("filename, expected", [
    ("vpn_chat_log.txt", Classes.VPN_CHAT),
    ("vpn_email_report.csv", Classes.VPN_MAIL),
    ("vpn_voipbuster_call.mp3", Classes.VPN_VOIP),
    ("vpn_audio_call.mp3", Classes.VPN_VOIP),
    ("vpn_video_conference.mp4", Classes.VPN_VIDEOCALL),
    ("vpn_ftps_transfer.log", Classes.VPN_FT),
    ("vpn_sftp_backup.tar", Classes.VPN_FT),
    ("vpn_scp_file.txt", Classes.VPN_FT),
    ("vpn_file_share.zip", Classes.VPN_FT),
    ("vpn_netflix_streaming.mkv", Classes.VPN_VIDEO_STREAMING),
    ("vpn_vimeo_watch.mp4", Classes.VPN_VIDEO_STREAMING),
    ("vpn_youtube_clip.webm", Classes.VPN_VIDEO_STREAMING),
    ("vpn_spotify_playlist.mp3", Classes.VPN_AUDIO_STREAMING),
    ("vpn_bittorrent_download.torrent", Classes.VPN_P2P),
    ("random_vpn_log.txt", Classes.VPN),
    ("chat_message.txt", Classes.CHAT),
    ("email_draft.msg", Classes.MAIL),
    ("voip_call.mp3", Classes.VOIP),
    ("audio_recording.wav", Classes.VOIP),
    ("video_meeting.mp4", Classes.VIDEOCALL),
    ("ftps_upload.log", Classes.FT),
    ("sftp_backup.tar", Classes.FT),
    ("scp_transfer.txt", Classes.FT),
    ("file_sharing.zip", Classes.FT),
    ("netflix_movie.mkv", Classes.VIDEO_STREAMING),
    ("vimeo_clip.mp4", Classes.VIDEO_STREAMING),
    ("youtube_video.webm", Classes.VIDEO_STREAMING),
    ("spotify_song.mp3", Classes.AUDIO_STREAMING),
    ("random_file.txt", Classes.FT),
])
def test_get_class(filename:str, expected:Classes)->None:
    path = Path(filename)
    assert get_class(path) == expected