import subprocess
import time

def open_vlc(file_path):
    # Đường dẫn của trình VLC
    vlc_path = 'C:\\VLC\\vlc.exe'  # Thay đổi đường dẫn tùy thuộc vào nơi bạn cài đặt VLC

    # Lệnh mở VLC với đường dẫn của file âm thanh
    command = [vlc_path, file_path]

    # Mở VLC từ chương trình Python
    process = subprocess.Popen(command, shell=True)

    # Chờ một khoảng thời gian và sau đó đóng VLC
    time.sleep(10)
    process.terminate()

file_path = input()
open_vlc(file_path)