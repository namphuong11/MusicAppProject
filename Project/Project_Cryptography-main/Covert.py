import librosa
import scipy.io.wavfile as wav

def convert_audio(input_path, output_path):
    # Load audio file
    data, sr = librosa.load(input_path, sr=None)

    # Ghi dữ liệu ra file
    wav.write(output_path, sr, data)

# Ví dụ: Cắt từ giây thứ 10 đến giây thứ 20
convert_audio("D:\\Project\\Database\\Dataset\\Joji - Glimpse of Us.mp3", "D:\\Project\\Database\\Dataset\\Joji - Glimpse of Us.wav")