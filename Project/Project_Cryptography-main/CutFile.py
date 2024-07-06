import librosa
import scipy.io.wavfile as wav

def cut_audio(input_path, output_path, start_time, end_time):
    # Load audio file
    data, sr = librosa.load(input_path, sr=None)

    # Chuyển đổi thời gian từ giây sang mẫu
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Cắt phần của file âm thanh
    cut_data = data[start_sample:end_sample]

    # Ghi dữ liệu ra file
    wav.write(output_path, sr, cut_data)

# Ví dụ: Cắt từ giây thứ 10 đến giây thứ 20
cut_audio("D:\\Project\\Audio\\Original.wav", "D:\\Project\\Audio\\Test.wav", start_time=0, end_time=2)