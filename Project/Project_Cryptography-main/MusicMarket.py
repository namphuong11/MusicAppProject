import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import customtkinter as ctk
import threading
import pygame
import time
import os
import scipy.io.wavfile
import librosa
import numpy as np
import math
import random
import mysql.connector
import base64
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="namphuong123",
  database="test"
)

# Initialize pygame mixer
pygame.mixer.init()

# Store the current position of the music
current_position = 0
paused = False
selected_folder_path = ""  # Store the selected folder path

# Initialize global variables to store the paths
path = ""
pathShareKey = ""
pathInitialVector = ""
pathOutput = ""
songslt = ""
songsltpath = ""


def update_progress():
    global current_position
    while True:
        if pygame.mixer.music.get_busy() and not paused:
            current_position = pygame.mixer.music.get_pos() / 1000
            pbar["value"] = current_position

            # Check if the current song has reached its maximum duration
            if current_position >= pbar["maximum"]:
                stop_music()  # Stop the music playback
                pbar["value"] = 0  # Reset the pbar

            window.update()
        time.sleep(0.1)

# Create a thread to update the progress bar
pt = threading.Thread(target=update_progress)
pt.daemon = True
pt.start()

def select_music_folder():
    global selected_folder_path
    selected_folder_path = "C:/Drive/code/N2/MMH/Project/nhac/encrypted"
    if selected_folder_path:
        lbox.delete(0, tk.END)
        for filename in os.listdir(selected_folder_path):
            if filename.endswith(".wav"):
                lbox.insert(tk.END, filename)  # Insert only the filename, not the full path

def previous_song():
    if len(lbox.curselection()) > 0:
        current_index = lbox.curselection()[0]
        if current_index > 0:
            lbox.selection_clear(0, tk.END)
            lbox.selection_set(current_index - 1)
            play_selected_song()

def next_song():
    if len(lbox.curselection()) > 0:
        current_index = lbox.curselection()[0]
        if current_index < lbox.size() - 1:
            lbox.selection_clear(0, tk.END)
            lbox.selection_set(current_index + 1)
            play_selected_song()

def play_music():
    global paused
    if paused:
        # If the music is paused, unpause it
        pygame.mixer.music.unpause()
        paused = False
    else:
        # If the music is not paused, play the selected song
        play_selected_song()

def play_selected_song():
    global current_position, paused, songslt, songsltpath
    if len(lbox.curselection()) > 0:
        current_index = lbox.curselection()[0]
        selected_song = lbox.get(current_index)
        songslt = selected_song
        full_path = os.path.join(selected_folder_path, selected_song)  # Add the full path again
        songsltpath = full_path
        pygame.mixer.music.load(full_path)  # Load the selected song
        pygame.mixer.music.play(start=current_position)  # Play the song from the current position
        paused = False
        song_duration = pygame.mixer.Sound(full_path).get_length()
        pbar["maximum"] = song_duration

def pause_music():
    global paused
    # Pause the currently playing music
    pygame.mixer.music.pause()
    paused = True

def stop_music():
    global paused
    # Stop the currently playing music and reset the progress bar
    pygame.mixer.music.stop()
    paused = False

def choose_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)
    entry.focus_set()  # Ensure the focus is set back to the entry field

# Danh sách để lưu tham chiếu tới các ô nhập liệu
entries = []

def show_text_boxes():
    global entries
    entries = []  # Reset danh sách các ô nhập liệu

    # Create a new window to hold the text boxes
    text_box_window = tk.Toplevel(window)
    text_box_window.title("Input Text Boxes")
    text_box_window.geometry("800x400")  # Increased size of the form

    # Create and place the text boxes with corresponding buttons
    for i in range(2):
        entry = tk.Entry(text_box_window, width=40, font=("TkDefaultFont", 16))
        entry.grid(row=i, column=0, padx=10, pady=10)
        button = ctk.CTkButton(text_box_window, text="Choose File", command=lambda e=entry: choose_file(e),
                               font=("TkDefaultFont", 14))
        button.grid(row=i, column=1, padx=10, pady=10)
        entries.append(entry)  # Lưu tham chiếu tới ô nhập liệu

    # Create Encrypt and Decrypt buttons
    btn_encrypt = ctk.CTkButton(text_box_window, text="UPLOAD", command=encrypt, font=("TkDefaultFont", 18))
    btn_encrypt.grid(row=5, column=0, pady=20)

def get_entries_values():
    values = [entry.get() for entry in entries]
    return values

def Input(filename):
  # Đọc dữ liệu âm thanh với kiểu dữ liệu là int16
  data, sample_rate = librosa.load(filename, sr=None, dtype=np.float32)
    
  # Chuyển đổi dữ liệu âm thanh thành kiểu int16
  data_int16 = (data * np.iinfo(np.int16).max).astype(np.int16)
        
  return data_int16, sample_rate

# Hàm ghi dữ liệu âm thanh ra file
def Output(filename, sample_rate, data):
  # Mở file âm thanh để ghi
  with open(filename, "wb") as f:
    # Ghi dữ liệu âm thanh
    scipy.io.wavfile.write(f, sample_rate, data)  

# Tạo random ra hai mảng ShareKey và IntialVector với 128 bits       
def Generation():
  ShareKey = [random.randint(0, 1) for i in range(128)]
  IntialVector = [random.randint(0, 1) for i in range(128)]
  return "".join(str(bit) for bit in ShareKey), "".join(str(bit) for bit in IntialVector)

# Hàm chuẩn hoá từ dạng binary sang dạng hex
def Hex(bin_string):
  return format(int(bin_string, 2), "02x")

# Hàm dời qua trái số bit cho trước
def left_rotate(x, n):
  if isinstance(x, str):
    x = int(x, 2)
  n %= 128  # Đảm bảo n không vượt quá 128
  result = ((x << n) | (x >> (128 - n))) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
  return bin(result)[2:].zfill(128)

# Tạo ra Sine Cosine Sequence
def SineCosineChaoticMap(x0, m, n):
  # Khởi tạo mảng kết quả
  SineCosineSequence = np.zeros(n)

  # Gán giá trị hạt giống ban đầu
  SineCosineSequence[0] = x0

  # Tính các giá trị tiếp theo của map
  for i in range(1, n):
    A = np.sin(-m * SineCosineSequence[i - 1] + pow(SineCosineSequence[i - 1], 3) - m * np.sin(SineCosineSequence[i - 1]))
    B = np.cos(-m * SineCosineSequence[i - 1] + pow(SineCosineSequence[i - 1], 3) - m * np.sin(SineCosineSequence[i - 1]))
    SineCosineSequence[i] = abs(abs(A) - abs(B))

  return SineCosineSequence

# Tạo ra Logistic Sine Cosine Sequence
def LogisticSineCosine(x0, r, n):
  # Khởi tạo mảng kết quả
  LogisticSineCosineSequence = np.zeros(n)

  # Gán giá trị hạt giống ban đầu
  LogisticSineCosineSequence[0] = x0
  
  # Tạo biến pi
  pi = math.pi

  # Tính các giá trị tiếp theo của map
  for i in range(1, n):
    LogisticSineCosineSequence[i] = np.cos(pi * (4 * r * LogisticSineCosineSequence[i - 1] * (1 - LogisticSineCosineSequence[i-1]) + (1 - r) * np.sin(pi * LogisticSineCosineSequence[i-1]) - 0.5))

  return LogisticSineCosineSequence

def Initial_Seq(IV3, SineCosineSequence, SequenceSize):
  arr = np.zeros(SequenceSize, dtype = np.bool_)
  outputarr = np.zeros(SequenceSize, dtype='uint8')
  inc = 1 # Used for creating different value of the value_data
  flag = 0
  while flag < SequenceSize:
    value_data = int(IV3 * inc * SineCosineSequence[inc]) % SequenceSize
    inc = inc + 1 # Increase inc to create diffirent values of the value_data
    if arr[value_data] == False:
      outputarr[flag - 1] = value_data
      flag = flag + 1
      arr[value_data] = True
  return outputarr 

def Permutation(Audio, IV3, SineCosineSequence):
  Length = len(Audio)
  rem = Length % 16 # used for calculating the total number of the variables SequenceSize
  div = int(Length / 16) # used for calculating the remaining number    
      
  # the 16 different dynamic sequence for the permutation process 
  InitialSequence1 = Initial_Seq(IV3, SineCosineSequence, 16) 
  InitialSequence2 = Initial_Seq(IV3, SineCosineSequence, rem)  
        
  audio_pos = 0
  output = np.array([], dtype = np.int16)
  Permutated_Val = np.zeros(16, dtype = np.int16)
      
  for i in range(div):    
    # Rotate the variable InitialSequence1 by I3 * (i + 1) % 16 and store in variable Sequence
    Sequence = np.roll(InitialSequence1, IV3 * (i + 1) % 16)
    Audio_16_val = Audio[audio_pos : audio_pos + 16]
    
    # Take 16 values of the audio data, and then permute these values according to the Sequence variable
    for j in range(16):
      Permutated_Val[j] = Audio_16_val[Sequence[j]] 
    output = np.concatenate((output, Permutated_Val))
    audio_pos = audio_pos + 16
        
  Audio_16_val = Audio[audio_pos:audio_pos + rem]  
  Permutated_Val1 = np.zeros(rem, dtype = np.int16)
      
  for i in range(rem):
    Permutated_Val1[i] = Audio_16_val[InitialSequence2[i]]
  output = np.concatenate((output, Permutated_Val1))

  return output

def int16_array_to_binary16(array):
    # Chuyển đổi mảng int16 sang chuỗi nhị phân 16 bit
    binary_array = [np.binary_repr(x, width=16) for x in array]
    return binary_array

def Binary_to_DNA_Seq(rule, binary_arr):
  encoding_rules = {
    0: ['A', 'C', 'T', 'G'],
    1: ['A', 'G', 'T', 'C'],
    2: ['T', 'C', 'A', 'G'],
    3: ['T', 'G', 'A', 'C'],
    4: ['C', 'A', 'G', 'T'],
    5: ['C', 'T', 'G', 'A'],
    6: ['G', 'A', 'C', 'T'],
    7: ['G', 'T', 'C', 'A']
  }
    
  # Chuyển đổi binary_arr thành chuỗi DNA
  dna_seq_arr = []
  for binary_str in binary_arr:
    dna_seq = ''
    for i in range(0, len(binary_str), 2):
      index = int(binary_str[i:i+2], 2)
      dna_seq += encoding_rules[rule][index]
    dna_seq_arr.append(dna_seq)
    
  return dna_seq_arr

def DNA_addition(dnaseq_arr_1, dnaseq_arr_2):
    # Tạo bảng cộng
    addition_table = {
        'A': ['A', 'C', 'G', 'T'],
        'C': ['C', 'G', 'T', 'A'],
        'G': ['G', 'T', 'A', 'C'],
        'T': ['T', 'A', 'C', 'G']
    }

    # Thực hiện phép cộng
    add_dna_arr = []
    for dna1, dna2 in zip(dnaseq_arr_1, dnaseq_arr_2):
        result = ''
        for base1, base2 in zip(dna1, dna2):
            # Chuyển đổi base2 thành số tương ứng
            tmp2 = 'ACGT'.index(base2)

            # Thực hiện cộng và thêm vào kết quả
            result += addition_table[base1][tmp2]
        add_dna_arr.append(result)

    return add_dna_arr

def DNA_Seq_to_Binary(rule, dna_seq_arr):
    # Tạo bảng giải mã
    decoding_rules = {
        0: {'A': '00', 'C': '01', 'T': '10', 'G': '11'},
        1: {'A': '00', 'G': '01', 'T': '10', 'C': '11'},
        2: {'T': '00', 'C': '01', 'A': '10', 'G': '11'},
        3: {'T': '00', 'G': '01', 'A': '10', 'C': '11'},
        4: {'C': '00', 'A': '01', 'G': '10', 'T': '11'},
        5: {'C': '00', 'T': '01', 'G': '10', 'A': '11'},
        6: {'G': '00', 'A': '01', 'C': '10', 'T': '11'},
        7: {'G': '00', 'T': '01', 'C': '10', 'A': '11'}
    }

    # Chuyển đổi chuỗi DNA thành chuỗi nhị phân
    binary_arr = []
    for dna_seq in dna_seq_arr:
        binary_str = ''
        for dna in dna_seq:
            binary_str += decoding_rules[rule][dna]
        binary_arr.append(binary_str)

    return binary_arr

def dna_apply(Key, binary_audio, binary_chaosValue1, binary_chaosValue2):
  rule = Key % 8
  dnaseq_audio = Binary_to_DNA_Seq(rule, binary_audio)
  dnaseq_val1 = Binary_to_DNA_Seq(rule, binary_chaosValue1)
  dnaseq_val2 = Binary_to_DNA_Seq(rule, binary_chaosValue2)
  
  AddValue1 = DNA_addition(dnaseq_audio, dnaseq_val1)  
  Result = DNA_addition(AddValue1, dnaseq_val2)
  
  dnaseq_output = DNA_Seq_to_Binary(rule, Result)
  return dnaseq_output

def generate_encryption_key(Length, pathShareKey, pathIntiialVector):
    # Tạo ShareKey (SecretKey) và InitialVector một cách ngẫu nhiên
    ShareKey, InitialVector = Generation()
    
    with open(pathShareKey, 'wb') as f:
      f.write(bytes(ShareKey, 'utf-8'))

    with open(pathIntiialVector, 'wb') as f:
      f.write(bytes(InitialVector, 'utf-8'))
    
    InitialVector_hex = Hex(InitialVector)

    I0 = int(InitialVector_hex[0:8], 16)
    I1 = int(InitialVector_hex[8:16], 16)
    I2 = int(InitialVector_hex[16:24], 16)
    I3 = int(InitialVector_hex[24:32], 16)

    IP_SC = 1 / (np.floor(int(ShareKey, 2)) + 1)
    CP_SC = 2.2 + ((I2 ^ I3) % 5)
    IP_LSC = 1 / (np.floor(int(InitialVector, 2)) + 1)
    CP_LSC = 1 / (np.floor(I3) + 1)

    val = I0 % 64
    IntermediateKey1 = ShareKey[val:val + 32]
    d = (2 * math.floor(np.floor(int(InitialVector, 2)) / 2)) + 1
    Tmp_Key = left_rotate(ShareKey, d)
    val1 = I1 % 64
    IntermediateKey2 = Tmp_Key[val1:val1 + 32]

    # CHAOTIC MAP GENERATION
    SineCosineSequence = SineCosineChaoticMap(IP_SC, CP_SC, Length)
    LogisticSineCosineSequence = LogisticSineCosine(IP_LSC, CP_LSC, Length)

    return InitialVector, IntermediateKey1, IntermediateKey2, SineCosineSequence, LogisticSineCosineSequence, I0, I1, I2, I3

def encrypt_song(path, pathShareKey, pathIntiialVector):
    # Đọc file âm thanh
    Audio, sample_rate = Input(path)
    Length = len(Audio)

    # Tạo mảng rỗng kiểu int16
    CipherVoice = np.zeros(Length, dtype=np.int16)

    # Tạo key cho mỗi bài hát
    InitialVector, IntermediateKey1, IntermediateKey2, SineCosineSequence, LogisticSineCosineSequence , I0, I1, I2, I3 = generate_encryption_key(Length, pathShareKey, pathIntiialVector)

    # PERMUTATION PHASE
    Permutated_Audio = Permutation(Audio, I3, SineCosineSequence)

    # DIFFUSION PHASE
    IntermediateKey1_int = int(IntermediateKey1, 2)
    IntermediateKey2_int = int(IntermediateKey2, 2)

    CM1 = ((IntermediateKey1_int * SineCosineSequence) / pow(2, 16)).astype(np.int16)
    CM2 = ((IntermediateKey2_int * LogisticSineCosineSequence) / pow(2, 16)).astype(np.int16)

    binary_audio = int16_array_to_binary16(Permutated_Audio)
    binary_val1 = int16_array_to_binary16(CM1)
    binary_val2 = int16_array_to_binary16(CM2)

    DNA_data = dna_apply(IntermediateKey2_int, binary_audio, binary_val1, binary_val2)

    # DYNAMIC SEQUENCE GENERATION
    I01 = np.zeros(Length, dtype=np.int16)
    I02 = np.zeros(Length, dtype=np.int16)

    for i in range(Length):
        I01[i] = np.int16((I1 * SineCosineSequence[i]))
        I02[i] = np.int16((I2 * LogisticSineCosineSequence[i]))

    DNA_Voice_data = np.zeros(Length, dtype=np.int16)
    for i in range(Length):
        DNA_Voice_data[i] = np.array(int(DNA_data[i], 2), dtype=np.uint16)

    for i in range(Length):
        val = (I0 * (i + 1)) % 2
        if val == 0:
            CipherVoice[i] = np.int16((DNA_Voice_data[i] + I01[i] * (i + 1)) % pow(2, 16))
        else:
            CipherVoice[i] = np.int16((DNA_Voice_data[i] + I02[i] * (i + 1)) % pow(2, 16))

    return CipherVoice, sample_rate, InitialVector


def encrypt():
    global path, pathShareKey, pathInitialVector, pathOutput
    folder_path = "C:/Drive/code/N2/MMH/Project/nhac"
    encrypt_path = "C:/Drive/code/N2/MMH/Project/nhac/encrypted"

# Kiểm tra xem thư mục đã tồn tại chưa
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(encrypt_path):
        os.makedirs(encrypt_path)  # Tạo thư mục nếu chưa tồn tại

    # Tên của tập tin bạn muốn tạo
    file_name = "key.txt"
    file_name1 = "vector.txt"

    # Đường dẫn đến tập tin
    file_path = os.path.join(folder_path, file_name)
    file_path1 = os.path.join(folder_path, file_name1)


    # Mở tập tin để tạo mới
    with open(file_path, "w") as file:
        file.write("")
    with open(file_path1, "w") as file:
        file.write("")
    pathShareKey = file_path
    pathInitialVector = file_path1
    # Lấy giá trị từ các ô nhập liệu

    # Mã hóa bài hát
    values = get_entries_values()
    if len(values) == 2:
        path, file_name2 = values
        file_path2 = os.path.join(encrypt_path, file_name2)
        with open(file_path2, "w") as file:
            file.write("")
        pathOutput = file_path2
        CipherVoice, sample_rate, InitialVector = encrypt_song(path, pathShareKey, pathInitialVector)
        Output(pathOutput, sample_rate, CipherVoice)
        with open(pathShareKey, 'r', encoding='utf-8') as file:
            key = file.read()
        with open(pathInitialVector, 'r', encoding='utf-8') as file:
            vector = file.read()
        mycursor = mydb.cursor()
        combined_binary = key + vector
        encoded = base64.b64encode(combined_binary.encode()).decode()
        sql = "INSERT INTO generation (nameFile, File) VALUES (%s, %s)"
        val = (file_name2, encoded)
        mycursor.execute(sql, val)
        mydb.commit()
        # os.remove(pathShareKey)
        # os.remove(pathInitialVector)
    else:
        messagebox.showinfo("Thông báo", "Không đủ giá trị từ các ô nhập liệu!")

    # Ví dụ: Hiển thị một hộp thoại thông báo
    messagebox.showinfo("Thông báo", "Dữ liệu đã được mã hóa thành công!")
    os.remove(pathShareKey)
    os.remove(pathInitialVector)

def extract_numbers(input_string):
    try:
        # Kiểm tra nếu đầu vào không phải là chuỗi
        if not isinstance(input_string, str):
            input_string = str(input_string)
        
        return ''.join([char for char in input_string if char.isdigit()])
    except Exception as e:
        print(f"Error: {e}")
        return ""

def ReadFromFile():
    global mydb
    cursor = mydb.cursor()

    # Giá trị của biến selected
    selected = songslt

    # Câu truy vấn SQL
    query = "SELECT `File` FROM generation WHERE `nameFile` = %s"

    # Thực hiện truy vấn
    cursor.execute(query, (selected,))

    # Lấy kết quả truy vấn
    result = cursor.fetchone()

    if result:
        file_content = result[0]  # Trích xuất giá trị từ tuple
        decoded_binary = base64.b64decode(file_content.encode())  # Giải mã từ base64
        key_binary = decoded_binary[:len(decoded_binary)//2]  # Chia nửa đầu
        iv_binary = decoded_binary[len(decoded_binary)//2:]  # Chia nửa sau

        # Gán giá trị vào biến ShareKey và InitialVector nếu tìm thấy kết quả
        ShareKey1 = extract_numbers(key_binary)
        InitialVector1 = extract_numbers(iv_binary)
    else:
        ShareKey1 = None
        InitialVector1 = None
    # Đóng con trỏ
    cursor.close()
    return ShareKey1, InitialVector1

def close_db_connection():
    global mydb
    mydb.close()


def Reverse_Permutation(Permutated_Audio, IV3, SineCosineSequence):
    Length = len(Permutated_Audio)
    rem = Length % 16
    div = int(Length / 16)

    InitialSequence1 = Initial_Seq(IV3, SineCosineSequence, 16)
    InitialSequence2 = Initial_Seq(IV3, SineCosineSequence, rem)

    audio_pos = 0
    output = np.array([], dtype=np.int16)
    Audio_Val = np.zeros(16, dtype=np.int16)

    for i in range(div):
        Sequence = np.roll(InitialSequence1, IV3 * (i + 1) % 16)
        Permutated_Audio_16_val = Permutated_Audio[audio_pos: audio_pos + 16]

        for j in range(16):
            Audio_Val[Sequence[j]] = Permutated_Audio_16_val[j]

        output = np.concatenate((output, Audio_Val))
        audio_pos = audio_pos + 16

    Permutated_Audio_16_val = Permutated_Audio[audio_pos: audio_pos + rem]
    Audio_Val1 = np.zeros(rem, dtype=np.int16)

    for i in range(rem):
        Audio_Val1[InitialSequence2[i]] = Permutated_Audio_16_val[i]

    output = np.concatenate((output, Audio_Val1))

    return output

def binary_array_to_int16(binary_array):
    # Chuyển đổi mảng chuỗi nhị phân 16 bit thành mảng int16
    int16_array = np.array([int(binary_str, 2) if int(binary_str, 2) <= np.iinfo(np.int16).max else np.iinfo(np.int16).max for binary_str in binary_array], dtype=np.int16)
    return int16_array


def DNA_subtract(dnaseq_arr_1, dnaseq_arr_2):
    # Tạo bảng trừ
    subtract_table = {
        'A': ['A', 'T', 'G', 'C'],
        'C': ['C', 'A', 'T', 'G'],
        'G': ['G', 'C', 'A', 'T'],
        'T': ['T', 'G', 'C', 'A']
    }

    # Thực hiện phép trừ
    subtract_dna_arr = []
    for dna1, dna2 in zip(dnaseq_arr_1, dnaseq_arr_2):
        result = ''
        for base1, base2 in zip(dna1, dna2):
            # Chuyển đổi base2 thành số tương ứng
            tmp2 = 'ACGT'.index(base2)

            # Thực hiện trừ và thêm vào kết quả
            result += subtract_table[base1][tmp2]
        subtract_dna_arr.append(result)

    return subtract_dna_arr

def dna_apply1(Key, binary_cipher, binary_chaosValue1, binary_chaosValue2):
  rule = Key % 8
  
  dnaseq_cipher = Binary_to_DNA_Seq(rule, binary_cipher)
  dnaseq_val1 = Binary_to_DNA_Seq(rule, binary_chaosValue1)
  dnaseq_val2 = Binary_to_DNA_Seq(rule, binary_chaosValue2)
  
  AddValue1 = DNA_subtract(dnaseq_cipher, dnaseq_val2)  
  Result = DNA_subtract(AddValue1, dnaseq_val1)
  
  dnaseq_output = DNA_Seq_to_Binary(rule, Result)
  return dnaseq_output

def generate_encryption_key1(Length):
    # Tạo ShareKey (SecretKey) và InitialVector một cách ngẫu nhiên
    ShareKey, InitialVector = ReadFromFile()
    InitialVector_hex = Hex(InitialVector)

    I0 = int(InitialVector_hex[0:8], 16)
    I1 = int(InitialVector_hex[8:16], 16)
    I2 = int(InitialVector_hex[16:24], 16)
    I3 = int(InitialVector_hex[24:32], 16)

    IP_SC = 1 / (np.floor(int(ShareKey, 2)) + 1)
    CP_SC = 2.2 + ((I2 ^ I3) % 5)
    IP_LSC = 1 / (np.floor(int(InitialVector, 2)) + 1)
    CP_LSC = 1 / (np.floor(I3) + 1)

    val = I0 % 64
    IntermediateKey1 = ShareKey[val:val + 32]
    d = (2 * math.floor(np.floor(int(InitialVector, 2)) / 2)) + 1
    Tmp_Key = left_rotate(ShareKey, d)
    val1 = I1 % 64
    IntermediateKey2 = Tmp_Key[val1:val1 + 32]

    # CHAOTIC MAP GENERATION
    SineCosineSequence = SineCosineChaoticMap(IP_SC, CP_SC, Length)
    LogisticSineCosineSequence = LogisticSineCosine(IP_LSC, CP_LSC, Length)

    return InitialVector, IntermediateKey1, IntermediateKey2, SineCosineSequence, LogisticSineCosineSequence, I0, I1, I2, I3

def decrypt_song(path):
    # Đọc file âm thanh
    CipherVoice, sample_rate = Input(path)

    Length = len(CipherVoice)

    for i in range(Length):
      if (CipherVoice[i] > 0):
        CipherVoice[i] = CipherVoice[i] + 1
      elif (CipherVoice[i] < 0):
        CipherVoice[i] = CipherVoice[i] - 1

    #Tạo mảng rỗng kiểu int16
    PlainVoice = np.zeros(Length, dtype=np.int16)

    # Tạo key cho mỗi bài hát
    InitialVector, IntermediateKey1, IntermediateKey2, SineCosineSequence, LogisticSineCosineSequence , I0, I1, I2, I3 = generate_encryption_key1(Length)

    # DECVOICE GENERATION

    I01 = np.zeros(Length, dtype=np.int16)
    I02 = np.zeros(Length, dtype=np.int16)

    for i in range(Length):
      I01[i] = np.int32((I1 * SineCosineSequence[i]))
      I02[i] = np.int32((I2 * LogisticSineCosineSequence[i]))

    DecVoice = np.zeros(Length, dtype=np.int16)

    for i in range(Length):
      val = (I0 * (i + 1)) % 2
      if val == 0:
        DecVoice[i] = np.int16((CipherVoice[i] - I01[i] * (i + 1)) % pow(2,16))
      else:
        DecVoice[i] = np.int16((CipherVoice[i] - I02[i] * (i + 1)) % pow(2,16))

    # DIFFUSION PHASE

    IntermediateKey1_int = int(IntermediateKey1, 2)
    IntermediateKey2_int = int(IntermediateKey2, 2)

    CM1 = ((IntermediateKey1_int * SineCosineSequence) / pow(2, 16)).astype(np.int16)
    CM2 = ((IntermediateKey2_int * LogisticSineCosineSequence) / pow(2, 16)).astype(np.int16)

    binary_cipher = int16_array_to_binary16(DecVoice)
    binary_val1 = int16_array_to_binary16(CM1)
    binary_val2 = int16_array_to_binary16(CM2)

    DNA_data = dna_apply1(IntermediateKey2_int, binary_cipher, binary_val1, binary_val2)

    Permutated_Audio = np.zeros(Length, dtype = np.int16)

    for i in range(Length):
      #Permutated_Audio[i] = np.int16(int(DNA_data[i], 2))
      Permutated_Audio[i] = np.array(int(DNA_data[i], 2), dtype=np.uint16)

    PlainVoice = Reverse_Permutation(Permutated_Audio, I3, SineCosineSequence)

    return PlainVoice, sample_rate, InitialVector

import tkinter as tk
from tkinter import filedialog, messagebox

def decrypt():
    # Placeholder function for decrypt functionality
    converted_string = songsltpath.replace("\\", "/")
    
    # Decrypt the song
    
    # Ask user for the output file location and name
    pathOutput = filedialog.asksaveasfilename(
        title="Save decrypted audio as...",
        defaultextension=".wav",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    PlainVoice, sample_rate, InitialVector = decrypt_song(converted_string)
    Output(pathOutput, sample_rate, PlainVoice)
    messagebox.showinfo("Thông báo", "Dữ liệu đã được tải xuống thành công!")

# Example usage of the function
# decrypt()


# Create the main window
window = tk.Tk()
window.title("SoundMarket App")
window.geometry("700x700")  # Increased height to fit the new button

# Create a label for the music player title
l_music_player = tk.Label(window, text="SoundMarket", font=("TkDefaultFont", 30, "bold"))
l_music_player.pack(pady=10)

# Create a button to select the music folder
btn_select_folder = ctk.CTkButton(window, text="Available Music's List",
                                  command=select_music_folder,
                                  font=("TkDefaultFont", 18))
btn_select_folder.pack(pady=20)

# Create a listbox to display the available songs
lbox = tk.Listbox(window, width=50, font=("TkDefaultFont", 16))
lbox.pack(pady=10)

# Create a frame to hold the control buttons
btn_frame = tk.Frame(window)
btn_frame.pack(pady=20)

# Create a button to go to the previous song
btn_previous = ctk.CTkButton(btn_frame, text="<", command=previous_song,
                            width=50, font=("TkDefaultFont", 18))
btn_previous.pack(side=tk.LEFT, padx=5)

# Create a button to play the music
btn_play = ctk.CTkButton(btn_frame, text="Play", command=play_music, width=50,
                         font=("TkDefaultFont", 18))
btn_play.pack(side=tk.LEFT, padx=5)

# Create a button to pause the music
btn_pause = ctk.CTkButton(btn_frame, text="Pause", command=pause_music, width=50,
                          font=("TkDefaultFont", 18))
btn_pause.pack(side=tk.LEFT, padx=5)

# Create a button to go to the next song
btn_next = ctk.CTkButton(btn_frame, text=">", command=next_song, width=50,
                         font=("TkDefaultFont", 18))
btn_next.pack(side=tk.LEFT, padx=5)

# Create a progress bar to indicate the current song's progress
pbar = Progressbar(window, length=300, mode="determinate")
pbar.pack(pady=10)

# Create a button to show the text boxes
btn_show_text_boxes = ctk.CTkButton(window, text="Upload Music", command=show_text_boxes,
                                    font=("TkDefaultFont", 18))
btn_show_text_boxes.pack(pady=10)
btn_show_text_boxes = ctk.CTkButton(window, text="Dowload", command=decrypt,
                                    font=("TkDefaultFont", 18))
btn_show_text_boxes.pack(pady=10)

window.mainloop()
