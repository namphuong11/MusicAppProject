import scipy.io.wavfile
import librosa
import numpy as np
import math

# Hàm nhập dữ liệu âm thanh từ file
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
       
# Đọc vào hai mảng ShareKey và IntialVector với 128 bits từ file
def ReadFromFile(pathShareKey, pathIntiialVector):
  with open(pathShareKey, 'rb') as f:
    ShareKey = f.read().decode('utf-8')

  with open(pathIntiialVector, 'rb') as f:
    IntialVector = f.read().decode('utf-8')

  return ShareKey, IntialVector

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

def int16_array_to_binary16(array):
    # Chuyển đổi mảng int16 sang chuỗi nhị phân 16 bit
    binary_array = [np.binary_repr(x, width=16) for x in array]
    return binary_array

def binary_array_to_int16(binary_array):
    # Chuyển đổi mảng chuỗi nhị phân 16 bit thành mảng int16
    int16_array = np.array([min(int(binary_str, 2), np.iinfo(np.int16).max) for binary_str in binary_array], dtype=np.int16)
    
    return int16_array

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

def dna_apply(Key, binary_cipher, binary_chaosValue1, binary_chaosValue2):
  rule = Key % 8
  
  dnaseq_cipher = Binary_to_DNA_Seq(rule, binary_cipher)
  dnaseq_val1 = Binary_to_DNA_Seq(rule, binary_chaosValue1)
  dnaseq_val2 = Binary_to_DNA_Seq(rule, binary_chaosValue2)
  
  AddValue1 = DNA_subtract(dnaseq_cipher, dnaseq_val2)  
  Result = DNA_subtract(AddValue1, dnaseq_val1)
  
  dnaseq_output = DNA_Seq_to_Binary(rule, Result)
  return dnaseq_output

def generate_encryption_key(Length, pathShareKey, pathIntiialVector):
    # Tạo ShareKey (SecretKey) và InitialVector một cách ngẫu nhiên
    ShareKey, InitialVector = ReadFromFile(pathShareKey, pathIntiialVector)
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

def decrypt_song(path, pathShareKey, pathIntiialVector):
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
    InitialVector, IntermediateKey1, IntermediateKey2, SineCosineSequence, LogisticSineCosineSequence , I0, I1, I2, I3 = generate_encryption_key(Length, pathShareKey, pathIntiialVector)

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

    DNA_data = dna_apply(IntermediateKey2_int, binary_cipher, binary_val1, binary_val2)

    Permutated_Audio = np.zeros(Length, dtype = np.int16)

    for i in range(Length):
      #Permutated_Audio[i] = np.int16(int(DNA_data[i], 2))
      Permutated_Audio[i] = np.array(int(DNA_data[i], 2), dtype=np.uint16)

    PlainVoice = Reverse_Permutation(Permutated_Audio, I3, SineCosineSequence)

    return PlainVoice, sample_rate, InitialVector
# -------------- DECRYPTION -------------- #

path = input("Nhập đầu vào: ")
pathShareKey = input("Nhập Key đầu vào: ")
pathIntiialVector = input("Nhập IntialVector đầu vào: ")
pathOutput = input("Output: ")

# Mã hóa bài hát
PlainVoice, sample_rate, InitialVector = decrypt_song(path, pathShareKey, pathIntiialVector)

Output(pathOutput, sample_rate, PlainVoice)