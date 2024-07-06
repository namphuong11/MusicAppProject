import numpy as np

def find_different_positions(array1, array2):
    # Chuyển đổi mảng thành NumPy array nếu chưa phải
    array1 = np.array(array1)
    array2 = np.array(array2)
    
    cnt = 0
    
    # Tìm vị trí các phần tử khác nhau
    for i in range(len(array1)):
        #print(array1[i] , " ---> ", array2[i])
        if (array1[i] != array2[i]):
            cnt = cnt + 1

    return cnt

# Đọc mảng từ file hoặc sử dụng các mảng đã có
# Đây chỉ là một ví dụ, bạn cần thay đổi tùy thuộc vào định dạng của file và cách lấy dữ liệu từ file
array1 = np.loadtxt("P_D.txt", dtype = str)
array2 = np.loadtxt("P_E.txt", dtype = str)

# Gọi hàm để tìm vị trí khác nhau
different_positions = find_different_positions(array1, array2)

if(different_positions == 0):
    print("MATCH")
else:
    print("NOT MATCH: ", different_positions, " pos.")