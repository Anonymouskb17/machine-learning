import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from tkinter import *
from tkinter import messagebox

# Đọc file csv
data = pd.read_csv('weather fix 1.csv')

# Chia data thành 2 tập train và test
dt_train, dt_test = train_test_split(data, test_size=0.3, shuffle=True)

# X_train lấy một bản sao train nhưng sẽ loại bỏ cột Raintoday
x_train = dt_train.drop(['RainToday'], axis=1)
# Y_train tạo mới một bộ từ cột Raintoday
y_train = dt_train['RainToday']
# X_test tạo ra một bản test loại bỏ cột Raintoday
x_test = dt_test.drop(['RainToday'], axis=1)
# Y_test đầu ra sau khi train
y_test = dt_test['RainToday']

# StadardScaler: chuyển đổi dữ liệu để có giá trị trung bình gần bằng 0 và độ lệch chuẩn gần bằng 1
# n_neighbors=5: mô hình K-Nearest Neighbors với k=5
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

# Huấn luyện mô hình
model.fit(x_train, y_train)

# Tính ra dự đoán nhờ bộ x_test
y_pred = model.predict(x_test)

# Tạo cửa sổ
new = Tk()
new.title("Dự báo thời tiết")
new.geometry('1000x700')

# phần giao diện của từng label
label_begin= Label(new,text="*DỰ BÁO THỜI TIẾT*", font=('Arial',20),fg='green')
label_begin.place(x=250,y=30)

label_MinT = Label(new,font=('Arial',12),text='Nhập nhiệt độ thấp nhất:',fg='red')
label_MinT.place(x=50,y=100)
entry_MinT = Entry(new,width=20,font=('Arial',12))
entry_MinT.place(x=250,y=100)

label_MaxT = Label(new,font=('Arial',12),text='Nhập nhiệt độ cao nhất:',fg='red')
label_MaxT.place(x=50,y=150)
entry_MaxT = Entry(new,width=20,font=('Arial',12))
entry_MaxT.place(x=250,y=150)


label_Rainf = Label(new,font=('Arial',12),text='Nhập lượng mưa:',fg='red')
label_Rainf.place(x=50,y=200)
entry_Rainf = Entry(new,width=20,font=('Arial',12))
entry_Rainf.place(x=250,y=200)


label_WinG = Label(new,font=('Arial',12),text='Nhập tốc độ gió:',fg='red')
label_WinG.place(x=50,y=250)
entry_WinG = Entry(new,width=20,font=('Arial',12))
entry_WinG.place(x=250,y=250)


label_Win0am = Label(new,font=('Arial',12),text='Nhập tốc độ gió 0 giờ sáng:',fg='red')
label_Win0am.place(x=50,y=300)
entry_Win0am = Entry(new,width=20,font=('Arial',12))
entry_Win0am.place(x=250,y=300)


label_Win5am = Label(new,font=('Arial',12),text='Nhập tốc độ gió 5 giờ sáng:',fg='red')
label_Win5am.place(x=50,y=350)
entry_Win5am = Entry(new,width=20,font=('Arial',12))
entry_Win5am.place(x=250,y=350)


label_Humi0am = Label(new,font=('Arial',12),text='Nhập độ ẩm 0 giờ sáng:',fg='red')
label_Humi0am.place(x=550,y=100)
entry_Humi0am = Entry(new,width=20,font=('Arial',12))
entry_Humi0am.place(x=750,y=100)


label_Humi5am = Label(new,font=('Arial',12),text='Nhập độ ẩm 5 giờ sáng:',fg='red')
label_Humi5am.place(x=550,y= 150)
entry_Humi5am = Entry(new,width=20,font=('Arial',12))
entry_Humi5am.place(x=750,y=150)


label_Press0am = Label(new,font=('Arial',12),text='Nhập áp xuất 0 giờ sáng:',fg='red')
label_Press0am.place(x=550,y=200)
entry_Press0am = Entry(new,width=20,font=('Arial',12))
entry_Press0am.place(x=750,y=200)


label_Press5am = Label(new,font=('Arial',12),text='Nhập áp xuất 5 giờ sáng:',fg='red')
label_Press5am.place(x=550,y=250)
entry_Press5am = Entry(new,width=20,font=('Arial',12))
entry_Press5am.place(x=750,y=250)


label_Temp0am = Label(new,font=('Arial',12),text='Nhập nhiệt độ 0 giờ sáng:',fg='red')
label_Temp0am.place(x=550,y=300)
entry_Temp0am = Entry(new,width=20,font=('Arial',12))
entry_Temp0am.place(x=750,y=300)


label_Temp5am = Label(new,font=('Arial',12),text='Nhập nhiệt độ 5 giờ sáng:',fg='red')
label_Temp5am.place(x=550,y=350)
entry_Temp5am = Entry(new,width=20,font=('Arial',12))
entry_Temp5am.place(x=750,y=350)

# hàm này để chuyển giá trị trong entry(string) thành float
def convert_to_float(string_value):
    try:
        float_value = float(string_value)
        return float_value
    except ValueError:
        return None

# sau khi chuyển thành float thì dùng hàm này để lấy những giá trị nhập vào từng label
def dudoan():
    minT = convert_to_float(entry_MinT.get())
    maxT = convert_to_float(entry_MaxT.get())
    rainF = convert_to_float(entry_Rainf.get())
    winG = convert_to_float( entry_WinG.get())
    win0am  = convert_to_float(entry_Win0am.get())
    win5am  = convert_to_float(entry_Win5am.get())
    humi0am = convert_to_float(entry_Humi0am.get())
    humi5am = convert_to_float(entry_Humi5am.get())
    press0am = convert_to_float(entry_Press0am.get())
    press5am = convert_to_float(entry_Press5am.get())
    temP0am = convert_to_float(entry_Temp0am.get())
    temP5am = convert_to_float(entry_Temp5am.get())
    
    # tạo một danh sách giá trị biến, np.array để chuyển các giá trị biến theo đúng danh sách
    # reshape để chuyển dạng thành ma trận (1,n) 
    # n là số lượng giá trị biến. Kích thước -1 thể hiện số lượng cột được tính toán dựa trên số lượng giá trị biến
    tranf=np.array([minT,maxT,rainF,winG,win0am,win5am,humi0am,humi5am,press0am,press5am,temP0am,temP5am]).reshape(1,-1)
    # tính toán kết quả dự đoán sử dụng model sau khi train và dùng giá trị các biến
    y_pred= model.predict(tranf)
   
       
    # nếu dự đoán bằng 0 hoặc 1 thì đưa ra kết quả
    if (y_pred==0):
        d = 'Không mưa'
    else:
        d = 'Có mưa'
    label_dudoan.config(text=d)

#  hàm để thể hiện tỉ lệ chính xác
def tile():
    accuracy=accuracy_score(y_test,y_pred)*100
    acc= round(accuracy,4)
    label_ketqua = Label(new,font=('Arial',12),text='Tỉ lệ chính xác: '+ str(acc)+' % ',fg='red')
    label_ketqua.place(x=350,y=450) 
    
    precision = precision_score(y_test, y_pred)*100
    prec = round(precision,4)
    label_precision = Label(new,font=('Arial',12),text='Tỉ lệ precision: '+ str(prec)+' % ',fg='red')
    label_precision.place(x=350,y=500)
    
    recall = recall_score(y_test,y_pred)*100
    re= round(recall,4)
    label_recall = Label(new,font=('Arial',12),text='Tỉ lệ recall: '+ str(re)+' % ',fg='red')
    label_recall.place(x=350,y=550)
    
    f1 = f1_score(y_test, y_pred)*100
    F1= round(f1,4)
    label_f1 = Label(new,font=('Arial',12),text='Tỉ lệ f1: '+ str(F1)+' % ',fg='red')
    label_f1.place(x=350,y=600)
    
    


# button để thực hiện đưa ra dự đoán
nut= Button(new,text='Dự đoán',width=8,height=2,command=dudoan)
nut.place(x=250,y=400)

# button thực hiện đưa ra tỉ lệ chính xác
nut_dd= Button(new,text='Tỉ lệ ',width=10,height=2,command=tile)
nut_dd.place(x=250,y=450)

label_dudoan = Label(new,width=20,height=2,text='',fg='red')
label_dudoan.place(x=400,y=400)

new.mainloop()
