import tkinter
from tkinter import filedialog
import customtkinter

import numpy as np
import pandas as pd

import random
from matplotlib import pyplot as plt


class App(customtkinter.CTk):
    WIDTH = 750
    HEIGHT = 350
    class popup_failed(customtkinter.CTk):
        def __init__(self):
            super().__init__()
            self.title("FAILED")
            self.geometry("290x120")
            self.protocol("WM_DELETE_WINDOW", self.on_closing)

            # ============ create_frames ============
            self.grid_columnconfigure(1, weight=1)
            self.grid_rowconfigure(0, weight=1)
            self.frame = customtkinter.CTkFrame(master=self)
            self.frame.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

            self.label_promt = customtkinter.CTkLabel(master=self.frame,text="Lỗi\nChọn đường dẫn cho file train.csv\nNhập số lần lặp (nên nhập 100/500/1000)\nBắt đầu huấn luyện")
            self.label_promt.grid(row=0, column=0, columnspan=1, pady=5, padx=5, sticky="")

        def on_closing(self, event=0):
            self.destroy()
    
    def __init__(self):
        super().__init__()
        self.title("Deep Learning Demo")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ============ create_frames ============
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.frame = customtkinter.CTkFrame(master=self)
        self.frame.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_main ===============
        self.label_PATH = customtkinter.CTkLabel(master=self.frame,text="Địa chỉ")

        self.label_PATH.grid(row=0, column=0, columnspan=1, pady=5, padx=5, sticky="")
        
        self.label_PATH_info = customtkinter.CTkLabel(master=self.frame, text="Nhập file .csv",
                                                      height=15,
                                                      corner_radius=6,
                                                      fg_color=("white", "gray38"),
                                                      justify=tkinter.LEFT)
        self.label_PATH_info.grid(column=1, row=0, sticky="", padx=10, pady=15)
        self.button_PATH_path = customtkinter.CTkButton(master=self.frame,
                                                        text="Nhập đường dẫn",
                                                        command=self.button_event_PATH_path)
        self.button_PATH_path.grid(column=2, row=0, padx=10, pady=10)

        
        self.label_info = customtkinter.CTkLabel(master=self.frame,
                                                   text="Nhập số lần lặp trước khi train",
                                                   height=100,
                                                   width= 300,
                                                   corner_radius=6, 
                                                   fg_color=("white", "gray38"), 
                                                   justify=tkinter.LEFT)
        self.label_info.grid(column=1, row=2, sticky="nwe", padx=15, pady=15)

        self.label_info_2 = customtkinter.CTkLabel(master=self.frame,
                                                   text="",
                                                   height=0,
                                                   width= 0,
                                                   corner_radius=6, 
                                                   fg_color=("#302c2c", "#302c2c"), 
                                                   justify=tkinter.LEFT)
        self.label_info_2.grid(column=99, row=2)
    

        self.entry_iterations = customtkinter.CTkEntry(master=self.frame, placeholder_text="Số lần lặp")
        self.entry_iterations.grid(column=1, pady=12, padx=10)

        self.button_start= customtkinter.CTkButton(master=self.frame,
                                                        text="Bắt đầu luyện",
                                                        command=self.button_train)
        self.button_start.grid(column=0, row=5, padx=10, pady=10)
        
        self.button_start= customtkinter.CTkButton(master=self.frame,
                                                        text="Test ngẫu nhiên",
                                                        command=self.button_test)
        self.button_start.grid(column=1, row=5, padx=10, pady=10)

        self.button_start= customtkinter.CTkButton(master=self.frame,
                                                        text="Test số lượng lớn",
                                                        command=self.button_test_multi)
        self.button_start.grid(column=2, row=5, padx=10, pady=10)

    def button_event_PATH_path(self):
        global filename_PATH
        filename_PATH = filedialog.askopenfilename(initialdir="/",title="Chon file (.csv)", filetypes=(("Microsoft Excel Comma Separated Values File (.csv)","*.csv"),("All files","*.*")) )
        filename_PATH_info = customtkinter.CTkLabel(master=self.frame, text=filename_PATH,
                                                      height=15,
                                                      corner_radius=6,
                                                      fg_color=("white", "gray38"),
                                                      justify=tkinter.LEFT)
        filename_PATH_info.grid(column=1, row=0, sticky="", padx=10, pady=15)

    def button_train(self):
        try:
            self.label_info_2.destroy()
            self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
            global data, W1, b1, W2, b2, X_train, Y_train, X_dev, Y_dev
            data = pd.read_csv(filename_PATH)

            data = np.array(data)
            m, n = data.shape
            np.random.shuffle(data) # shuffle

            data_dev = data[0:1000].T
            Y_dev = data_dev[0]
            X_dev = data_dev[1:n]
            X_dev = X_dev / 255.

            data_train = data[1000:m].T
            Y_train = data_train[0]
            X_train = data_train[1:n]
            X_train = X_train / 255.
            _,m_train = X_train.shape

            def init_params():
                W1 = np.random.rand(10, 784) -0.5
                b1 = np.random.rand(10, 1) -0.5
                W2 = np.random.rand(10, 10) -0.5
                b2 = np.random.rand(10, 1) -0.5
                return W1, b1, W2, b2

            def ReLU(Z):
                return np.maximum(Z, 0)

            def softmax(Z):
                return np.exp(Z) / sum(np.exp(Z))
                
            def one_hot(Y):
                one_hot_Y = np.zeros((Y.size, Y.max() + 1))
                one_hot_Y[np.arange(Y.size), Y] = 1
                one_hot_Y = one_hot_Y.T
                return one_hot_Y

            def derivative_ReLU(Z):
                return Z > 0

            def forward_propagation(W1, b1, W2, b2, X):
                Z1 = W1.dot(X) + b1
                A1 = ReLU(Z1)
                Z2 = W2.dot(A1) + b2
                A2 = softmax(Z2)
                return Z1, A1, Z2, A2

            def backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y):
                one_hot_Y = one_hot(Y)
                dZ2 = A2 - one_hot_Y
                dW2 = 1 / m * dZ2.dot(A1.T)
                db2 = 1 / m * np.sum(dZ2)
                dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
                dW1 = 1 / m * dZ1.dot(X.T)
                db1 = 1 / m * np.sum(dZ1)
                return dW1, db1, dW2, db2

            def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
                W1 = W1 - alpha * dW1
                b1 = b1 - alpha * db1
                W2 = W2 - alpha * dW2
                b2 = b2 - alpha * db2
                return W1, b1, W2, b2

            def get_accuracy(predictions, Y):
                return np.sum(predictions == Y) / Y.size

            def get_predictions(A2):
                return np.argmax(A2, 0)

            def gradient_descent(X, Y, alpha, iterations):
                W1, b1, W2, b2 = init_params()
                for i in range(iterations):
                    Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
                    dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y)
                    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
                    if i % 10 == 0:
                        predictions = get_predictions(A2)
                        self.label_info.configure(text="Iteration: {}\n{} {}\n{}".format(i,predictions,Y,get_accuracy(predictions, Y)))
                        self.label_info.grid(column=1, row=2, sticky="nwe", padx=15, pady=15)     
                        self.update()
                return W1, b1, W2, b2

            
            W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, int(self.entry_iterations.get()))
            self.update()
        except:
            pop = self.popup_failed()
            pop.mainloop()
        
    def button_test(self):
        try:
            self.label_info_2.destroy()
            def test_prediction(index, W1, b1, W2, b2):
                current_image = X_train[:, index, None]
                prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
                label = Y_train[index]

                self.label_info.configure(text="Prediction: {}\nLabel: {}".format(prediction, label))
                self.label_info.grid(column=1, row=2, sticky="nwe", padx=15, pady=15)
                self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
                self.update()

                current_image = current_image.reshape((28, 28)) * 255
                plt.gray()
                plt.imshow(current_image, interpolation='nearest')
                plt.show()

            def ReLU(Z):
                return np.maximum(Z, 0)

            def softmax(Z):
                return np.exp(Z) / sum(np.exp(Z))
                
            def one_hot(Y):
                one_hot_Y = np.zeros((Y.size, Y.max() + 1))
                one_hot_Y[np.arange(Y.size), Y] = 1
                one_hot_Y = one_hot_Y.T
                return one_hot_Y

            def derivative_ReLU(Z):
                return Z > 0

            def get_predictions(A2):
                return np.argmax(A2, 0)


            def forward_propagation(W1, b1, W2, b2, X):
                Z1 = W1.dot(X) + b1
                A1 = ReLU(Z1)
                Z2 = W2.dot(A1) + b2
                A2 = softmax(Z2)
                return Z1, A1, Z2, A2

                
            def make_predictions(X, W1, b1, W2, b2):
                _,_,_, A2 = forward_propagation(W1, b1, W2, b2, X)
                predictions = get_predictions(A2)
                return predictions
            
            test_prediction(random.randint(0, 1000), W1, b1, W2, b2)
        except:
            pop = self.popup_failed()
            pop.mainloop()

        
    def button_test_multi(self):
        try:
            def ReLU(Z):
                return np.maximum(Z, 0)

            def get_accuracy(predictions, Y):
                self.label_info_2 = customtkinter.CTkLabel(master=self.frame,
                                                       text="{}".format(predictions),
                                                       height=100,
                                                       width= 300,
                                                       corner_radius=6, 
                                                       fg_color=("white", "gray38"), 
                                                       justify=tkinter.LEFT)
                self.label_info_2.grid(column=0, row=2, sticky="nwe", padx=15, pady=15)
                self.label_info.configure(text="{}".format(Y))
                self.label_info.grid(column=2, row=2, sticky="nwe", padx=15, pady=15)
                self.geometry("1050x650")
                self.update()
                return np.sum(predictions == Y) / Y.size
            
            def softmax(Z):
                return np.exp(Z) / sum(np.exp(Z))
                
            def one_hot(Y):
                one_hot_Y = np.zeros((Y.size, Y.max() + 1))
                one_hot_Y[np.arange(Y.size), Y] = 1
                one_hot_Y = one_hot_Y.T
                return one_hot_Y

            def derivative_ReLU(Z):
                return Z > 0

            def get_predictions(A2):
                return np.argmax(A2, 0)


            def forward_propagation(W1, b1, W2, b2, X):
                Z1 = W1.dot(X) + b1
                A1 = ReLU(Z1)
                Z2 = W2.dot(A1) + b2
                A2 = softmax(Z2)
                return Z1, A1, Z2, A2

                
            def make_predictions(X, W1, b1, W2, b2):
                _,_,_, A2 = forward_propagation(W1, b1, W2, b2, X)
                predictions = get_predictions(A2)
                return predictions
            
            dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
            print(get_accuracy(dev_predictions, Y_dev))
        except:
            pop = self.popup_failed()
            pop.mainloop()
    def on_closing(self, event=0):
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
