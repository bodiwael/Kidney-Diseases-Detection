from keras.models import load_model
import gradio as gr

model = load_model('kidney_stone_detection_model.h5')

import sklearn.externals
import joblib
svc = joblib.load("svc.pkl") 

# Function to Select Image
def browse_btn():
    global image_name
    
    label_cnn.configure(text="")
    label.configure(text="")
    
    image_name = askopenfilename(title='Select Image')
    img = Image.open(image_name)
    img = img.resize((200, 200), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(root, image=img)
    panel.image = img
    panel.grid(row=0,column=1,sticky='nw',padx=20,pady=28)

# Function to Predict CNN
def predict_btn_cnn():
    global label_prediction
    global image_name
    test_img = image.load_img(image_name, target_size=(150, 150))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    result = model.predict(test_img)
    if result[0][0] == 1:
        label_cnn.configure(text="Kidney Stone Detected")
    elif result[0][0] == 0:
        label_cnn.configure(text="No Kidney Stone Detected")
        
#Function for Predict SVM        
def predict_btn_svm():
    global label_prediction
    global image_name
    test_img = cv2.imread(image_name)
    #test_img = image.load_img(image_name, target_size=(150, 150))
    #test_img = image.img_to_array(test_img)
    feature_list_of_img = extract_features([test_img])
    result = svc.predict(feature_list_of_img)    
    #Displaying the output
    if result[0] == 'Stone':
        label.configure(text = "Kidney Stone Detected")
    elif result[0] == 'Normal':
        label.configure(text = "No Kidney Stone Detected")

# Creating the GUI
from tkinter import *
from PIL import Image, ImageTk
import customtkinter
import tkinter
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import keras.utils as image
import numpy as np
import os

customtkinter.set_appearance_mode("System")
root = customtkinter.CTk()

#window size
root.rowconfigure(0,weight=1)    
root.columnconfigure(0,weight=1)

root.geometry('420x380')
root.title('Kidney Stone Detection')
        
# Browse Button
browsebtn = customtkinter.CTkButton(master=root, text="Browse Image", command=browse_btn)
browsebtn.grid(row=0, column=0,sticky='nw',padx=20,pady=20)


# Predict Butoon CNN
predictbtn = customtkinter.CTkButton(master=root, text="Predict CNN", command=predict_btn_cnn)
predictbtn.grid(row=1, column=0,sticky='nw',padx=20,pady=20)

#Label Result CNN
label_cnn = customtkinter.CTkLabel(root, text="")
label_cnn.grid(row=1,column=1,sticky='nw',padx=20,pady=20)

#Label Result SVM
label = customtkinter.CTkLabel(root, text="")
label.grid(row=2,column=1,sticky='nw',padx=20,pady=20)
# Predict Butoon SVM
predictbtnsvm = customtkinter.CTkButton(master=root, text="Predict SVM", command=predict_btn_svm)
predictbtnsvm.grid(row=2, column=0,sticky='nw',padx=20,pady=20)

# Running the GUI
# root.mainloop()

def custom_Image_preprocessing(image_data, target_size=(150, 150)):
    img = image.array_to_img(image_data, data_format='channels_last')
    img = img.resize(target_size)  # Resize the image if needed
    img_arr = image.img_to_array(img)
    img_arr = img_arr * 1./255
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr
# Function to Predict CNN
def predict(img):
    image_preprocess = custom_Image_preprocessing(img)
    result = model.predict(image_preprocess)
    print(result[0][0])
    if result[0][0] > 0.5:
        return 'Kidney Stone Detected (Positive)',round(result[0][0]*100,2),'%'
    else:
        return 'No Kidney Stone Detected  (Negative)',round(result[0][0]*100,2),'%'
# Create a Gradio interface
input_component =  gr.components.Image(label = "Upload the CT-Image")
output_component = gr.components.Textbox(label = "Prediction")

iface = gr.Interface(
    fn=predict,
    inputs=input_component, 
    outputs=output_component,
    title = "Kidney Stone Classification",
    description="This web app provides predictions based on CT-images and predict either the CT-Scan  contains sympotms of Kidney stone or not "
)

iface.launch()
