import requests
import json
from tkinter import *
from tkinter import ttk
from PIL import ImageTk,Image
from tkinter import messagebox
from tkinter import filedialog
import server
from flask import Flask,jsonify,request


def create_app():
    app = Flask(__name__)

    with app.app_context():
        server.get_location_structure()

    return app

try:
    create_app()
    url = "http://127.0.0.1:5000/get_location_structure"
    response = requests.get( url, headers={'Accept': "application/json"})
except Exception as e:
    print(e)


class UI:
    def __init__(self,master):
        self.master = master
        self.master.title("ML_omoriswaga")
        self.master.resizable(False, False)
        # resizing the window created
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        window_height = int(self.screen_height) / 1.5428
        window_width = int(self.screen_width) / 2.7428
        x_cordinate = int((self.screen_width / 2) - (window_width / 2))
        y_cordinate = int((self.screen_height / 2) - (window_height / 2))
        # implementing it
        self.master.geometry("{}x{}+{}+{}".format(int(window_width), int(window_height), x_cordinate, y_cordinate))
        self.master.configure(bg='white')

        self.Area_in_sqft = Label(self.master, text="Area (Square feet)")
        self.Area_in_sqft.pack()

        self.Area_value_insqft = Entry(self.master)
        self.Area_value_insqft.pack(pady=10)

        self.bhk = Label(self.master,text='BHK')
        self.bhk.pack(pady=10)

        bhk_val = [1,2,3,4,5]
        bath_val = [1,2,3,4,5]

        self.bhk_entry = ttk.Combobox(self.master,value=bhk_val, width=30)
        self.bhk_entry.set("Choose a number of bed-room")
        self.bhk_entry.bind("<<ComboboxSelected>>", self.bhk_combo_clicked)
        self.bhk_entry.pack(pady=10)

        self.bath = Label(self.master,text='BATH')
        self.bath.pack(pady=10)

        self.bath_entry = ttk.Combobox(self.master, value=bath_val,width=30)
        self.bath_entry.set("Choose a number of bath-room")
        self.bath_entry.bind("<<ComboboxSelected>>", self.bath_combo_clicked)
        self.bath_entry.pack(pady=10)

        self.location = Label(self.master, text='Location')
        self.location.pack(pady=10)

        self.location_type = ttk.Combobox(self.master,value=response.json()['location'],width=30)
        self.location_type.set("Choose a location type")
        self.location_type.bind("<<ComboboxSelected>>", self.location_combo_clicked)
        self.location_type.pack(pady=10)

        self.area = Label(self.master, text='Area')
        self.area.pack(pady=10)

        self.area_type = ttk.Combobox(self.master, value=response.json()['area'],width=30)
        self.area_type.set("Choose an area type")
        self.area_type.bind("<<ComboboxSelected>>", self.area_combo_clicked)
        self.area_type.pack(pady=10)

        button = Button(self.master, text="Submit", command = self.clicked)
        button.pack(pady=30)

        self.answer = Label(self.master, text = "")
        self.answer.pack(pady=10)


    def bhk_combo_clicked(self, event):
        self.bhk_value = self.bhk_entry.get()

    def bath_combo_clicked(self,event):
        self.bath_value = self.bath_entry.get()

    def location_combo_clicked(self, event):
        self.location_value = self.location_type.get()

    def area_combo_clicked(self, event):
        self.area_value = self.area_type.get()

    def clicked(self):
        try:
            answer = server.get_estimated_price(self.area_value,self.location_value,self.Area_value_insqft.get(),self.bhk_value,self.bath_value)
            self.answer.config(text=str(answer) + " " + "Lakh", fg='red')
        except Exception as e:
            print(e)



if __name__ == '__main__' :
    root = Tk()
    UI(root)
    root.mainloop()