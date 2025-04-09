import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

app = tk.Tk()
app.geometry("532x622")
app.title("Stable Bud")
ctk.set_appearance_model("dark")

prompt = ctk.CTkEntry(height = 40,width = 512,text_font=("Arial",20),text_color = "black")
prompt.place(x=10,y=10)

lmain = ctk.CTkLabel(height=512,width=512)
lmain.place(x=10,y=110)

app.mainloop()
