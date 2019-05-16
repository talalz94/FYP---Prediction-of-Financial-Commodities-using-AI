##from tkinter import *
## 
##window = Tk()
## 
##window.title("Welcome to LikeGeeks app")
## 
##window.geometry('350x200')
## 
##lbl = Label(window, text="Hello")
## 
##lbl.grid(column=0, row=0)
## 
##def clicked():
## 
##    lbl.configure(text="Button was clicked !!")
##    
## 
##btn = Button(window, text="Load Data", command=clicked)
## 
##btn.grid(column=1, row=0)
## 
##window.mainloop()

from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
from tkcalendar import Calendar, DateEntry

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    import Tkinter as tk
    import ttk
 
root = Tk()
root.title("Tk dropdown example")
#root.geometry("600x600")
root.configure(background='grey')

#file path
path = "stock.jpg"
img = ImageTk.PhotoImage(Image.open(path))

# Add a grid
mainframe = Frame(root)
mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
mainframe.columnconfigure(0, weight = 1)
mainframe.rowconfigure(0, weight = 1)
#mainframe.pack(pady = 100, padx = 100)
mainframe.configure(background='grey')

root.resizable(0, 0)
# Create a Tkinter variable
tkvar = StringVar(root)

# Dictionary with options
choices = { 'Google','Microsoft','Apple Inc','Amazon','Facebook'}
tkvar.set('Click') # set the default option
 
popupMenu = OptionMenu(mainframe, tkvar, *choices)
Label(mainframe, text="Choose a Commodity").grid(row = 0, column = 0)
popupMenu.grid(row = 1, column =0)

###

lbl = Label(mainframe, text="")
 
lbl.grid(column=3, row=3)
def clicked():

    if (tkvar.get() in choices):
        lbl.configure(text="Data Loaded")
        loadimg()
    else:
        messagebox.showerror("Error", "Select a Commodity first!")

    
 
btn = Button(mainframe, text="Load Data", command=clicked)
 
btn.grid(column=0, row=3)
btn.grid_propagate(False)
 
# on change dropdown value
def change_dropdown(*args):
    print( tkvar.get() )
 
# link function to change dropdown
tkvar.trace('w', change_dropdown)
 




def example1():
    def print_sel():
        print(cal.selection_get())

    top = tk.Toplevel(root)

    cal = Calendar(top,
                   font="Arial 14", selectmode='day',
                   cursor="hand1", year=2018, month=2, day=5)
    cal.pack(fill="both", expand=True)
    ttk.Button(top, text="ok", command=print_sel).pack()

def example2():
    top = tk.Toplevel(root)

    ttk.Label(top, text='Choose date').pack(padx=10, pady=10)

    cal = DateEntry(top, width=12, background='darkblue',
                    foreground='white', borderwidth=2)
    cal.pack(padx=10, pady=10)

#root = tk.Tk()
#s = ttk.Style(root)
#s.theme_use('clam')

def loadimg():

    top = tk.Toplevel(root)
    panel = tk.Label(top, image = img)

#The Pack geometry manager packs widgets in rows or columns.
#panel.pack(side = "bottom", fill = "both", expand = "yes")
    panel.grid(row = 0, column = 0)

ttk.Button(mainframe, text='Start Date', command=example2).grid(row = 4, column = 0)
ttk.Button(mainframe, text='End Date', command=example2).grid(row = 5, column = 0)

root.mainloop()
