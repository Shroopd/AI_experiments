from tkinter import *
from tkinter import ttk


class FeetToMeters:
    def __init__(self, root) -> None:
        root.title("Shadow Wizard Money Gang")

        self.mainframe = ttk.Frame(root, padding=(3, 3, 12, 12))

        self.mainframe.grid(column=0, row=0, sticky=NSEW)

        self.feet = StringVar()
        self.feet_entry = ttk.Entry(self.mainframe, width=7, textvariable=self.feet)
        self.feet_entry.grid(column=2, row=1, sticky=EW)

        self.meters = StringVar()
        ttk.Label(self.mainframe, textvariable=self.meters).grid(
            column=2, row=2, sticky=EW
        )

        ttk.Button(self.mainframe, text="Calculate", command=self.calc).grid(
            column=3, row=3
        )

        ttk.Label(self.mainframe, text="Feet").grid(row=1, column=3, sticky=W)
        ttk.Label(self.mainframe, text="is equivalent to").grid(
            column=1, row=2, sticky=E
        )
        ttk.Label(self.mainframe, text="meters").grid(column=3, row=2, sticky=W)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.mainframe.columnconfigure(2, weight=1)
        for child in self.mainframe.winfo_children():
            assert isinstance(child, Widget)
            child.grid_configure(padx=5, pady=5)

    def calc(self, *args):
        try:
            value = float(self.feet.get())
            self.meters.set(str(round(0.3048 * value, 4)))
        except ValueError:
            pass


root = Tk()
FeetToMeters(root)
root.mainloop()
