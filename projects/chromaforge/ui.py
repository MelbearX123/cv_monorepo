import tkinter as tk
from tkinter import filedialog
import cv2

from segment import colourSegment
from extract import colourExtract
from merge import mergeLayers

root = tk.Tk()
root.title("ChromaForge")
root.geometry("300x300")

mode = tk.StringVar(value="flatten")
k_value = tk.IntVar(value=6)
selected_files = []

def browse_files():
    if mode.get() == "merge":
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg")])
    else:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg")])
        paths = (path,) if path else ()
    selected_files.clear()
    selected_files.extend(paths)
    file_label.config(text=f"{len(selected_files)} file(s) selected")

def run():
    if not selected_files:
        return

    if mode.get() == "flatten":
        img = cv2.imread(selected_files[0])
        final_img, _ = colourSegment(img, k=k_value.get())
        out = filedialog.asksaveasfilename(defaultextension=".png")
        if out:
            cv2.imwrite(out, final_img)

    elif mode.get() == "layers":
        img = cv2.imread(selected_files[0])
        final_img, labels = colourSegment(img, k=k_value.get())
        layers = colourExtract(final_img, labels)
        folder = filedialog.askdirectory()
        if folder:
            for i, layer in enumerate(layers):
                cv2.imwrite(f"{folder}/layer_{i}.png", layer)

    elif mode.get() == "merge":
        layers = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in selected_files]
        merged = mergeLayers(layers)
        out = filedialog.asksaveasfilename(defaultextension=".png")
        if out:
            cv2.imwrite(out, merged)

def update_k_visibility(*args):
    if mode.get() == "flatten":
        k_label.pack(pady=(10, 0), before=select_button)
        k_scale.pack(before=select_button)
    else:
        k_label.pack_forget()
        k_scale.pack_forget()

tk.Radiobutton(root, text="Flatten colours", variable=mode, value="flatten", command=update_k_visibility).pack()
tk.Radiobutton(root, text="Flatten + split layers", variable=mode, value="layers", command=update_k_visibility).pack()
tk.Radiobutton(root, text="Merge layers", variable=mode, value="merge", command=update_k_visibility).pack()

k_label = tk.Label(root, text="How many colours?")
k_scale = tk.Scale(root, from_=2, to=20, orient=tk.HORIZONTAL, variable=k_value)

select_button = tk.Button(root, text="Select file(s)", command=browse_files)
select_button.pack(pady=5)
file_label = tk.Label(root, text="0 file(s) selected")
file_label.pack()

update_k_visibility()

tk.Button(root, text="Run", command=run).pack(pady=10)

root.mainloop()