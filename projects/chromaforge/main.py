import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

from segment import colourSegment
from extract import colourExtract
from merge import mergeLayers


def main():
    root = tk.Tk()
    root.title("ChromaForge")
    root.geometry("300x450")

    mode = tk.StringVar(value="flatten")
    k_value = tk.IntVar(value=6)
    sp_value = tk.IntVar(value=20)
    sr_value = tk.IntVar(value=45)
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
            final_img, _ = colourSegment(
                img, sp=sp_value.get(), sr=sr_value.get(), k=k_value.get()
            )
            out = filedialog.asksaveasfilename(defaultextension=".png")
            if out:
                cv2.imwrite(out, final_img)

        elif mode.get() == "split":
            # Expects an already-flattened image
            img = cv2.imread(selected_files[0])
            flat = img.reshape(-1, 3)
            _, labels_flat = np.unique(flat, axis=0, return_inverse=True)
            labels = labels_flat.reshape(img.shape[:2])

            layers = colourExtract(img, labels)
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

    def update_param_visibility(*args):
        if mode.get() == "flatten":
            sp_label.pack(pady=(10, 0), before=select_button)
            sp_scale.pack(before=select_button)
            sr_label.pack(pady=(10, 0), before=select_button)
            sr_scale.pack(before=select_button)
            k_label.pack(pady=(10, 0), before=select_button)
            k_scale.pack(before=select_button)
        else:
            sp_label.pack_forget()
            sp_scale.pack_forget()
            sr_label.pack_forget()
            sr_scale.pack_forget()
            k_label.pack_forget()
            k_scale.pack_forget()

        if mode.get() == "split":
            split_note.pack(pady=(10, 0), before=select_button)
        else:
            split_note.pack_forget()

    tk.Radiobutton(
        root,
        text="Flatten colours",
        variable=mode,
        value="flatten",
        command=update_param_visibility,
    ).pack()
    tk.Radiobutton(
        root,
        text="Split into layers",
        variable=mode,
        value="split",
        command=update_param_visibility,
    ).pack()
    tk.Radiobutton(
        root,
        text="Merge layers",
        variable=mode,
        value="merge",
        command=update_param_visibility,
    ).pack()

    sp_label = tk.Label(root, text="Blob size (how far away to merge)")
    sp_scale = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, variable=sp_value)

    sr_label = tk.Label(root, text="Colour tolerance (how similar counts as same)")
    sr_scale = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL, variable=sr_value)

    k_label = tk.Label(root, text="How many colours?")
    k_scale = tk.Scale(root, from_=2, to=20, orient=tk.HORIZONTAL, variable=k_value)

    split_note = tk.Label(root, text="Only works on a flattened image")

    select_button = tk.Button(root, text="Select file(s)", command=browse_files)
    select_button.pack(pady=5)
    file_label = tk.Label(root, text="0 file(s) selected")
    file_label.pack()

    update_param_visibility()

    tk.Button(root, text="Run", command=run).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
