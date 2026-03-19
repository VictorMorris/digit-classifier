import glob
import os
import random
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator

from network import Network
from readdata import MnistDataloader

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


# Locations of data
TRAINING_IMAGES = 'input/train-images.idx3-ubyte'
TRAINING_LABELS = 'input/train-labels.idx1-ubyte'
TEST_IMAGES     = 'input/t10k-images.idx3-ubyte'
TEST_LABELS     = 'input/t10k-labels.idx1-ubyte'


class App:
    CELL_SIZE = 10  # pixels per cell in the drawing canvas

    def __init__(self, root, x_train, y_train, network, training_data, test_data):
        self.root = root
        self.x_train = x_train
        self.y_train = y_train
        self.network = network
        self.training_data = training_data
        self.test_data = test_data
        self.current_index = random.randint(0, len(self.x_train) - 1)
        self.num_correct = 0
        self.num_guess = 0
        self.is_training = False
        self._graph_dirty = False
        self._lock = threading.Lock()
        self.flash_time = 10  # ms between image flashes during training
        self.draw_pixels = np.zeros((28, 28))  # drawing canvas pixel data

        root.title("Digit Classifier")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self._build_config_frame()
        self._build_training_frame()
        self._build_results_frame()

        self._show_frame(self.config_frame)

    def _show_frame(self, frame):
        """Hide all frames and show the given one."""
        for f in (self.config_frame, self.training_frame, self.results_frame):
            f.grid_remove()
        frame.grid(row=0, column=0)

    # Config screen 
    def _build_config_frame(self):
        self.config_frame = ttk.Frame(self.root, padding="60 60 60 60")
        self.config_frame.columnconfigure(0, weight=1)
        self.config_frame.columnconfigure(1, weight=1)

        ttk.Label(self.config_frame, text="Digit Classifier", font=("Arial", 24, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 40))

        # Mode selection: Train New or Load Existing
        self.mode_var = tk.StringVar(value="train")
        mode_frame = ttk.Frame(self.config_frame)
        mode_frame.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        ttk.Radiobutton(mode_frame, text="Train New Network", variable=self.mode_var,
                        value="train", command=self._on_mode_change).grid(row=0, column=0, padx=15)
        ttk.Radiobutton(mode_frame, text="Load Existing Network", variable=self.mode_var,
                        value="load", command=self._on_mode_change).grid(row=0, column=1, padx=15)

        # Training config fields
        self.train_fields_frame = ttk.Frame(self.config_frame)
        self.train_fields_frame.grid(row=2, column=0, columnspan=2)
        fields = [
            ("Epochs",        "30",  "epochs_var"),
            ("Batch Size",    "10",  "batch_var"),
            ("Learning Rate", "3.0", "lr_var"),
        ]
        for i, (label, default, attr) in enumerate(fields):
            ttk.Label(self.train_fields_frame, text=f"{label}:", font=("Arial", 12)).grid(
                row=i, column=0, sticky=tk.E, padx=10, pady=8)
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(self.train_fields_frame, textvariable=var, width=10).grid(
                row=i, column=1, sticky=tk.W, pady=8)

        # Load model selection
        self.load_frame = ttk.Frame(self.config_frame)
        ttk.Label(self.load_frame, text="Select Model:", font=("Arial", 12)).grid(
            row=0, column=0, sticky=tk.E, padx=10, pady=8)
        self.model_listbox = tk.Listbox(self.load_frame, width=45, height=8, font=("Arial", 10))
        self.model_listbox.grid(row=1, column=0, columnspan=2, pady=8)
        self._refresh_model_list()

        # Start / Load button
        self.action_btn = tk.Button(self.config_frame, text="Start Training", font=("Arial", 14),
                                    command=self._on_action)
        self.action_btn.grid(row=4, column=0, columnspan=2, pady=40)

    def _on_mode_change(self):
        if self.mode_var.get() == "train":
            self.load_frame.grid_remove()
            self.train_fields_frame.grid(row=2, column=0, columnspan=2)
            self.action_btn.config(text="Start Training")
        else:
            self.train_fields_frame.grid_remove()
            self._refresh_model_list()
            self.load_frame.grid(row=2, column=0, columnspan=2)
            self.action_btn.config(text="Load Network")

    def _refresh_model_list(self):
        self.model_listbox.delete(0, tk.END)
        self._model_files = []
        if os.path.isdir(MODELS_DIR):
            files = sorted(glob.glob(os.path.join(MODELS_DIR, '*.npz')),
                           key=os.path.getmtime, reverse=True)
            for f in files:
                self._model_files.append(f)
                self.model_listbox.insert(tk.END, os.path.basename(f))

    def _on_action(self):
        if self.mode_var.get() == "train":
            self._start_training()
        else:
            self._load_network()

    # Training screen 
    def _build_training_frame(self):
        self.training_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.training_frame.columnconfigure(0, weight=1)

        self.fig_img, self.ax_img = plt.subplots()
        self.show_image()
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=self.training_frame)
        self.canvas_img.draw()

        self.fig_graph, self.ax_graph = plt.subplots()
        self.canvas_graph = FigureCanvasTkAgg(self.fig_graph, master=self.training_frame)
        self.show_graph()
        self.canvas_graph.draw()

        self.training_label = tk.Label(self.training_frame, text="Network Training...", font=("Arial", 14))

        self.canvas_img.get_tk_widget().grid(row=0, column=0)
        self.canvas_graph.get_tk_widget().grid(row=0, column=1)
        self.training_label.grid(row=1, column=0, columnspan=2, pady=10)

    # Results screen
    def _build_results_frame(self):
        self.results_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.results_frame.columnconfigure(0, weight=1)

        # Test image display
        self.fig_result_img, self.ax_result_img = plt.subplots()
        self.canvas_result_img = FigureCanvasTkAgg(self.fig_result_img, master=self.results_frame)
        self.canvas_result_img.draw()
        self.canvas_result_img.get_tk_widget().grid(row=0, column=0)

        # Buttons
        self.results_buttons_frame = ttk.Frame(self.results_frame)
        tk.Button(self.results_buttons_frame, text="New Image", command=self.new_image).grid(
            row=0, column=0, padx=5)
        tk.Button(self.results_buttons_frame, text="Back to Config", command=self._back_to_config).grid(
            row=0, column=1, padx=5)
        self.results_buttons_frame.grid(row=1, column=0, pady=5)

        self.prediction_label = tk.Label(self.results_frame, text="", font=("Arial", 14))
        self.prediction_label.grid(row=2, column=0, pady=5)
        self.result_label = tk.Label(self.results_frame, text="", font=("Arial", 14))
        self.result_label.grid(row=3, column=0, pady=5)

        # Drawing panel
        self._build_draw_panel()
        self.draw_frame.grid(row=0, column=1, rowspan=4, padx=(10, 0), sticky="n")

    def _build_draw_panel(self):
        fig_w, fig_h = self.fig_result_img.get_size_inches()
        fig_dpi = self.fig_result_img.get_dpi()
        canvas_size = int(min(fig_w, fig_h) * fig_dpi)
        self.CELL_SIZE = canvas_size // 28
        cs = self.CELL_SIZE
        self.draw_frame = ttk.LabelFrame(self.results_frame, text="Draw a Digit", padding="10 10 10 10")

        self.draw_canvas = tk.Canvas(
            self.draw_frame, width=28 * cs, height=28 * cs,
            bg="black", cursor="crosshair",
        )
        self.draw_canvas.grid(row=0, column=0, columnspan=3)

        # Mouse bindings
        self.draw_canvas.bind("<B1-Motion>", self._draw_on_canvas)
        self.draw_canvas.bind("<Button-1>", self._draw_on_canvas)
        self.draw_canvas.bind("<B3-Motion>", self._erase_on_canvas)
        self.draw_canvas.bind("<Button-3>", self._erase_on_canvas)

        # Buttons
        btn_row = ttk.Frame(self.draw_frame)
        btn_row.grid(row=1, column=0, columnspan=3, pady=(8, 0))
        tk.Button(btn_row, text="Reset", command=self._draw_reset).grid(row=0, column=0, padx=4)
        tk.Button(btn_row, text="Randomize", command=self._draw_randomize).grid(row=0, column=1, padx=4)
        tk.Button(btn_row, text="Predict", command=self._draw_predict).grid(row=0, column=2, padx=4)

        self.draw_prediction_label = tk.Label(self.draw_frame, text="", font=("Arial", 14))
        self.draw_prediction_label.grid(row=2, column=0, columnspan=3, pady=(6, 0))

    # Drawing helpers
    def _render_draw_canvas(self):
        cs = self.CELL_SIZE
        self.draw_canvas.delete("all")
        for y in range(28):
            for x in range(28):
                color = "white" if self.draw_pixels[y][x] else "black"
                self.draw_canvas.create_rectangle(
                    x * cs, y * cs, (x + 1) * cs, (y + 1) * cs,
                    fill=color, outline="",
                )

    BRUSH_SIZE = 2  # brush width in pixels

    def _set_pixel(self, event, value):
        cs = self.CELL_SIZE
        cx, cy = event.x // cs, event.y // cs
        for dy in range(self.BRUSH_SIZE):
            for dx in range(self.BRUSH_SIZE):
                gx, gy = cx + dx - self.BRUSH_SIZE // 2, cy + dy - self.BRUSH_SIZE // 2
                if 0 <= gx < 28 and 0 <= gy < 28:
                    self.draw_pixels[gy][gx] = value
                    color = "white" if value else "black"
                    self.draw_canvas.create_rectangle(
                        gx * cs, gy * cs, (gx + 1) * cs, (gy + 1) * cs,
                        fill=color, outline="",
                    )

    def _draw_on_canvas(self, event):
        self._set_pixel(event, 1)

    def _erase_on_canvas(self, event):
        self._set_pixel(event, 0)

    def _draw_reset(self):
        self.draw_pixels = np.zeros((28, 28))
        self._render_draw_canvas()
        self.draw_prediction_label.config(text="")

    def _draw_randomize(self):
        self.draw_pixels = np.random.default_rng().integers(low=0, high=2, size=(28, 28)).astype(float)
        self._render_draw_canvas()
        self.draw_prediction_label.config(text="")

    def _draw_predict(self):
        input_vec = self.draw_pixels.reshape(784, 1).astype(float)
        activations = self.network.feed_forward(input_vec)
        guess = int(np.argmax(activations))
        self.draw_prediction_label.config(text=f"Network thinks: {guess}")

    # Actions
    def _load_network(self):
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a model to load.")
            return
        filepath = self._model_files[selection[0]]
        self.network = Network.load(filepath)
        self._go_to_results()

    def _start_training(self):
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_var.get())
            lr = float(self.lr_var.get())
        except ValueError:
            return

        self._train_epochs = epochs
        self._train_batch = batch_size
        self._train_lr = lr

        self.network.test_errors = np.array([])
        self.is_training = True
        self._show_frame(self.training_frame)
        self.show_graph()
        self.canvas_graph.draw()
        self.root.after(50, self._training_loop)

        def run_training():
            self.network.SGD(
                self.training_data, epochs, batch_size, lr,
                test_data=self.test_data,
                epoch_callback=self.on_epoch_complete,
            )
            self.root.after(0, self._training_done)

        threading.Thread(target=run_training, daemon=True).start()

    def _back_to_config(self):
        self.num_correct = 0
        self.num_guess = 0
        self.network = Network([784, 16, 16, 10])
        self._draw_reset()
        self._refresh_model_list()
        self._show_frame(self.config_frame)

    def _go_to_results(self):
        self._show_result_image()
        self.canvas_result_img.draw()
        self.predict()
        self._render_draw_canvas()
        self._show_frame(self.results_frame)

    # Image / graph helpers
    def show_image(self):
        image = self.x_train[self.current_index]
        label = self.y_train[self.current_index]
        self.ax_img.clear()
        self.ax_img.imshow(image.reshape(28, 28), cmap=plt.cm.gray)
        if label != '':
            self.ax_img.set_title(f"Label: {label}", fontsize=15)

    def _show_result_image(self):
        image = self.x_train[self.current_index]
        label = self.y_train[self.current_index]
        self.ax_result_img.clear()
        self.ax_result_img.imshow(image.reshape(28, 28), cmap=plt.cm.gray)
        if label != '':
            self.ax_result_img.set_title(f"Label: {label}", fontsize=15)

    def show_graph(self):
        self.ax_graph.clear()
        self.ax_graph.set_title("Error Rate per Epoch")
        self.ax_graph.set_ylabel("Error Rate")
        self.ax_graph.set_xlabel("Epoch")
        self.ax_graph.xaxis.set_major_locator(MaxNLocator(integer=True))
        errors = self.network.test_errors
        epochs = list(range(0, len(errors)))
        self.ax_graph.set_xlim(-0.5, max(len(errors), 2) + 0.5)
        self.ax_graph.plot(epochs, errors)

    # Training callbacks / loop
    def on_epoch_complete(self, _test_error):
        """Called from the training thread after each epoch."""
        with self._lock:
            self._graph_dirty = True

    def _training_done(self):
        """Called on the main thread when training finishes."""
        self.is_training = False
        self.show_graph()
        self.canvas_graph.draw()
        # Prompt to save before transitioning
        if hasattr(self, '_train_epochs'):
            self._prompt_save()
            del self._train_epochs
        self._go_to_results()

    def _prompt_save(self):
        correct = self.network.evaluate(self.test_data)
        total = len(self.test_data)
        accuracy = 100 * correct / total
        msg = (f"Training complete!\n\n"
               f"Accuracy on test data: {correct}/{total} ({accuracy:.2f}%)\n\n"
               f"Would you like to save this network?")
        if messagebox.askyesno("Save Network", msg):
            filename = (f"network_{self._train_epochs}epochs_"
                        f"{self._train_batch}batch_{self._train_lr}eta.npz")
            filepath = os.path.join(MODELS_DIR, filename)
            self.network.save(filepath)
            messagebox.showinfo("Saved", f"Network saved to:\n{filename}")

    def _training_loop(self):
        """Runs on the main thread via after(); cycles images and refreshes the graph."""
        self.current_index = random.randint(0, len(self.x_train) - 1)
        self.show_image()
        self.canvas_img.draw()

        with self._lock:
            dirty = self._graph_dirty
            self._graph_dirty = False
        if dirty:
            self.show_graph()
            self.canvas_graph.draw()

        if self.is_training:
            self.root.after(self.flash_time, self._training_loop)

    # User interactions
    def new_image(self):
        self.current_index = random.randint(0, len(self.x_train) - 1)
        self._show_result_image()
        self.canvas_result_img.draw()
        self.predict()

    def predict(self):
        activations = self.network.feed_forward(self.x_train[self.current_index])
        print(activations)
        guess = list(activations).index(max(activations))
        if guess == self.y_train[self.current_index]:
            self.num_correct += 1
        self.num_guess += 1
        self.prediction_label.config(text=f"Network thinks: {guess}")
        self.result_label.config(text=f"{self.num_correct}/{self.num_guess}")


def _quit(root):
    root.quit()
    root.destroy()


def main():
    loader = MnistDataloader(TRAINING_IMAGES, TRAINING_LABELS, TEST_IMAGES, TEST_LABELS)
    (x_train, y_train), (x_test, y_test) = loader.load_data()

    network = Network([784, 16, 16, 10])

    def to_one_hot(label):
        v = np.zeros((10, 1))
        v[label] = 1.0
        return v

    training_data = [(x, to_one_hot(y)) for x, y in zip(x_train, y_train)]
    test_data = [(x, to_one_hot(y)) for x, y in zip(x_test, y_test)]

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: _quit(root))
    App(root, x_train, y_train, network, training_data, test_data)
    root.mainloop()


if __name__ == '__main__':
    main()
