import tkinter as tk 
from tkinter import ttk, filedialog, messagebox 
import pandas as pd 
import numpy as np 
from PIL import Image, ImageTk 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
import os 
import cv2 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix, classification_report 
import matplotlib 
matplotlib.use('TkAgg') 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from matplotlib.figure import Figure 
import matplotlib.pyplot as plt 
import seaborn as sns 
 
class ImageClassificationSystem: 
    def __init__(self, root): 
        self.root = root 
        self.root.title("Image Classification System - Deep Learning") 
        self.root.geometry("1000x800") 
        self.root.resizable(True, True) 
         
        # Model and data variables 
        self.model = None 
        self.label_encoder = None 
        self.class_names = [] 
        self.image_size = (160, 160)
        self.training_history = None 
        self.class_distribution = {} 
         
        # Configure style 
        style = ttk.Style() 
        style.theme_use('clam') 
         
        self.create_widgets() 
         
    def create_widgets(self): 
        # Create a canvas and scrollbar for the main window 
        self.main_canvas = tk.Canvas(self.root, highlightthickness=0) 
        self.main_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview) 
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set) 
         
        self.main_scrollbar.pack(side="right", fill="y") 
        self.main_canvas.pack(side="left", fill="both", expand=True) 
         
        # Main container inside the canvas 
        main_container = ttk.Frame(self.main_canvas, padding="15") 
        self.main_canvas_window = self.main_canvas.create_window((0, 0), window=main_container, anchor="nw") 
         
        def _on_frame_configure(event): 
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")) 
             
        def _on_canvas_configure(event): 
            # Update the width of the frame to fill the canvas 
            self.main_canvas.itemconfig(self.main_canvas_window, width=event.width) 
             
        main_container.bind("<Configure>", _on_frame_configure) 
        self.main_canvas.bind("<Configure>", _on_canvas_configure) 
         
        # Bind mouse wheel 
        def _on_mouse_wheel(event): 
            if event.num == 4 or event.delta > 0: 
                self.main_canvas.yview_scroll(-1, "units") 
            elif event.num == 5 or event.delta < 0: 
                self.main_canvas.yview_scroll(1, "units") 
                 
        self.main_canvas.bind_all("<MouseWheel>", _on_mouse_wheel) 
        self.main_canvas.bind_all("<Button-4>", _on_mouse_wheel) 
        self.main_canvas.bind_all("<Button-5>", _on_mouse_wheel) 
         
        main_container.columnconfigure(0, weight=1) 
        main_container.rowconfigure(3, weight=1) 
         
        # Header 
        header = ttk.Label(main_container, text="Image Classification System",  
                          font=('Arial', 18, 'bold')) 
        header.grid(row=0, column=0, pady=(0, 15)) 
         
        subtitle = ttk.Label(main_container, text="Deep Learning Neural Network (TensorFlow/Keras)", 
                           font=('Arial', 10)) 
        subtitle.grid(row=1, column=0, pady=(0, 20)) 
         
        # Control Panel Frame 
        control_frame = ttk.LabelFrame(main_container, text="Controls", padding="10") 
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15)) 
        control_frame.columnconfigure(0, weight=1) 
         
        # Train Section 
        train_frame = ttk.Frame(control_frame) 
        train_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5) 
         
        ttk.Label(train_frame, text="Training:").grid(row=0, column=0, padx=5) 
        ttk.Button(train_frame, text="Load CSV & Train Model",  
                  command=self.load_and_train).grid(row=0, column=1, padx=5) 
        ttk.Button(train_frame, text="Save Model",  
                  command=self.save_model).grid(row=0, column=2, padx=5) 
        ttk.Button(train_frame, text="Load Model",  
                  command=self.load_model).grid(row=0, column=3, padx=5) 
         
        # Classification Section 
        classify_frame = ttk.Frame(control_frame) 
        classify_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5) 
         
        ttk.Label(classify_frame, text="Classification:").grid(row=0, column=0, padx=5) 
        ttk.Button(classify_frame, text="Load Single Image",  
                  command=self.classify_single_image).grid(row=0, column=1, padx=5) 
        ttk.Button(classify_frame, text="Batch Classify Folder",  
                  command=self.batch_classify).grid(row=0, column=2, padx=5) 
         
        # Info Label 
        self.info_label = ttk.Label(control_frame, text="Ready", foreground="gray") 
        self.info_label.grid(row=2, column=0, pady=(10, 0)) 
         
        # Middle container for side-by-side display 
        mid_container = ttk.Frame(main_container) 
        mid_container.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10)) 
        mid_container.columnconfigure(0, weight=1) 
        mid_container.columnconfigure(1, weight=1) 
        mid_container.rowconfigure(0, weight=1) 
 
        # Display Area 
        display_frame = ttk.LabelFrame(mid_container, text="Image Display", padding="10") 
        display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5)) 
        display_frame.columnconfigure(0, weight=1) 
        display_frame.rowconfigure(0, weight=1) 
         
        # Canvas for image display 
        self.canvas = tk.Canvas(display_frame, bg='#f0f0f0', width=640, height=440, highlightthickness=0) 
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) 
         
        # Scrollbar for canvas 
        scrollbar = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.canvas.yview) 
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S)) 
        self.canvas.configure(yscrollcommand=scrollbar.set) 
        self.canvas.bind('<Configure>', lambda event: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
         
        # Results Area 
        results_frame = ttk.LabelFrame(mid_container, text="Classification Results", padding="10") 
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0)) 
        results_frame.columnconfigure(0, weight=1) 
        results_frame.rowconfigure(0, weight=1) 
         
        # Text widget for results 
        self.results_text = tk.Text(results_frame, wrap=tk.WORD) 
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) 
         
        scrollbar_text = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview) 
        scrollbar_text.grid(row=0, column=1, sticky=(tk.N, tk.S)) 
        self.results_text.configure(yscrollcommand=scrollbar_text.set) 
         
        # Visualization Area 
        self.viz_frame = ttk.LabelFrame(main_container, text="Visualizations", padding="10") 
        self.viz_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0)) 
        self.viz_frame.columnconfigure(0, weight=1) 
        self.viz_frame.columnconfigure(1, weight=1) 
        self.viz_frame.rowconfigure(0, weight=1) 
         
        # Training History Plot 
        self.create_training_plot() 
         
        # Class Distribution Plot 
        self.create_class_distribution_plot() 
     
    def create_training_plot(self): 
        """Create training history visualization frame""" 
        plot_frame = ttk.LabelFrame(self.viz_frame, text="Training History", padding="5") 
        plot_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5)) 
        plot_frame.columnconfigure(0, weight=1) 
        plot_frame.rowconfigure(0, weight=1) 
         
        self.training_canvas = tk.Canvas(plot_frame, bg='#f5f5f5', width=680, height=320, highlightthickness=0) 
        self.training_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) 
        training_v_scroll = ttk.Scrollbar(plot_frame, orient=tk.VERTICAL, command=self.training_canvas.yview) 
        training_v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S)) 
        training_h_scroll = ttk.Scrollbar(plot_frame, orient=tk.HORIZONTAL, command=self.training_canvas.xview) 
        training_h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E)) 
        self.training_canvas.configure(yscrollcommand=training_v_scroll.set, xscrollcommand=training_h_scroll.set) 
         
        training_inner = ttk.Frame(self.training_canvas) 
        self.training_canvas.create_window((0, 0), window=training_inner, anchor='nw') 
        training_inner.bind("<Configure>", lambda event: self.training_canvas.configure(scrollregion=self.training_canvas.bbox("all"))) 
         
        self.fig_training = Figure(figsize=(10, 4.5), facecolor='#f5f5f5') 
        self.ax_training = self.fig_training.add_subplot(111) 
        self.ax_training.set_title('Accuracy & Loss Over Epochs') 
        self.ax_training.set_xlabel('Epoch') 
        self.ax_training.set_ylabel('Value') 
        self.ax_training.grid(True, alpha=0.3) 
         
        self.canvas_training = FigureCanvasTkAgg(self.fig_training, training_inner) 
        widget_training = self.canvas_training.get_tk_widget() 
        widget_training.pack() 
        widget_training.config(width=960, height=360) 
     
    def create_class_distribution_plot(self): 
        """Create class distribution visualization frame""" 
        dist_frame = ttk.LabelFrame(self.viz_frame, text="Class Distribution", padding="5") 
        dist_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0)) 
        dist_frame.columnconfigure(0, weight=1) 
        dist_frame.rowconfigure(0, weight=1) 
         
        self.dist_canvas = tk.Canvas(dist_frame, bg='#f5f5f5', width=680, height=320, highlightthickness=0) 
        self.dist_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) 
        dist_v_scroll = ttk.Scrollbar(dist_frame, orient=tk.VERTICAL, command=self.dist_canvas.yview) 
        dist_v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S)) 
        dist_h_scroll = ttk.Scrollbar(dist_frame, orient=tk.HORIZONTAL, command=self.dist_canvas.xview) 
        dist_h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E)) 
        self.dist_canvas.configure(yscrollcommand=dist_v_scroll.set, xscrollcommand=dist_h_scroll.set) 
         
        dist_inner = ttk.Frame(self.dist_canvas) 
        self.dist_canvas.create_window((0, 0), window=dist_inner, anchor='nw') 
        dist_inner.bind("<Configure>", lambda event: self.dist_canvas.configure(scrollregion=self.dist_canvas.bbox("all"))) 
         
        self.fig_dist = Figure(figsize=(10, 4.5), facecolor='#f5f5f5') 
        self.ax_dist = self.fig_dist.add_subplot(111) 
        self.ax_dist.set_title('Images per Class') 
        self.ax_dist.set_xlabel('Class') 
        self.ax_dist.set_ylabel('Count') 
        self.ax_dist.grid(True, alpha=0.3) 
         
        self.canvas_dist = FigureCanvasTkAgg(self.fig_dist, dist_inner) 
        widget_dist = self.canvas_dist.get_tk_widget() 
        widget_dist.pack() 
        widget_dist.config(width=960, height=360) 
     
    def update_training_plot(self, history): 
        """Update training history plot with new data""" 
        self.ax_training.clear() 
         
        epochs = range(1, len(history.history['accuracy']) + 1) 
         
        # Plot accuracy 
        self.ax_training.plot(epochs, history.history['accuracy'], 'b-', label='Train Accuracy', linewidth=2) 
        if 'val_accuracy' in history.history: 
            self.ax_training.plot(epochs, history.history['val_accuracy'], 'r--', label='Val Accuracy', linewidth=2) 
         
        self.ax_training.set_title('Model Accuracy') 
        self.ax_training.set_xlabel('Epoch') 
        self.ax_training.set_ylabel('Accuracy') 
        self.ax_training.legend(loc='lower right') 
        self.ax_training.grid(True, alpha=0.3) 
         
        self.canvas_training.draw() 
         
        # Also create loss plot in a new figure 
        self.fig_loss = Figure(figsize=(4, 3), facecolor='#f5f5f5') 
        self.ax_loss = self.fig_loss.add_subplot(111) 
         
        self.ax_loss.plot(epochs, history.history['loss'], 'b-', label='Train Loss', linewidth=2) 
        if 'val_loss' in history.history: 
            self.ax_loss.plot(epochs, history.history['val_loss'], 'r--', label='Val Loss', linewidth=2) 
         
        self.ax_loss.set_title('Model Loss') 
        self.ax_loss.set_xlabel('Epoch') 
        self.ax_loss.set_ylabel('Loss') 
        self.ax_loss.legend(loc='upper right') 
        self.ax_loss.grid(True, alpha=0.3) 
         
        # Show loss plot in a popup window 
        self.show_loss_window() 
     
    def show_loss_window(self): 
        """Display loss plot in a new window""" 
        loss_win = tk.Toplevel(self.root) 
        loss_win.title("Training Loss") 
        loss_win.geometry("500x400") 
         
        canvas_loss = FigureCanvasTkAgg(self.fig_loss, loss_win) 
        canvas_loss.get_tk_widget().pack(fill=tk.BOTH, expand=True) 
     
    def update_class_distribution(self, labels): 
        """Update class distribution bar chart""" 
        self.ax_dist.clear() 
         
        # Count images per class 
        class_counts = {} 
        for label in labels: 
            class_counts[label] = class_counts.get(label, 0) + 1 
         
        classes = list(class_counts.keys()) 
        counts = list(class_counts.values()) 
         
        # Create bar chart 
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes))) 
        bars = self.ax_dist.bar(classes, counts, color=colors) 
         
        self.ax_dist.set_title('Images per Class') 
        self.ax_dist.set_xlabel('Class') 
        self.ax_dist.set_ylabel('Count') 
        self.ax_dist.tick_params(axis='x', rotation=45) 
         
        # Add value labels on bars 
        for bar, count in zip(bars, counts): 
            self.ax_dist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,  
                             str(count), ha='center', va='bottom', fontsize=9) 
         
        self.ax_dist.grid(True, alpha=0.3, axis='y') 
        self.canvas_dist.draw() 
         
        self.class_distribution = class_counts 
     
    def show_evaluation_metrics(self, history, X_val, y_val): 
        """Show detailed evaluation metrics in a new window""" 
        eval_win = tk.Toplevel(self.root) 
        eval_win.title("Model Evaluation Metrics") 
        eval_win.geometry("700x500") 
         
        # Get predictions 
        y_pred = self.model.predict(X_val, verbose=0) 
        y_pred_classes = np.argmax(y_pred, axis=1) 
        y_true_classes = np.argmax(y_val, axis=1) 
         
        # Create notebook for tabs 
        notebook = ttk.Notebook(eval_win) 
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) 
         
        # Tab 1: Metrics Summary 
        metrics_frame = ttk.Frame(notebook) 
        notebook.add(metrics_frame, text="Metrics Summary") 
         
        # Calculate metrics 
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
         
        acc = accuracy_score(y_true_classes, y_pred_classes) 
        prec = precision_score(y_true_classes, y_pred_classes, average='weighted') 
        rec = recall_score(y_true_classes, y_pred_classes, average='weighted') 
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted') 
         
        # Display metrics 
        metrics_container = ttk.Frame(metrics_frame) 
        metrics_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) 
         
        metrics_text = tk.Text(metrics_container, wrap=tk.WORD, font=('Consolas', 11)) 
        metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) 
         
        metrics_scroll = ttk.Scrollbar(metrics_container, orient="vertical", command=metrics_text.yview) 
        metrics_scroll.pack(side=tk.RIGHT, fill=tk.Y) 
        metrics_text.configure(yscrollcommand=metrics_scroll.set) 
         
        metrics_content = f""" 
╔══════════════════════════════════════════════════════════════╣ 
                     MODEL EVALUATION METRICS                   
╠══════════════════════════════════════════════════════════════╣ 
   Accuracy:        {acc:.4f} ({acc*100:.2f}%)                           
   Precision:       {prec:.4f} ({prec*100:.2f}%)                          
   Recall:          {rec:.4f} ({rec*100:.2f}%)                           
   F1-Score:        {f1:.4f} ({f1*100:.2f}%)                           
╠══════════════════════════════════════════════════════════════╣ 
   Training Epochs: {len(history.history['accuracy'])}                                    
   Best Val Accuracy: {max(history.history.get('val_accuracy', [0])):.4f}                       
╚══════════════════════════════════════════════════════════════╣ 
""" 
        metrics_text.insert(tk.END, metrics_content) 
         
        # Tab 2: Confusion Matrix 
        cm_frame = ttk.Frame(notebook) 
        notebook.add(cm_frame, text="Confusion Matrix") 
         
        fig_cm = Figure(figsize=(6, 5), facecolor='white') 
        ax_cm = fig_cm.add_subplot(111) 
         
        cm = confusion_matrix(y_true_classes, y_pred_classes) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                   xticklabels=self.class_names, yticklabels=self.class_names) 
        ax_cm.set_title('Confusion Matrix') 
        ax_cm.set_xlabel('Predicted') 
        ax_cm.set_ylabel('Actual') 
         
        canvas_cm = FigureCanvasTkAgg(fig_cm, cm_frame) 
        canvas_cm.get_tk_widget().pack(fill=tk.BOTH, expand=True) 
         
        # Tab 3: Classification Report 
        report_frame = ttk.Frame(notebook) 
        notebook.add(report_frame, text="Classification Report") 
         
        report_container = ttk.Frame(report_frame) 
        report_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) 
         
        report_text = tk.Text(report_container, wrap=tk.WORD, font=('Consolas', 10)) 
        report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) 
         
        report_scroll = ttk.Scrollbar(report_container, orient="vertical", command=report_text.yview) 
        report_scroll.pack(side=tk.RIGHT, fill=tk.Y) 
        report_text.configure(yscrollcommand=report_scroll.set) 
         
        report = classification_report(y_true_classes, y_pred_classes,  
                                        target_names=self.class_names) 
        report_text.insert(tk.END, "CLASSIFICATION REPORT\n") 
        report_text.insert(tk.END, "="*60 + "\n\n") 
        report_text.insert(tk.END, report) 
         
        # Tab 4: Training Curves 
        curves_frame = ttk.Frame(notebook) 
        notebook.add(curves_frame, text="Training Curves") 
         
        fig_curves = Figure(figsize=(6, 5), facecolor='white') 
         
        # Accuracy subplot 
        ax_acc = fig_curves.add_subplot(211) 
        epochs = range(1, len(history.history['accuracy']) + 1) 
        ax_acc.plot(epochs, history.history['accuracy'], 'b-', label='Train', linewidth=2) 
        if 'val_accuracy' in history.history: 
            ax_acc.plot(epochs, history.history['val_accuracy'], 'r--', label='Validation', linewidth=2) 
        ax_acc.set_title('Accuracy Over Epochs') 
        ax_acc.set_xlabel('Epoch') 
        ax_acc.set_ylabel('Accuracy') 
        ax_acc.legend() 
        ax_acc.grid(True, alpha=0.3) 
         
        # Loss subplot 
        ax_loss = fig_curves.add_subplot(212) 
        ax_loss.plot(epochs, history.history['loss'], 'b-', label='Train', linewidth=2) 
        if 'val_loss' in history.history: 
            ax_loss.plot(epochs, history.history['val_loss'], 'r--', label='Validation', linewidth=2) 
        ax_loss.set_title('Loss Over Epochs') 
        ax_loss.set_xlabel('Epoch') 
        ax_loss.set_ylabel('Loss') 
        ax_loss.legend() 
        ax_loss.grid(True, alpha=0.3) 
         
        fig_curves.tight_layout() 
         
        canvas_curves = FigureCanvasTkAgg(fig_curves, curves_frame) 
        canvas_curves.get_tk_widget().pack(fill=tk.BOTH, expand=True) 
     
    def create_filename_mapping(self, folder_path): 
        """Create a mapping from ID to actual filename""" 
        mapping = {} 
        for filename in os.listdir(folder_path): 
            # Get filename without extension 
            name_without_ext = os.path.splitext(filename)[0] 
            mapping[name_without_ext] = filename 
            # Also store the full filename 
            mapping[filename] = filename 
        return mapping 
         
    def load_and_train(self): 
        """Load CSV and images, then train the neural network""" 
        csv_path = filedialog.askopenfilename( 
            title="Select CSV file", 
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")] 
        ) 
         
        if not csv_path: 
            return 
         
        # Ask for image folder 
        folder_path = filedialog.askdirectory(title="Select folder containing images") 
        if not folder_path: 
            return 
         
        try: 
            self.info_label.config(text="Loading data...", foreground="blue") 
            self.results_text.delete(1.0, tk.END)  # Clear previous results 
            self.root.update() 
             
            # Load CSV 
            df = pd.read_csv(csv_path) 
             
            # Display CSV structure for debugging 
            self.results_text.insert(tk.END, f"CSV Columns found: {list(df.columns)}\n") 
            self.results_text.insert(tk.END, f"Total rows in CSV: {len(df)}\n\n") 
             
            # Find image column (could be 'image', 'filename', 'id', etc.) 
            image_col = None 
            for col in ['image', 'filename', 'file_name', 'id', 'image_id']: 
                if col in df.columns: 
                    image_col = col 
                    break 
             
            if image_col is None: 
                messagebox.showerror("Error", f"CSV must contain an 'image', 'filename', or 'id' column.\nFound: {list(df.columns)}") 
                return 
             
            # Find label column (could be 'label', 'class', 'category', etc.) 
            label_col = None 
            for col in ['label', 'class', 'category', 'type']: 
                if col in df.columns: 
                    label_col = col 
                    break 
             
            if label_col is None: 
                messagebox.showerror("Error", f"CSV must contain a 'label', 'class', or 'category' column.\nFound: {list(df.columns)}") 
                return 
             
            self.results_text.insert(tk.END, f"Using '{image_col}' as image identifier column\n") 
            self.results_text.insert(tk.END, f"Using '{label_col}' as label column\n\n") 
             
            # Create filename mapping for faster lookup 
            filename_mapping = self.create_filename_mapping(folder_path) 
             
            # Load and preprocess images 
            images = [] 
            labels = [] 
            skipped_images = [] 
             
            # Supported image extensions 
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'} 
             
            for idx, row in df.iterrows(): 
                # Get image identifier from CSV 
                img_id = str(row[image_col]).strip() 
                 
                # Try different possible filename patterns 
                img_path = None 
                 
                # Pattern 1: Direct match with filename in mapping 
                if img_id in filename_mapping: 
                    potential_path = os.path.join(folder_path, filename_mapping[img_id]) 
                    if os.path.exists(potential_path): 
                        img_path = potential_path 
                 
                # Pattern 2: Try with common extensions 
                if img_path is None: 
                    for ext in image_extensions: 
                        potential_path = os.path.join(folder_path, img_id + ext) 
                        if os.path.exists(potential_path): 
                            img_path = potential_path 
                            break 
                         
                        # Pattern 3: Try with underscore variations 
                        potential_path = os.path.join(folder_path, img_id.replace('-', '_') + ext) 
                        if os.path.exists(potential_path): 
                            img_path = potential_path 
                            break 
                 
                if img_path and os.path.exists(img_path): 
                    img = cv2.imread(img_path) 
                    if img is not None: 
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                        img = cv2.resize(img, self.image_size) 
                        img = img / 255.0  # Normalize 
                        images.append(img) 
                        labels.append(row[label_col]) 
                    else: 
                        skipped_images.append(f"{img_id} (corrupted file)") 
                else: 
                    if idx < 10:  # Show first 10 missing images as example 
                        skipped_images.append(f"{img_id} (file not found)") 
             
            if len(images) == 0: 
                error_msg = f"No valid images found!\nChecked in folder: {folder_path}\n" 
                error_msg += f"First few image IDs from CSV: {df[image_col].head(5).tolist()}\n" 
                error_msg += f"Files in folder: {os.listdir(folder_path)[:5]}" 
                messagebox.showerror("Error", error_msg) 
                self.info_label.config(text="No valid images found", foreground="red") 
                return 
             
            # Show summary of loaded images 
            self.results_text.insert(tk.END, f"Successfully loaded {len(images)} images\n") 
            if skipped_images: 
                self.results_text.insert(tk.END, f"Warning: Skipped {len(skipped_images)} images:\n") 
                for skip in skipped_images[:10]:  # Show first 10 
                    self.results_text.insert(tk.END, f"  - {skip}\n") 
                self.results_text.insert(tk.END, "\n") 
             
            # Filter out 'Not sure' or other ambiguous labels 
            ambiguous_labels = ['Not sure', 'Unsure', 'Unknown', '?', 'none', 'None'] 
            filtered_images = [] 
            filtered_labels = [] 
             
            for img, lbl in zip(images, labels): 
                if lbl not in ambiguous_labels and pd.notna(lbl) and str(lbl).strip(): 
                    filtered_images.append(img) 
                    filtered_labels.append(str(lbl).strip()) 
             
            removed_count = len(images) - len(filtered_images) 
            if removed_count > 0: 
                self.results_text.insert(tk.END, f"Removed {removed_count} images with ambiguous labels (e.g., 'Not sure')\n") 
                images = filtered_images 
                labels = filtered_labels 
             
            if len(images) == 0: 
                messagebox.showerror("Error", "No valid images after filtering ambiguous labels!") 
                return 
             
            # Show unique labels found 
            unique_labels = set(labels) 
            self.results_text.insert(tk.END, f"\nUnique labels found: {sorted(unique_labels)}\n") 
            self.results_text.insert(tk.END, f"Total classes: {len(unique_labels)}\n\n") 
             
            # Encode labels 
            self.label_encoder = LabelEncoder() 
            labels_encoded = self.label_encoder.fit_transform(labels) 
            self.class_names = self.label_encoder.classes_ 
            num_classes = len(self.class_names) 
             
            # Check if we have enough classes 
            if num_classes < 2: 
                messagebox.showerror("Error", f"Need at least 2 classes for classification. Found: {self.class_names}") 
                return 
             
            # Convert to numpy arrays 
            X = np.array(images) 
            y = tf.keras.utils.to_categorical(labels_encoded, num_classes) 
             
            # Split data (stratify to maintain class distribution) 
            X_train, X_val, y_train, y_val = train_test_split( 
                X, y, test_size=0.2, random_state=42, stratify=labels_encoded 
            ) 
             
            self.info_label.config(text=f"Training on {len(X_train)} images from {num_classes} classes...",  
                                  foreground="orange") 
            self.root.update() 
             
            # Build model (adjusted for smaller dataset) 
            self.model = keras.Sequential([ 
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)), 
                layers.MaxPooling2D((2, 2)), 
                layers.Conv2D(64, (3, 3), activation='relu'), 
                layers.MaxPooling2D((2, 2)), 
                layers.Conv2D(64, (3, 3), activation='relu'), 
                layers.MaxPooling2D((2, 2)), 
                layers.Flatten(), 
                layers.Dense(64, activation='relu'), 
                layers.Dropout(0.5), 
                layers.Dense(num_classes, activation='softmax') 
            ]) 
             
            # Compile model 
            self.model.compile( 
                optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'] 
            ) 
             
            # Add callbacks 
            callbacks = [ 
                keras.callbacks.EarlyStopping( 
                    monitor='val_loss', 
                    patience=5, 
                    restore_best_weights=True, 
                    verbose=1 
                ), 
                keras.callbacks.ReduceLROnPlateau( 
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=3, 
                    verbose=1 
                ) 
            ] 
             
            # Train model 
            history = self.model.fit( 
                X_train, y_train, 
                validation_data=(X_val, y_val), 
                epochs=30, 
                batch_size=min(32, len(X_train)), 
                verbose=1, 
                callbacks=callbacks 
            ) 
             
            # Display results 
            val_accuracy = max(history.history['val_accuracy']) if history.history['val_accuracy'] else 0 
            best_epoch = np.argmax(history.history['val_accuracy']) + 1 
             
            # Store training history for visualization 
            self.training_history = history 
             
            self.info_label.config(text=f"Training complete! Validation accuracy: {val_accuracy:.2%}",  
                                  foreground="green") 
             
            self.results_text.insert(tk.END, f"\n{'='*50}\n") 
            self.results_text.insert(tk.END, f"TRAINING COMPLETE!\n") 
            self.results_text.insert(tk.END, f"{'='*50}\n") 
            self.results_text.insert(tk.END, f"Classes: {list(self.class_names)}\n") 
            self.results_text.insert(tk.END, f"Total training samples: {len(X_train)}\n") 
            self.results_text.insert(tk.END, f"Validation samples: {len(X_val)}\n") 
            self.results_text.insert(tk.END, f"Best Validation Accuracy: {val_accuracy:.2%} (Epoch {best_epoch})\n\n") 
             
            # Show per-class distribution 
            self.results_text.insert(tk.END, "Class Distribution in Training Set:\n") 
            for i, class_name in enumerate(self.class_names): 
                count = sum(labels_encoded == i) 
                percentage = (count / len(labels_encoded)) * 100 
                self.results_text.insert(tk.END, f"  - {class_name}: {count} images ({percentage:.1f}%)\n") 
             
            self.results_text.insert(tk.END, "\n") 
            self.results_text.see(tk.END) 
             
            # Update visualizations 
            self.update_training_plot(history) 
            self.update_class_distribution(labels) 
             
            # Show model evaluation metrics 
            self.show_evaluation_metrics(history, X_val, y_val) 
             
            messagebox.showinfo("Success", f"Model trained successfully!\n\nValidation Accuracy: {val_accuracy:.2%}") 
             
        except Exception as e: 
            error_msg = f"Training failed: {str(e)}\n\n" 
            error_msg += "Please check:\n" 
            error_msg += "1. CSV file format matches your data\n" 
            error_msg += "2. Image folder path is correct\n" 
            error_msg += "3. Image files exist and are not corrupted" 
            messagebox.showerror("Error", error_msg) 
            self.info_label.config(text="Training failed", foreground="red") 
            import traceback 
            traceback.print_exc() 
     
    def save_model(self): 
        """Save trained model""" 
        if self.model is None: 
            messagebox.showwarning("Warning", "No model to save. Train or load a model first.") 
            return 
         
        file_path = filedialog.asksaveasfilename( 
            defaultextension=".h5", 
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")] 
        ) 
         
        if file_path: 
            self.model.save(file_path) 
            # Save class names 
            if self.class_names: 
                np.save(file_path.replace('.h5', '_classes.npy'), self.class_names) 
            messagebox.showinfo("Success", f"Model saved to {file_path}") 
     
    def load_model(self): 
        """Load pre-trained model""" 
        file_path = filedialog.askopenfilename( 
            title="Select model file", 
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")] 
        ) 
         
        if file_path: 
            try: 
                self.model = keras.models.load_model(file_path) 
                classes_path = file_path.replace('.h5', '_classes.npy') 
                if os.path.exists(classes_path): 
                    self.class_names = np.load(classes_path, allow_pickle=True) 
                self.info_label.config(text=f"Model loaded from {os.path.basename(file_path)}",  
                                      foreground="green") 
                messagebox.showinfo("Success", "Model loaded successfully!") 
            except Exception as e: 
                messagebox.showerror("Error", f"Failed to load model: {str(e)}") 
     
    def classify_single_image(self): 
        """Classify a single image""" 
        if self.model is None: 
            messagebox.showwarning("Warning", "No model loaded. Please load or train a model first.") 
            return 
         
        file_path = filedialog.askopenfilename( 
            title="Select image to classify", 
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")] 
        ) 
         
        if file_path: 
            try: 
                # Load and preprocess image 
                img = cv2.imread(file_path) 
                if img is None: 
                    messagebox.showerror("Error", "Could not load image. File might be corrupted.") 
                    return 
                     
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                display_img = img.copy() 
                img_resized = cv2.resize(img, self.image_size) 
                img_normalized = img_resized / 255.0 
                img_batch = np.expand_dims(img_normalized, axis=0) 
                 
                # Display the original image at a larger size 
                self.display_image(display_img) 
                 
                # Predict 
                predictions = self.model.predict(img_batch, verbose=0) 
                predicted_class_idx = np.argmax(predictions[0]) 
                confidence = predictions[0][predicted_class_idx] 
                predicted_label = self.class_names[predicted_class_idx] 
                 
                # Display results 
                result_text = f"\n{'='*50}\n" 
                result_text += f"CLASSIFICATION RESULT\n" 
                result_text += f"{'='*50}\n" 
                result_text += f"Image: {os.path.basename(file_path)}\n" 
                result_text += f"Predicted Class: {predicted_label}\n" 
                result_text += f"Confidence: {confidence:.2%}\n\n" 
                 
                # Show top 3 predictions 
                result_text += "Top 3 Predictions:\n" 
                top_indices = np.argsort(predictions[0])[-3:][::-1] 
                for idx in top_indices: 
                    result_text += f"  • {self.class_names[idx]}: {predictions[0][idx]:.2%}\n" 
                result_text += f"{'='*50}\n\n" 
                 
                self.results_text.insert(tk.END, result_text) 
                self.results_text.see(tk.END) 
                 
                self.info_label.config(text=f"Classified as: {predicted_label} ({confidence:.2%})",  
                                      foreground="blue") 
                 
            except Exception as e: 
                messagebox.showerror("Error", f"Classification failed: {str(e)}") 
     
    def batch_classify(self): 
        """Classify all images in a folder""" 
        if self.model is None: 
            messagebox.showwarning("Warning", "No model loaded. Please load or train a model first.") 
            return 
         
        folder_path = filedialog.askdirectory(title="Select folder with images to classify") 
        if not folder_path: 
            return 
             
        results = [] 
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif') 
         
        # Get list of image files 
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)] 
         
        if not image_files: 
            messagebox.showwarning("Warning", "No image files found in selected folder!") 
            return 
         
        self.info_label.config(text=f"Classifying {len(image_files)} images...", foreground="orange") 
        self.root.update() 
         
        for filename in image_files: 
            img_path = os.path.join(folder_path, filename) 
            try: 
                img = cv2.imread(img_path) 
                if img is not None: 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                    img_resized = cv2.resize(img, self.image_size) 
                    img_normalized = img_resized / 255.0 
                    img_batch = np.expand_dims(img_normalized, axis=0) 
                     
                    predictions = self.model.predict(img_batch, verbose=0) 
                    predicted_class_idx = np.argmax(predictions[0]) 
                    confidence = predictions[0][predicted_class_idx] 
                     
                    results.append({ 
                        'image': filename, 
                        'predicted_class': self.class_names[predicted_class_idx], 
                        'confidence': confidence 
                    }) 
                else: 
                    results.append({ 
                        'image': filename, 
                        'predicted_class': 'Error', 
                        'confidence': 'Failed to load image' 
                    }) 
            except Exception as e: 
                results.append({ 
                    'image': filename, 
                    'predicted_class': 'Error', 
                    'confidence': f'Failed: {str(e)[:50]}' 
                }) 
         
        # Display results 
        self.results_text.insert(tk.END, f"\n{'='*50}\n") 
        self.results_text.insert(tk.END, f"BATCH CLASSIFICATION RESULTS\n") 
        self.results_text.insert(tk.END, f"{'='*50}\n") 
        self.results_text.insert(tk.END, f"Total images processed: {len(results)}\n\n") 
         
        # Group by predicted class 
        class_counts = {} 
        for r in results: 
            if r['predicted_class'] not in ['Error']: 
                class_counts[r['predicted_class']] = class_counts.get(r['predicted_class'], 0) + 1 
         
        if class_counts: 
            self.results_text.insert(tk.END, "Summary by class:\n") 
            for class_name, count in sorted(class_counts.items()): 
                self.results_text.insert(tk.END, f"  • {class_name}: {count} images\n") 
            self.results_text.insert(tk.END, "\n") 
         
        self.results_text.insert(tk.END, "Detailed results:\n") 
        self.results_text.insert(tk.END, "-" * 50 + "\n") 
        for r in results: 
            if isinstance(r['confidence'], float): 
                self.results_text.insert(tk.END, f"{r['image']}: {r['predicted_class']} ({r['confidence']:.2%})\n") 
            else: 
                self.results_text.insert(tk.END, f"{r['image']}: {r['predicted_class']} - {r['confidence']}\n") 
        self.results_text.insert(tk.END, f"\n{'='*50}\n\n") 
         
        self.results_text.see(tk.END) 
        self.info_label.config(text=f"Classified {len(results)} images successfully", foreground="green") 
        messagebox.showinfo("Complete", f"Classified {len(results)} images successfully!") 
     
    def display_image(self, image): 
        """Display image on canvas""" 
        # Clear canvas 
        self.canvas.delete("all") 
         
        # Resize for display (max 600px) 
        h, w = image.shape[:2] 
        max_size = 600 
        if h > max_size or w > max_size: 
            scale = max_size / max(h, w) 
            new_w = int(w * scale) 
            new_h = int(h * scale) 
            image = cv2.resize(image, (new_w, new_h)) 
         
        # Convert to PhotoImage 
        image = Image.fromarray(image) 
        photo = ImageTk.PhotoImage(image) 
         
        # Store reference to prevent garbage collection 
        self.current_image = photo 
         
        # Display on canvas 
        self.canvas.create_image(5, 5, anchor=tk.NW, image=photo) 
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL)) 
 
 
def main(): 
    root = tk.Tk() 
    app = ImageClassificationSystem(root) 
    root.mainloop() 
 
if __name__ == "__main__": 
    main()
