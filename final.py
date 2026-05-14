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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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
        self.dt_model = None  # Decision Tree model
        self.lr_model = None  # Logistic Regression model
        self.label_encoder = None 
        self.class_names = [] 
        self.image_size = (96, 96)  # Reduced from 160x160 for faster processing
        self.training_history = None 
        self.class_distribution = {} 
        self.confidence_threshold = 0.85  # Very strict: reject unless 85%+ confident
        self.dataset_image_ids = set()    # Store IDs from the CSV
        self.model_comparison_results = {}  # Store comparison metrics
        self.X_val_data = None  # Store validation data for metrics
        self.y_val_data = None  # Store validation labels for metrics
         
        # Vibrant Sky Blue Palette
        self.colors = {
            'p': '#0ea5e9',      # Primary Sky Blue
            'ph': '#0284c7',     # Hover
            'bg': '#f1f5f9',     # Soft Blue-Gray BG
            'box': '#ffffff',    # White Card
            'txt': '#1e293b',    # Slate Text
            'brd': '#e0f2fe'     # Border
        }
        self.root.configure(bg=self.colors['bg'])
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('Box.TFrame', background=self.colors['box'])
        style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['txt'], font=('Arial', 10))
        style.configure('Box.TLabel', background=self.colors['box'], foreground=self.colors['txt'], font=('Arial', 10))
        style.configure('Header.TLabel', font=('Arial', 24, 'bold'), background=self.colors['bg'], foreground='#0c4a6e')
        
        # Designed Button Style
        style.configure('TButton', font=('Arial', 9, 'bold'), padding=(10, 5), borderwidth=0)
        style.map('TButton', background=[('active', self.colors['ph']), ('!disabled', self.colors['p'])], foreground=[('!disabled', 'white')])
        
        style.configure('TLabelframe', background=self.colors['bg'], bordercolor=self.colors['brd'], borderwidth=1)
        style.configure('TLabelframe.Label', background=self.colors['bg'], foreground=self.colors['p'], font=('Arial', 10, 'bold'))
        
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
         
        # Designed Title Section
        header_frame = ttk.Frame(main_container)
        header_frame.grid(row=0, column=0, pady=(10, 5), sticky="ew")
        header_frame.columnconfigure(0, weight=1)

        # Title with accent color
        title_container = ttk.Frame(header_frame)
        title_container.grid(row=0, column=0)
        
        ttk.Label(title_container, text="Image ", font=('Arial Black', 24), 
                  foreground=self.colors['txt']).grid(row=0, column=0)
        ttk.Label(title_container, text="Classification", font=('Arial Black', 24), 
                  foreground=self.colors['p']).grid(row=0, column=1)
        ttk.Label(title_container, text=" System", font=('Arial Black', 24), 
                  foreground=self.colors['txt']).grid(row=0, column=2)
        
        # Decorative separator
        line = tk.Frame(header_frame, height=2, width=100, bg=self.colors['p'])
        line.grid(row=1, column=0, pady=10)

        subtitle = ttk.Label(header_frame, text="Deep Learning Neural Network (TensorFlow/Keras)", 
                           font=('Arial', 10), foreground='#6b7280') 
        subtitle.grid(row=2, column=0, pady=(0, 20)) 
         
        # Control Panel Frame 
        control_frame = ttk.LabelFrame(main_container, text=" Controls", padding="15") 
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 20)) 
        control_frame.columnconfigure(0, weight=1) 
         
        # Inner content with subtle background
        content_box = ttk.Frame(control_frame, style='Box.TFrame')
        content_box.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=2, pady=2)
        content_box.columnconfigure(1, weight=1)

        # Train Section 
        train_frame = ttk.Frame(content_box, style='Box.TFrame') 
        train_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=10, padx=10) 
         
        ttk.Label(train_frame, text="Neural Network:", style='Box.TLabel').grid(row=0, column=0, padx=5) 
        ttk.Button(train_frame, text="Initialize & Train System",  
                  command=self.load_and_train).grid(row=0, column=1, padx=5) 
         
        # Classification Section 
        classify_frame = ttk.Frame(content_box, style='Box.TFrame') 
        classify_frame.grid(row=0, column=2, sticky=(tk.W, tk.E), pady=10, padx=10) 
         
        ttk.Label(classify_frame, text="Single Image:", style='Box.TLabel').grid(row=0, column=0, padx=5) 
        ttk.Button(classify_frame, text="Run Classification",  
                  command=self.classify_single_image).grid(row=0, column=1, padx=5) 
        
        # Model Comparison Section
        comparison_frame = ttk.Frame(content_box, style='Box.TFrame')
        comparison_frame.grid(row=0, column=3, sticky=(tk.W, tk.E), pady=10, padx=10)
        
        ttk.Label(comparison_frame, text="Model Comparison:", style='Box.TLabel').grid(row=0, column=0, padx=5)
        ttk.Button(comparison_frame, text="View Comparison",
                  command=self.show_model_comparison).grid(row=0, column=1, padx=5)
        
        # Metrics Section
        metrics_frame = ttk.Frame(content_box, style='Box.TFrame')
        metrics_frame.grid(row=0, column=4, sticky=(tk.W, tk.E), pady=10, padx=10)
        
        ttk.Label(metrics_frame, text="Evaluation:", style='Box.TLabel').grid(row=0, column=0, padx=5)
        ttk.Button(metrics_frame, text="View Metrics",
                  command=self.show_stored_metrics).grid(row=0, column=1, padx=5)
        
        # Threshold Slider with styled container
        threshold_frame = ttk.Frame(content_box, style='Box.TFrame') 
        threshold_frame.grid(row=1, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(0, 10), padx=15) 
        
        ttk.Label(threshold_frame, text="Strictness Threshold:", style='Box.TLabel').grid(row=0, column=0, padx=5) 
        self.threshold_var = tk.DoubleVar(value=self.confidence_threshold)
        self.threshold_scale = ttk.Scale(threshold_frame, from_=0, to=1, orient='horizontal',
                                         variable=self.threshold_var, command=self.update_threshold)
        self.threshold_scale.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        self.threshold_label = ttk.Label(threshold_frame, text=f"{self.confidence_threshold:.0%}", 
                                      font=('Arial', 10, 'bold'), foreground=self.colors['p'], style='Box.TLabel')
        self.threshold_label.grid(row=0, column=2, padx=5)
         
        # Info Label centered at the bottom of the box
        self.info_label = ttk.Label(content_box, text="READY", foreground=self.colors['p'], 
                                   font=('Arial', 8, 'bold'), style='Box.TLabel') 
        self.info_label.grid(row=2, column=0, columnspan=5, pady=(5, 10)) 
         
        # Middle container for side-by-side display 
        mid_container = ttk.Frame(main_container) 
        mid_container.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10)) 
        mid_container.columnconfigure(0, weight=1) 
        mid_container.columnconfigure(1, weight=1) 
        mid_container.columnconfigure(2, weight=1) 
        mid_container.rowconfigure(0, weight=1) 
 
        # Display Area 
        display_frame = ttk.LabelFrame(mid_container, text="Image Display", padding="10") 
        display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5)) 
        display_frame.columnconfigure(0, weight=1) 
        display_frame.rowconfigure(0, weight=1) 
         
        # Canvas for image display 
        self.canvas = tk.Canvas(display_frame, bg='#f0f0f0', width=300, height=400, highlightthickness=0) 
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
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, width=30, height=20) 
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) 
         
        scrollbar_text = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview) 
        scrollbar_text.grid(row=0, column=1, sticky=(tk.N, tk.S)) 
        self.results_text.configure(yscrollcommand=scrollbar_text.set) 
         
        # Prediction Pie Chart Area
        self.create_prediction_pie_plot(mid_container)
         
        # Visualization Area 
        self.viz_frame = ttk.LabelFrame(main_container, text="Visualizations", padding="10") 
        self.viz_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0)) 
        self.viz_frame.columnconfigure(0, weight=1) 
        self.viz_frame.columnconfigure(1, weight=1) 
        self.viz_frame.rowconfigure(0, weight=1) 
         
        # Training History Plot 
        self.create_training_plot() 
         
        # Class Distribution Plot 
        self.create_class_distribution_plot() 
    
    def update_threshold(self, value):
        """Update confidence threshold when slider is moved"""
        self.confidence_threshold = float(value)
        self.threshold_label.config(text=f"{self.confidence_threshold:.0%}")
     
    def create_training_plot(self): 
        """Create training history visualization frame""" 
        plot_frame = ttk.LabelFrame(self.viz_frame, text="Training History", padding="5") 
        plot_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5)) 
        plot_frame.columnconfigure(0, weight=1) 
        plot_frame.rowconfigure(0, weight=1) 
         
        self.fig_training = Figure(figsize=(5, 3.5), dpi=100, facecolor='#f5f5f5') 
        self.ax_training = self.fig_training.add_subplot(111) 
        self.ax_training.set_title('Accuracy & Loss Over Epochs') 
        self.ax_training.set_xlabel('Epoch') 
        self.ax_training.set_ylabel('Value') 
        self.ax_training.grid(True, alpha=0.3) 
         
        self.canvas_training = FigureCanvasTkAgg(self.fig_training, plot_frame) 
        widget_training = self.canvas_training.get_tk_widget() 
        widget_training.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) 
        self.fig_training.tight_layout() 
     
    def create_class_distribution_plot(self): 
        """Create class distribution visualization frame""" 
        dist_frame = ttk.LabelFrame(self.viz_frame, text="Class Distribution", padding="5") 
        dist_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0)) 
        dist_frame.columnconfigure(0, weight=1) 
        dist_frame.rowconfigure(0, weight=1) 
         
        self.fig_dist = Figure(figsize=(5, 3.5), dpi=100, facecolor='#f5f5f5') 
        self.ax_dist = self.fig_dist.add_subplot(111) 
        self.ax_dist.set_title('Images per Class') 
        self.ax_dist.set_xlabel('Class') 
        self.ax_dist.set_ylabel('Count') 
        self.ax_dist.grid(True, alpha=0.3) 
         
        self.canvas_dist = FigureCanvasTkAgg(self.fig_dist, dist_frame) 
        widget_dist = self.canvas_dist.get_tk_widget() 
        widget_dist.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) 
        self.fig_dist.tight_layout() 

    def create_prediction_pie_plot(self, parent):
        """Create prediction probability pie chart frame"""
        pie_frame = ttk.LabelFrame(parent, text="Prediction Confidence", padding="5")
        pie_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        pie_frame.columnconfigure(0, weight=1)
        pie_frame.rowconfigure(0, weight=1)
        
        self.fig_pie = Figure(figsize=(3, 3), dpi=100, facecolor='#f5f5f5')
        self.ax_pie = self.fig_pie.add_subplot(111)
        self.ax_pie.set_title('Confidence Distribution')
        
        self.canvas_pie = FigureCanvasTkAgg(self.fig_pie, pie_frame)
        self.widget_pie = self.canvas_pie.get_tk_widget()
        self.widget_pie.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.fig_pie.tight_layout()

    def update_prediction_pie(self, predictions):
        """Update prediction pie chart with current results"""
        if self.class_names is None or len(self.class_names) == 0:
            return
            
        self.ax_pie.clear()
        
        # Filter classes with significant probability (> 1%)
        labels = []
        sizes = []
        for i, prob in enumerate(predictions):
            if prob > 0.01:
                labels.append(self.class_names[i])
                sizes.append(prob)
        
        # If everything is low, just show the top one
        if not labels:
            idx = np.argmax(predictions)
            labels = [self.class_names[idx]]
            sizes = [predictions[idx]]
            
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        self.ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
                        textprops={'fontsize': 8})
        self.ax_pie.set_title('Prediction Confidence')
        
        self.fig_pie.tight_layout()
        self.canvas_pie.draw()
     
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
         
        self.fig_training.tight_layout() 
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
        self.fig_dist.tight_layout() 
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
        prec = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0) 
        rec = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0) 
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0) 
         
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
                                        target_names=self.class_names, zero_division=0) 
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
             
            self.results_text.insert(tk.END, f"Using '{label_col}' as label column\n\n") 
             
            # Store all image IDs from CSV for later verification
            self.dataset_image_ids = set(df[image_col].astype(str).str.strip().tolist())
             
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
                        self.dataset_image_ids.add(img_id)
                        self.dataset_image_ids.add(os.path.basename(img_path))
                        self.dataset_image_ids.add(img_id)
                        self.dataset_image_ids.add(os.path.basename(img_path))
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
             
            # Build model with Augmentation
            self.model = keras.Sequential([ 
                layers.RandomFlip("horizontal", input_shape=(*self.image_size, 3)),
                layers.RandomRotation(0.1),
                layers.Conv2D(32, (3, 3), activation='relu'), 
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
             
            # Callbacks including UI refresh
            callbacks = [ 
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                keras.callbacks.LambdaCallback(on_batch_end=lambda *a, **k: self.root.update())
            ] 
             
            # Train model 
            history = self.model.fit( 
                X_train, y_train, 
                validation_data=(X_val, y_val), 
                epochs=20,           # Reduced epochs
                batch_size=min(64, len(X_train)), # Increased batch size
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
             
            # Train Decision Tree and Logistic Regression models for comparison
            self.results_text.insert(tk.END, "\n" + "="*50 + "\n")
            self.results_text.insert(tk.END, "TRAINING ALTERNATIVE MODELS FOR COMPARISON...\n")
            self.results_text.insert(tk.END, "="*50 + "\n\n")
            self.root.update()
            
            # Flatten images for scikit-learn models (they don't handle 3D arrays well)
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            # Get true labels from one-hot encoding
            y_train_labels = np.argmax(y_train, axis=1)
            y_val_labels = np.argmax(y_val, axis=1)
            
            # Train Decision Tree
            self.results_text.insert(tk.END, "Training Decision Tree...\n")
            self.root.update()
            try:
                self.dt_model = DecisionTreeClassifier(max_depth=20, random_state=42, n_jobs=-1)
                self.dt_model.fit(X_train_flat, y_train_labels)
                dt_pred = self.dt_model.predict(X_val_flat)
                dt_accuracy = accuracy_score(y_val_labels, dt_pred)
                dt_precision = precision_score(y_val_labels, dt_pred, average='weighted', zero_division=0)
                dt_recall = recall_score(y_val_labels, dt_pred, average='weighted', zero_division=0)
                dt_f1 = f1_score(y_val_labels, dt_pred, average='weighted', zero_division=0)
                
                self.results_text.insert(tk.END, f"✓ Decision Tree Accuracy: {dt_accuracy:.2%}\n\n")
                self.root.update()
            except Exception as e:
                self.results_text.insert(tk.END, f"✗ Decision Tree training failed: {str(e)}\n\n")
                dt_accuracy = dt_precision = dt_recall = dt_f1 = 0
            
            # Train Logistic Regression
            self.results_text.insert(tk.END, "Training Logistic Regression...\n")
            self.root.update()
            try:
                self.lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
                self.lr_model.fit(X_train_flat, y_train_labels)
                lr_pred = self.lr_model.predict(X_val_flat)
                lr_accuracy = accuracy_score(y_val_labels, lr_pred)
                lr_precision = precision_score(y_val_labels, lr_pred, average='weighted', zero_division=0)
                lr_recall = recall_score(y_val_labels, lr_pred, average='weighted', zero_division=0)
                lr_f1 = f1_score(y_val_labels, lr_pred, average='weighted', zero_division=0)
                
                self.results_text.insert(tk.END, f"✓ Logistic Regression Accuracy: {lr_accuracy:.2%}\n\n")
                self.root.update()
            except Exception as e:
                self.results_text.insert(tk.END, f"✗ Logistic Regression training failed: {str(e)}\n\n")
                lr_accuracy = lr_precision = lr_recall = lr_f1 = 0
            
            # Get Neural Network metrics
            nn_pred = self.model.predict(X_val, verbose=0)
            nn_pred_classes = np.argmax(nn_pred, axis=1)
            nn_accuracy = accuracy_score(y_val_labels, nn_pred_classes)
            nn_precision = precision_score(y_val_labels, nn_pred_classes, average='weighted', zero_division=0)
            nn_recall = recall_score(y_val_labels, nn_pred_classes, average='weighted', zero_division=0)
            nn_f1 = f1_score(y_val_labels, nn_pred_classes, average='weighted', zero_division=0)
            
            # Store comparison results
            self.model_comparison_results = {
                'Neural Network': {'accuracy': nn_accuracy, 'precision': nn_precision, 'recall': nn_recall, 'f1': nn_f1},
                'Decision Tree': {'accuracy': dt_accuracy, 'precision': dt_precision, 'recall': dt_recall, 'f1': dt_f1},
                'Logistic Regression': {'accuracy': lr_accuracy, 'precision': lr_precision, 'recall': lr_recall, 'f1': lr_f1}
            }
            
            self.results_text.insert(tk.END, "\n" + "="*50 + "\n")
            self.results_text.insert(tk.END, "MODEL COMPARISON SUMMARY\n")
            self.results_text.insert(tk.END, "="*50 + "\n\n")
            for model_name, metrics in self.model_comparison_results.items():
                self.results_text.insert(tk.END, f"{model_name}:\n")
                self.results_text.insert(tk.END, f"  Accuracy:  {metrics['accuracy']:.2%}\n")
                self.results_text.insert(tk.END, f"  Precision: {metrics['precision']:.2%}\n")
                self.results_text.insert(tk.END, f"  Recall:    {metrics['recall']:.2%}\n")
                self.results_text.insert(tk.END, f"  F1-Score:  {metrics['f1']:.2%}\n\n")
            
            self.results_text.see(tk.END) 
            
            # Store validation data for later use
            self.X_val_data = X_val
            self.y_val_data = y_val
             
            messagebox.showinfo("Success", f"Model trained successfully!\n\nValidation Accuracy: {val_accuracy:.2%}\n\nClick 'View Metrics' or 'View Comparison' buttons to see detailed results.") 
             
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
                # Check if image belongs to dataset 
                image_id = os.path.splitext(os.path.basename(file_path))[0]
                full_filename = os.path.basename(file_path)
                
                if image_id not in self.dataset_image_ids and full_filename not in self.dataset_image_ids:
                    messagebox.showerror("Error", "it's not belong in the dataset or not including on training")
                    return

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
                
                # Update pie chart
                self.update_prediction_pie(predictions[0])
                 
                # Predict and show results
                p = self.model.predict(img_batch, verbose=0)[0]
                idx = np.argmax(p); conf = p[idx]; lbl = self.class_names[idx]
                self.update_prediction_pie(p)
                
                status = "✓ RECOGNIZED" if conf >= self.confidence_threshold else "⚠ REJECTED: Low Confidence"
                res = f"\n{'='*40}\n{status}\n{'='*40}\nImg: {os.path.basename(file_path)}\n\n"
                res += f"PRIMARY RESULT: {lbl}\n"
                res += f"CONFIDENCE: {conf:.2%}\n\n"
                
                res += "TOP 5 PREDICTIONS:\n"
                top_idx = np.argsort(p)[-5:][::-1]
                for i in top_idx:
                    res += f"  • {self.class_names[i]:<15} {p[i]:.2%}\n"
                
                self.results_text.insert(tk.END, res + "="*40 + "\n")
                self.results_text.see(tk.END)
                self.info_label.config(text=f"{status} ({conf:.1%})" if conf >= self.confidence_threshold else f"⚠ LOW CONFIDENCE: {lbl} ({conf:.1%})")
                 
            except Exception as e: 
                messagebox.showerror("Error", f"Classification failed: {str(e)}") 
     
     
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
    
    def show_stored_metrics(self):
        """Show evaluation metrics from stored validation data"""
        if self.model is None or self.X_val_data is None or self.y_val_data is None:
            messagebox.showwarning("No Data", "Train a model first to see metrics")
            return
        
        self.show_evaluation_metrics(self.training_history, self.X_val_data, self.y_val_data)
 
    def show_model_comparison(self):
        """Display model comparison visualization"""
        if not self.model_comparison_results:
            messagebox.showwarning("No Data", "Train a model first to see comparison")
            return
        
        comparison_win = tk.Toplevel(self.root)
        comparison_win.title("Model Performance Comparison")
        comparison_win.geometry("1400x600")
        
        # Main container with three columns
        main_frame = ttk.Frame(comparison_win)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Column 1: Bar Chart Comparison
        col1_frame = ttk.LabelFrame(main_frame, text="Metrics Comparison", padding="5")
        col1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        fig_bar = Figure(figsize=(5, 5), dpi=100, facecolor='white')
        ax_bar = fig_bar.add_subplot(111)
        
        models = list(self.model_comparison_results.keys())
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics_names))
        width = 0.25
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red
        
        for i, model in enumerate(models):
            values = [
                self.model_comparison_results[model]['accuracy'],
                self.model_comparison_results[model]['precision'],
                self.model_comparison_results[model]['recall'],
                self.model_comparison_results[model]['f1']
            ]
            ax_bar.bar(x + i*width, values, width, label=model, color=colors[i], alpha=0.8)
        
        ax_bar.set_ylabel('Score', fontsize=10)
        ax_bar.set_title('All Metrics', fontsize=11, fontweight='bold')
        ax_bar.set_xticks(x + width)
        ax_bar.set_xticklabels(metrics_names, fontsize=9)
        ax_bar.legend(loc='lower right', fontsize=9)
        ax_bar.grid(True, alpha=0.3, axis='y')
        ax_bar.set_ylim([0, 1.05])
        
        # Add value labels on bars
        for i, model in enumerate(models):
            values = [
                self.model_comparison_results[model]['accuracy'],
                self.model_comparison_results[model]['precision'],
                self.model_comparison_results[model]['recall'],
                self.model_comparison_results[model]['f1']
            ]
            for j, v in enumerate(values):
                ax_bar.text(j + i*width, v + 0.02, f'{v:.0%}', ha='center', va='bottom', fontsize=7)
        
        fig_bar.tight_layout()
        canvas_bar = FigureCanvasTkAgg(fig_bar, col1_frame)
        canvas_bar.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Column 2: Accuracy Comparison
        col2_frame = ttk.LabelFrame(main_frame, text="Accuracy Comparison", padding="5")
        col2_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        fig_radar = Figure(figsize=(5, 5), dpi=100, facecolor='white')
        ax_radar = fig_radar.add_subplot(111)
        
        models_list = list(self.model_comparison_results.keys())
        accuracies = [self.model_comparison_results[m]['accuracy'] for m in models_list]
        
        bars = ax_radar.barh(models_list, accuracies, color=colors, alpha=0.7)
        ax_radar.set_xlabel('Accuracy', fontsize=10)
        ax_radar.set_title('Accuracy Comparison', fontsize=11, fontweight='bold')
        ax_radar.set_xlim([0, 1.05])
        ax_radar.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax_radar.text(acc + 0.02, i, f'{acc:.2%}', va='center', fontsize=10, fontweight='bold')
        
        fig_radar.tight_layout()
        canvas_radar = FigureCanvasTkAgg(fig_radar, col2_frame)
        canvas_radar.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Column 3: Summary Table
        col3_frame = ttk.LabelFrame(main_frame, text="Summary Table", padding="5")
        col3_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        table_container = ttk.Frame(col3_frame)
        table_container.pack(fill=tk.BOTH, expand=True)
        
        table_text = tk.Text(table_container, wrap=tk.WORD, font=('Courier', 8))
        table_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        table_scroll = ttk.Scrollbar(table_container, orient="vertical", command=table_text.yview)
        table_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        table_text.configure(yscrollcommand=table_scroll.set)
        
        # Create formatted table
        table_content = "\n─────────────────────────────\n"
        table_content += "  MODEL COMPARISON SUMMARY   \n"
        table_content += "─────────────────────────────\n"
        
        # Find best model for each metric
        best_accuracy = max(self.model_comparison_results.items(), key=lambda x: x[1]['accuracy'])
        
        for model, metrics in self.model_comparison_results.items():
            is_best = "" if model == best_accuracy[0] else " "
            table_content += f" {is_best} {model:<23} \n"
            table_content += f"  Acc:  {metrics['accuracy']:.2%}              \n"
            table_content += f"  Prec: {metrics['precision']:.2%}              \n"
            table_content += f"  Rec:  {metrics['recall']:.2%}              \n"
            table_content += f"  F1:   {metrics['f1']:.2%}              \n"
            table_content += "─────────────────────────────\n"
        
        table_content += f" BEST: {best_accuracy[0]:<18} \n"
        table_content += f" ACC:  {best_accuracy[1]['accuracy']:.2%}            \n"
        table_content += "─────────────────────────────\n"
        
        table_text.insert(tk.END, table_content)
        table_text.config(state=tk.DISABLED)
 
 
def main(): 
    root = tk.Tk() 
    app = ImageClassificationSystem(root) 
    root.mainloop() 
 
if __name__ == "__main__": 
    main()
