"""
GUI Application for Drone Object Detection & Tracking
Modern interface with tabbed layout
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import json
from collections import defaultdict, deque
import numpy as np
from queue import Queue, Empty

from video_source import open_video_source
from detector import ObjectDetector


class DroneDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Computer Vision")
        self.root.geometry("1600x900")
        self.root.configure(bg='#2b2b2b')
        
        # State variables
        self.cap = None
        self.detector = None
        self.is_running = False
        self.is_paused = False
        self.is_recording = False
        self.video_writer = None
        self.follow_target_id = None
        self.trajectories = defaultdict(lambda: deque(maxlen=50))
        self.object_stats = {}
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # ROI and counting line
        self.roi_points = []
        self.counting_line = None
        self.object_counts = {"in": 0, "out": 0}
        self.crossed_ids = set()
        
        # Drawing state
        self.drawing_mode = None
        self.temp_points = []
        
        # Visualization settings
        self.show_trajectories = tk.BooleanVar(value=False)
        self.show_heatmap = tk.BooleanVar(value=False)
        self.heatmap_data = None
        
        # Image enhancement
        self.enhance_mode = tk.StringVar(value="none")
        
        # Class filtering
        self.all_class_names = []
        
        # Multi-threading for display
        self.frame_queue = Queue(maxsize=30)  # Buffer for 30 frames
        self.display_thread = None
        self.display_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Build modern tabbed GUI layout"""
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Top control bar
        top_bar = tk.Frame(self.root, bg='#1e1e1e', height=70)
        top_bar.pack(side=tk.TOP, fill=tk.X)
        top_bar.pack_propagate(False)
        
        # Main START/STOP buttons
        btn_frame = tk.Frame(top_bar, bg='#1e1e1e')
        btn_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="▶ START", font=("Arial", 14, "bold"),
                                    bg='#28a745', fg='white', width=12, height=2,
                                    command=self.start_detection, relief=tk.FLAT,
                                    cursor='hand2')
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹ STOP", font=("Arial", 14, "bold"),
                                   bg='#dc3545', fg='white', width=12, height=2,
                                   command=self.stop_detection, relief=tk.FLAT,
                                   state=tk.DISABLED, cursor='hand2')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = tk.Button(btn_frame, text="⏸ PAUSE", font=("Arial", 12),
                                    bg='#ffc107', fg='black', width=10, height=2,
                                    command=self.toggle_pause, relief=tk.FLAT,
                                    state=tk.DISABLED, cursor='hand2')
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        # Status display
        status_frame = tk.Frame(top_bar, bg='#1e1e1e')
        status_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)
        
        tk.Label(status_frame, text="Status:", bg='#1e1e1e', fg='white', font=("Arial", 10)).pack(anchor=tk.W)
        self.status_label = tk.Label(status_frame, text="Ready", bg='#1e1e1e', fg='#28a745', font=("Arial", 12, "bold"))
        self.status_label.pack(anchor=tk.W)
        
        tk.Label(status_frame, text="FPS:", bg='#1e1e1e', fg='white', font=("Arial", 10)).pack(anchor=tk.W, pady=(5, 0))
        self.fps_label = tk.Label(status_frame, text="0.0", bg='#1e1e1e', fg='yellow', font=("Arial", 14, "bold"))
        self.fps_label.pack(anchor=tk.W)
        
        # Quick actions
        action_frame = tk.Frame(top_bar, bg='#1e1e1e')
        action_frame.pack(side=tk.RIGHT, padx=20, pady=10)
        
        tk.Button(action_frame, text="📷", font=("Arial", 16), width=3, height=1,
                  command=self.take_screenshot, bg='#6c757d', fg='white', relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=2)
        
        self.record_btn = tk.Button(action_frame, text="⏺", font=("Arial", 16), width=3, height=1,
                                     command=self.toggle_recording, bg='#6c757d', fg='white', relief=tk.FLAT,
                                     state=tk.DISABLED, cursor='hand2')
        self.record_btn.pack(side=tk.LEFT, padx=2)
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#2b2b2b')
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: Tabbed control panel
        self.notebook = ttk.Notebook(content_frame, width=350)
        self.notebook.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Setup tabs
        self.setup_config_tab()
        self.setup_filters_tab()
        self.setup_advanced_tab()
        
        # Right: Video display
        video_frame = tk.Frame(content_frame, bg='black')
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        
    def setup_config_tab(self):
        """Configuration tab"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="⚙ Configuration")
        
        row = 0
        
        # Source
        ttk.Label(tab, text="Video Source", font=("Arial", 11, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        # Preset selection
        preset_frame = ttk.Frame(tab)
        preset_frame.grid(row=row, column=0, sticky=tk.EW, pady=(0, 5))
        row += 1
        
        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT, padx=(0, 5))
        self.source_preset = tk.StringVar(value="Webcam 0")
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.source_preset, 
                                    values=["Webcam 0", "Webcam 1", "Webcam 2", "Video File...", "RTSP Stream", "Custom"],
                                    state="readonly", width=20)
        preset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        preset_combo.bind("<<ComboboxSelected>>", self.on_preset_change)
        
        # Source input
        source_frame = ttk.Frame(tab)
        source_frame.grid(row=row, column=0, sticky=tk.EW, pady=(0, 15))
        row += 1
        
        self.source_var = tk.StringVar(value="0")
        ttk.Entry(source_frame, textvariable=self.source_var, width=25).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(source_frame, text="📁 Browse", width=10, command=self.browse_source).pack(side=tk.LEFT, padx=(5, 0))
        
        # Model
        ttk.Label(tab, text="YOLO Model", font=("Arial", 11, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        model_frame = ttk.Frame(tab)
        model_frame.grid(row=row, column=0, sticky=tk.EW, pady=(0, 15))
        row += 1
        
        self.model_var = tk.StringVar(value="yolo26m.pt")
        models = [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
        ]
        ttk.Combobox(model_frame, textvariable=self.model_var, values=models, width=22).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="📁", width=3, command=self.browse_model).pack(side=tk.LEFT, padx=(5, 0))
        
        # Confidence
        ttk.Label(tab, text="Confidence Threshold", font=("Arial", 11, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        conf_frame = ttk.Frame(tab)
        conf_frame.grid(row=row, column=0, sticky=tk.EW, pady=(0, 15))
        row += 1
        
        self.conf_var = tk.DoubleVar(value=0.50)
        self.conf_label = ttk.Label(conf_frame, text="0.50", width=5)
        self.conf_label.pack(side=tk.RIGHT)
        ttk.Scale(conf_frame, from_=0.1, to=0.9, variable=self.conf_var, orient=tk.HORIZONTAL,
                  command=self.update_conf_label).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Image Size
        ttk.Label(tab, text="Image Size", font=("Arial", 11, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        self.img_size_var = tk.IntVar(value=640)
        sizes = [320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280]
        ttk.Combobox(tab, textvariable=self.img_size_var, values=sizes, width=27, state="readonly").grid(row=row, column=0, sticky=tk.EW, pady=(0, 20))
        row += 1
        
        # Visualization
        ttk.Label(tab, text="Visualization", font=("Arial", 11, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        ttk.Checkbutton(tab, text="Show Trajectories", variable=self.show_trajectories).grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        ttk.Checkbutton(tab, text="Show Heatmap", variable=self.show_heatmap).grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        tab.columnconfigure(0, weight=1)
        
    def setup_filters_tab(self):
        """Filters and detection tab"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="🎨 Filters & Detection")
        
        row = 0
        
        # Image Enhancement
        ttk.Label(tab, text="Image Enhancement", font=("Arial", 11, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        enhance_frame = ttk.LabelFrame(tab, text="Mode", padding=5)
        enhance_frame.grid(row=row, column=0, sticky=tk.EW, pady=(0, 15))
        row += 1
        
        modes = [
            ("None", "none"),
            ("Edge Detection", "edge"),
            ("High Contrast", "contrast"),
            ("Thermal-like", "thermal"),
            ("Night Vision", "night")
        ]
        
        for label, value in modes:
            ttk.Radiobutton(enhance_frame, text=label, variable=self.enhance_mode, value=value).pack(anchor=tk.W, pady=2)
        
        # Class Filtering
        ttk.Label(tab, text="Detect Only (Select Classes)", font=("Arial", 11, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        # Search box
        search_frame = ttk.Frame(tab)
        search_frame.grid(row=row, column=0, sticky=tk.EW, pady=(0, 5))
        row += 1
        
        self.class_search_var = tk.StringVar()
        self.class_search_var.trace('w', self.filter_class_list)
        ttk.Entry(search_frame, textvariable=self.class_search_var, width=25).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(search_frame, text="🔍").pack(side=tk.LEFT, padx=(5, 0))
        
        # Class list frame with scrollbar
        list_frame = ttk.Frame(tab)
        list_frame.grid(row=row, column=0, sticky=tk.NSEW, pady=(0, 5))
        row += 1
        
        # Listbox with scrollbar (cleaner than checkboxes)
        self.class_listbox_widget = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, 
                                               height=15, font=("Arial", 9),
                                               bg='white', fg='black')
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.class_listbox_widget.yview)
        self.class_listbox_widget.configure(yscrollcommand=scrollbar.set)
        
        self.class_listbox_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Quick select buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=row, column=0, sticky=tk.EW)
        row += 1
        
        ttk.Button(btn_frame, text="Select All", command=self.select_all_classes, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Clear All", command=self.clear_all_classes, width=15).pack(side=tk.LEFT)
        
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)
        
    def setup_advanced_tab(self):
        """Advanced features tab"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="🔧 Advanced")
        
        row = 0
        
        # ROI Tools
        ttk.Label(tab, text="ROI & Counting Tools", font=("Arial", 11, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        row += 1
        
        ttk.Button(tab, text="Draw Detection Zone (ROI)", command=lambda: self.set_drawing_mode('roi'), width=30).grid(row=row, column=0, pady=5)
        row += 1
        ttk.Button(tab, text="Draw Counting Line", command=lambda: self.set_drawing_mode('line'), width=30).grid(row=row, column=0, pady=5)
        row += 1
        ttk.Button(tab, text="Clear All ROI/Lines", command=self.clear_roi, width=30).grid(row=row, column=0, pady=(5, 15))
        row += 1
        
        # Follow Target
        ttk.Label(tab, text="Follow Specific Target", font=("Arial", 11, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        row += 1
        
        follow_frame = ttk.Frame(tab)
        follow_frame.grid(row=row, column=0, sticky=tk.EW, pady=(0, 15))
        row += 1
        
        ttk.Label(follow_frame, text="ID:").pack(side=tk.LEFT)
        self.follow_id_var = tk.StringVar()
        ttk.Entry(follow_frame, textvariable=self.follow_id_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(follow_frame, text="Set", command=self.set_follow_target, width=8).pack(side=tk.LEFT)
        
        # Export
        ttk.Label(tab, text="Export Data", font=("Arial", 11, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        row += 1
        
        ttk.Button(tab, text="Export Trajectories (JSON)", command=self.export_trajectories, width=30).grid(row=row, column=0, pady=5)
        row += 1
        
        # Stats
        ttk.Label(tab, text="Object Statistics", font=("Arial", 11, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(10, 5))
        row += 1
        
        self.stats_text = tk.Text(tab, height=12, width=35, font=("Courier", 9), bg='#1e1e1e', fg='#00ff00')
        self.stats_text.grid(row=row, column=0, sticky=tk.NSEW, pady=(0, 5))
        row += 1
        
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(9, weight=1)
    
    def populate_class_checkboxes(self):
        """Populate class listbox with COCO classes"""
        self.class_listbox_widget.delete(0, tk.END)
        
        if self.detector and hasattr(self.detector.model, 'names'):
            self.all_class_names = list(self.detector.model.names.values())
        else:
            # Default COCO classes
            self.all_class_names = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ]
        
        # Populate listbox
        for cls_name in sorted(self.all_class_names):
            self.class_listbox_widget.insert(tk.END, cls_name)
        
        # Select all by default
        self.class_listbox_widget.select_set(0, tk.END)
    
    def filter_class_list(self, *args):
        """Filter class list based on search"""
        search_term = self.class_search_var.get().lower()
        
        # Save current selections
        selected_items = [self.class_listbox_widget.get(i) for i in self.class_listbox_widget.curselection()]
        
        # Clear and repopulate with filtered items
        self.class_listbox_widget.delete(0, tk.END)
        
        for cls_name in sorted(self.all_class_names):
            if search_term in cls_name.lower():
                self.class_listbox_widget.insert(tk.END, cls_name)
                # Restore selection if it was selected before
                if cls_name in selected_items:
                    self.class_listbox_widget.select_set(tk.END)
    
    def select_all_classes(self):
        """Select all classes in listbox"""
        self.class_listbox_widget.select_set(0, tk.END)
    
    def clear_all_classes(self):
        """Clear all class selections"""
        self.class_listbox_widget.selection_clear(0, tk.END)
    
    def get_enabled_classes(self):
        """Get list of enabled class names"""
        selected_indices = self.class_listbox_widget.curselection()
        if not selected_indices:
            return set()
        return {self.class_listbox_widget.get(i) for i in selected_indices}
    
    def update_conf_label(self, val):
        self.conf_label.config(text=f"{float(val):.2f}")
        if self.detector:
            self.detector.conf_threshold = float(val)
    
    def browse_source(self):
        filename = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.source_var.set(filename)
    
    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select YOLO model (.pt)",
            filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.model_var.set(filename)
    
    def on_preset_change(self, event=None):
        """Handle preset source selection"""
        preset = self.source_preset.get()
        if preset == "Webcam 0":
            self.source_var.set("0")
        elif preset == "Webcam 1":
            self.source_var.set("1")
        elif preset == "Webcam 2":
            self.source_var.set("2")
        elif preset == "Video File...":
            self.browse_source()
        elif preset == "RTSP Stream":
            self.source_var.set("rtsp://")
        # Custom: user can type directly
    
    def start_detection(self):
        """Start detection"""
        try:
            source = self.source_var.get()
            source = int(source) if source.isdigit() else source
            
            self.cap = open_video_source(source)
            
            ret, frame = self.cap.read()
            if ret:
                self.frame_height, self.frame_width = frame.shape[:2]
                self.heatmap_data = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self.detector = ObjectDetector(
                model_name=self.model_var.get(),
                conf_threshold=self.conf_var.get(),
                img_size=self.img_size_var.get()
            )
            
            # Populate class filters
            self.populate_class_checkboxes()
            
            self.is_running = True
            self.is_paused = False
            self.frame_count = 0
            self.last_time = time.time()
            
            # Clear queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    break
            
            # Update UI
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.NORMAL)
            self.record_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Running ({self.detector.device.upper()})", fg='#28a745')
            
            # Start threads
            self.display_running = True
            self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
            self.display_thread.start()
            
            threading.Thread(target=self.process_frames, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start:\n{str(e)}")
            self.status_label.config(text="Error", fg='#dc3545')
    
    def stop_detection(self):
        """Stop detection"""
        self.is_running = False
        self.display_running = False
        
        # Wait for threads to finish
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        if self.is_recording:
            self.toggle_recording()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.record_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped", fg='#ffc107')
        self.canvas.delete("all")
    
    def toggle_pause(self):
        """Toggle pause"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶ RESUME", bg='#28a745', fg='white')
            self.status_label.config(text="Paused", fg='#ffc107')
        else:
            self.pause_btn.config(text="⏸ PAUSE", bg='#ffc107', fg='black')
            self.status_label.config(text=f"Running ({self.detector.device.upper()})", fg='#28a745')
    
    def toggle_recording(self):
        """Toggle recording"""
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (self.frame_width, self.frame_height))
            self.is_recording = True
            self.record_btn.config(bg='#dc3545')
        else:
            if self.video_writer:
                self.video_writer.release()
            self.is_recording = False
            self.record_btn.config(bg='#6c757d')
    
    def take_screenshot(self):
        """Save screenshot"""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.status_label.config(text=f"Screenshot saved", fg='#17a2b8')
    
    def set_drawing_mode(self, mode):
        """Set drawing mode with user feedback"""
        if not self.is_running:
            messagebox.showwarning("Warning", "Please start detection first before drawing ROI/Line")
            return
            
        self.drawing_mode = mode
        self.temp_points = []
        
        if mode == 'roi':
            self.status_label.config(text="Click to draw ROI polygon. Right-click to finish (min 3 points).", fg='#ffc107')
        elif mode == 'line':
            self.status_label.config(text="Click two points to draw counting line.", fg='#ffc107')
    
    def clear_roi(self):
        self.roi_points = []
        self.counting_line = None
        self.temp_points = []
        self.drawing_mode = None
        self.object_counts = {"in": 0, "out": 0}
        self.crossed_ids = set()
    
    def set_follow_target(self):
        try:
            target_id = self.follow_id_var.get().strip()
            self.follow_target_id = int(target_id) if target_id else None
        except ValueError:
            self.follow_target_id = None
    
    def export_trajectories(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if filename:
            data = {str(obj_id): [{"x": int(x), "y": int(y)} for x, y in points] for obj_id, points in self.trajectories.items()}
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
    
    def on_canvas_click(self, event):
        """Handle left-click on canvas"""
        if not self.is_running:
            return
            
        # Convert canvas coordinates to frame coordinates
        frame_x, frame_y = self.canvas_to_frame_coords(event.x, event.y)
        if frame_x is None:
            return
            
        if self.drawing_mode == 'roi':
            self.temp_points.append((frame_x, frame_y))
            self.status_label.config(text=f"ROI point {len(self.temp_points)} added. Right-click to finish.", fg='#ffc107')
        elif self.drawing_mode == 'line':
            self.temp_points.append((frame_x, frame_y))
            if len(self.temp_points) == 1:
                self.status_label.config(text="Click second point for counting line", fg='#ffc107')
            elif len(self.temp_points) == 2:
                self.counting_line = self.temp_points.copy()
                self.temp_points = []
                self.drawing_mode = None
                self.status_label.config(text="Counting line set!", fg='#28a745')
    
    def on_canvas_right_click(self, event):
        """Handle right-click to finish ROI drawing"""
        if self.drawing_mode == 'roi' and len(self.temp_points) >= 3:
            self.roi_points = self.temp_points.copy()
            self.temp_points = []
            self.drawing_mode = None
            self.status_label.config(text="ROI zone set!", fg='#28a745')
    
    def canvas_to_frame_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to frame coordinates"""
        if not hasattr(self, 'frame_width') or not hasattr(self, 'frame_height'):
            return None, None
            
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            return None, None
        
        # Calculate displayed frame size and position
        aspect = self.frame_width / self.frame_height
        if aspect > canvas_w / canvas_h:
            display_w = canvas_w
            display_h = int(canvas_w / aspect)
        else:
            display_h = canvas_h
            display_w = int(canvas_h * aspect)
        
        offset_x = (canvas_w - display_w) // 2
        offset_y = (canvas_h - display_h) // 2
        
        # Check if click is within displayed frame
        if canvas_x < offset_x or canvas_x > offset_x + display_w:
            return None, None
        if canvas_y < offset_y or canvas_y > offset_y + display_h:
            return None, None
        
        # Convert to frame coordinates
        frame_x = int((canvas_x - offset_x) * self.frame_width / display_w)
        frame_y = int((canvas_y - offset_y) * self.frame_height / display_h)
        
        return frame_x, frame_y
    
    def process_frames(self):
        """Main loop"""
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                if self.source_var.get().isdigit():
                    break
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            
            # Apply enhancement
            frame = self.apply_enhancement(frame)
            
            # Detection
            results = self.detector.track(frame)
            
            # Filter by enabled classes
            enabled_classes = self.get_enabled_classes()
            if enabled_classes and len(enabled_classes) < len(self.all_class_names):
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    keep = []
                    for i, box in enumerate(boxes):
                        cls_name = results[0].names[int(box.cls[0])]
                        if cls_name in enabled_classes:
                            keep.append(i)
                    if keep:
                        results[0].boxes = boxes[keep]
                    else:
                        results[0].boxes = None
            
            annotated_frame = results[0].plot(line_width=1, font_size=0.4)
            
            # Process detections
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    if box.id is not None:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        obj_id = int(box.id[0])
                        
                        self.trajectories[obj_id].append((cx, cy))
                        
                        if self.heatmap_data is not None:
                            cv2.circle(self.heatmap_data, (cx, cy), 20, 1, -1)
                        
                        if obj_id not in self.object_stats:
                            self.object_stats[obj_id] = {
                                "class": results[0].names[int(box.cls[0])],
                                "first_seen": self.frame_count,
                                "last_seen": self.frame_count
                            }
                        else:
                            self.object_stats[obj_id]["last_seen"] = self.frame_count
            
            # Draw trajectories
            if self.show_trajectories.get():
                for points in self.trajectories.values():
                    if len(points) > 1:
                        pts = np.array(points, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [pts], False, (0, 255, 255), 1)
            
            # Draw heatmap
            if self.show_heatmap.get() and self.heatmap_data is not None:
                hm = cv2.normalize(self.heatmap_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                hm_colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, hm_colored, 0.3, 0)
            
            # Draw ROI
            if self.roi_points:
                pts = np.array(self.roi_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, (0, 255, 0), 2)
            
            # Draw counting line
            if self.counting_line:
                cv2.line(annotated_frame, self.counting_line[0], self.counting_line[1], (255, 0, 255), 2)
            
            # Draw follow target
            if self.follow_target_id and self.follow_target_id in self.trajectories:
                points = self.trajectories[self.follow_target_id]
                if points:
                    cx, cy = points[-1]
                    cv2.circle(annotated_frame, (cx, cy), 25, (0, 0, 255), 2)
            
            if self.is_recording and self.video_writer:
                self.video_writer.write(annotated_frame)
            
            self.current_frame = annotated_frame.copy()
            
            # FPS
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                current_time = time.time()
                self.fps = 10 / (current_time - self.last_time)
                self.last_time = current_time
                # Update UI in main thread
                self.root.after(0, lambda: self.fps_label.config(text=f"{self.fps:.1f}"))
                self.root.after(0, self.update_stats)
            
            # Put frame in queue for display thread (non-blocking)
            try:
                self.frame_queue.put_nowait(annotated_frame)
            except:
                # Queue full, skip this frame
                pass
        
        if self.cap:
            self.cap.release()
    
    def display_loop(self):
        """Separate thread for smooth display from frame cache"""
        while self.display_running:
            try:
                # Get frame from queue (blocking with timeout)
                frame = self.frame_queue.get(timeout=0.1)
                
                # Display in main thread
                self.root.after(0, self._display_frame_internal, frame)
                
            except Empty:
                # No frame available, continue
                time.sleep(0.01)
                continue
    
    def _display_frame_internal(self, frame):
        """Internal method to display frame (runs in main thread)"""
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w > 1 and h > 1:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fh, fw = frame_rgb.shape[:2]
            aspect = fw / fh
            if aspect > w / h:
                new_w, new_h = w, int(w / aspect)
            else:
                new_h, new_w = h, int(h * aspect)
            
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            img = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            self.canvas.create_image(w // 2, h // 2, image=img, anchor=tk.CENTER)
            self.canvas.imgtk = img
    
    def display_frame(self, frame):
        """Legacy method - now redirects to internal display"""
        self._display_frame_internal(frame)
    
    def update_stats(self):
        """Update stats display"""
        active = {k: v for k, v in self.object_stats.items() if self.frame_count - v["last_seen"] < 30}
        stats = f"Active: {len(active)}\nFrame: {self.frame_count}\nFPS: {self.fps:.1f}\n" + "-" * 30 + "\n"
        for obj_id, s in sorted(active.items())[:10]:
            stats += f"ID {obj_id}: {s['class']}\n"
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
    
    def apply_enhancement(self, frame):
        """Apply enhancement modes"""
        mode = self.enhance_mode.get()
        
        if mode == "edge":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(frame, 0.7, edges_bgr, 0.3, 0)
        
        elif mode == "contrast":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        elif mode == "thermal":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        elif mode == "night":
            enhanced = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
            b, g, r = cv2.split(enhanced)
            g = cv2.add(g, 50)
            return cv2.merge([b, g, r])
        
        return frame


def main():
    root = tk.Tk()
    app = DroneDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
