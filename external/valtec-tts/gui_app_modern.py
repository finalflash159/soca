import sys
import os
import threading
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import numpy as np
import soundfile as sf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import customtkinter as ctk

# Cấu hình CustomTkinter - Modern Theme
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue", "green", "dark-blue"

# Thêm thư mục hiện tại vào đường dẫn để import valtec_tts local
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Modern Color Palette
COLORS = {
    "primary": "#6366F1",        # Indigo 500
    "primary_hover": "#4F46E5",  # Indigo 600
    "primary_light": "#818CF8",  # Indigo 400
    "success": "#10B981",        # Emerald 500
    "warning": "#F59E0B",        # Amber 500
    "error": "#EF4444",          # Red 500
    "info": "#3B82F6",           # Blue 500
    "bg_dark": "#0F172A",        # Slate 900
    "bg_card": "#1E293B",        # Slate 800
    "bg_hover": "#334155",       # Slate 700
    "text_primary": "#F8FAFC",   # Slate 50
    "text_secondary": "#94A3B8", # Slate 400
    "border": "#334155",         # Slate 700
}

# ---- Redirect stdout/stderr để bắt log ----
class LogRedirector:
    def __init__(self, text_widget, queue_obj):
        self.text_widget = text_widget
        self.queue = queue_obj

    def write(self, string):
        if string.strip():
            self.queue.put(string)

    def flush(self):
        pass
# ---------------------------------------------

class ModernButton(ctk.CTkButton):
    """Custom button with modern styling"""
    def __init__(self, master, **kwargs):
        defaults = {
            "fg_color": COLORS["primary"],
            "hover_color": COLORS["primary_hover"],
            "corner_radius": 8,
            "font": ctk.CTkFont(size=13, weight="bold"),
            "height": 40,
            "border_width": 0,
        }
        defaults.update(kwargs)
        super().__init__(master, **defaults)

class ModernCard(ctk.CTkFrame):
    """Card component with modern styling"""
    def __init__(self, master, **kwargs):
        defaults = {
            "fg_color": COLORS["bg_card"],
            "corner_radius": 12,
            "border_width": 1,
            "border_color": COLORS["border"],
        }
        defaults.update(kwargs)
        super().__init__(master, **defaults)

class IconButton(ctk.CTkButton):
    """Button with icon"""
    def __init__(self, master, icon="", text="", **kwargs):
        defaults = {
            "width": 36,
            "height": 36,
            "corner_radius": 8,
            "fg_color": "transparent",
            "hover_color": COLORS["bg_hover"],
            "text_color": COLORS["text_secondary"],
            "font": ctk.CTkFont(size=16),
        }
        defaults.update(kwargs)
        super().__init__(master, text=f"{icon}", **defaults)

class ValtecTTSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Valtec TTS - Tiếng Việt")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        # Cấu hình grid chính
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Variables
        self.tts = None
        self.speakers = []
        self.audio_results = {}
        self.task_counter = 0
        
        # Thread pool - increased for multi-line processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.pending_tasks = 0
        self.log_queue = queue.Queue()
        
        # Setup UI
        self.setup_header()
        self.setup_main_layout()
        
        # Bắt đầu theo dõi hàng đợi log
        self.after(100, self.process_log_queue)

        # Chuyển hướng stdout/stderr
        self.redirector = LogRedirector(self.log_textbox, self.log_queue)
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.redirector
        sys.stderr = self.redirector

        # Load mô hình ở background
        threading.Thread(target=self._load_model, daemon=True).start()

    def setup_header(self):
        """Setup modern header bar"""
        self.header = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color=COLORS["bg_card"])
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.grid_columnconfigure(1, weight=1)
        
        # Logo và tiêu đề
        logo_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=20, pady=10)
        
        logo_icon = ctk.CTkLabel(
            logo_frame, 
            text="🔊", 
            font=ctk.CTkFont(size=24)
        )
        logo_icon.pack(side="left", padx=(0, 10))
        
        logo_text = ctk.CTkLabel(
            logo_frame,
            text="Valtec TTS",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        logo_text.pack(side="left")
        
        version_label = ctk.CTkLabel(
            logo_frame,
            text="v1.0.5",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["primary_light"],
            fg_color=COLORS["primary"],
            corner_radius=4,
            width=40
        )
        version_label.pack(side="left", padx=(10, 0))
        
        # Status indicator
        self.status_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        self.status_frame.grid(row=0, column=2, padx=20, pady=10)
        
        self.status_dot = ctk.CTkLabel(
            self.status_frame,
            text="●",
            font=ctk.CTkFont(size=14),
            text_color=COLORS["warning"]
        )
        self.status_dot.pack(side="left", padx=(0, 5))
        
        self.status_text = ctk.CTkLabel(
            self.status_frame,
            text="Đang khởi tạo...",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"]
        )
        self.status_text.pack(side="left")

    def setup_main_layout(self):
        """Setup main layout with sidebar and content"""
        self.main_container = ctk.CTkFrame(self, fg_color=COLORS["bg_dark"])
        self.main_container.grid(row=1, column=0, sticky="nsew")
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)
        
        # Sidebar
        self.setup_sidebar()
        
        # Content area
        self.setup_content_area()

    def setup_sidebar(self):
        """Setup modern sidebar"""
        self.sidebar = ctk.CTkFrame(
            self.main_container, 
            width=280, 
            corner_radius=0,
            fg_color=COLORS["bg_card"]
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(3, weight=1)
        
        # Section: Voice Settings
        self.create_section_header(self.sidebar, "⚙️ Cài đặt Giọng đọc", 0)
        
        voice_card = ModernCard(self.sidebar)
        voice_card.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="ew")
        
        # Speaker selection
        speaker_label = ctk.CTkLabel(
            voice_card,
            text="🎙️ Giọng đọc",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_secondary"]
        )
        speaker_label.pack(anchor="w", padx=15, pady=(15, 5))
        
        self.speaker_combo = ctk.CTkComboBox(
            voice_card,
            values=["Đang tải..."],
            font=ctk.CTkFont(size=13),
            dropdown_font=ctk.CTkFont(size=12),
            corner_radius=8,
            height=40,
            border_color=COLORS["border"],
            button_color=COLORS["primary"],
            button_hover_color=COLORS["primary_hover"]
        )
        self.speaker_combo.pack(fill="x", padx=15, pady=(0, 15))
        
        # Speed control
        speed_header = ctk.CTkFrame(voice_card, fg_color="transparent")
        speed_header.pack(fill="x", padx=15, pady=(0, 5))
        
        speed_label = ctk.CTkLabel(
            speed_header,
            text="⚡ Tốc độ",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_secondary"]
        )
        speed_label.pack(side="left")
        
        self.speed_value_label = ctk.CTkLabel(
            speed_header,
            text="1.00x",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["primary_light"]
        )
        self.speed_value_label.pack(side="right")
        
        self.speed_slider = ctk.CTkSlider(
            voice_card,
            from_=0.5,
            to=2.0,
            number_of_steps=15,
            command=self._update_speed_label,
            progress_color=COLORS["primary"],
            button_color=COLORS["primary"],
            button_hover_color=COLORS["primary_hover"]
        )
        self.speed_slider.set(1.0)
        self.speed_slider.pack(fill="x", padx=15, pady=(0, 15))
        
        # Section: Tools
        self.create_section_header(self.sidebar, "🛠️ Công cụ", 2)
        
        tools_card = ModernCard(self.sidebar)
        tools_card.grid(row=3, column=0, padx=15, pady=(0, 15), sticky="new")
        
        self.load_btn = ModernButton(
            tools_card,
            text="📁  Mở File (.txt/.srt)",
            command=self.load_file,
            height=44
        )
        self.load_btn.pack(fill="x", padx=15, pady=(15, 10))
        
        self.clear_btn = ctk.CTkButton(
            tools_card,
            text="🗑️  Xoá Nội Dung",
            command=self.clear_text,
            height=44,
            corner_radius=8,
            fg_color="transparent",
            border_width=2,
            border_color=COLORS["border"],
            text_color=COLORS["text_secondary"],
            hover_color=COLORS["bg_hover"],
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.clear_btn.pack(fill="x", padx=15, pady=(0, 15))
        
        # Multi-line mode toggle
        multiline_frame = ctk.CTkFrame(voice_card, fg_color="transparent")
        multiline_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        self.multiline_var = ctk.BooleanVar(value=False)
        self.multiline_switch = ctk.CTkSwitch(
            multiline_frame,
            text="🔀 Xử lý từng dòng riêng biệt",
            variable=self.multiline_var,
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"],
            progress_color=COLORS["primary"],
            button_color=COLORS["primary"],
            button_hover_color=COLORS["primary_hover"]
        )
        self.multiline_switch.pack(anchor="w")
        
        # Separator line
        separator = ctk.CTkFrame(voice_card, height=1, fg_color=COLORS["border"])
        separator.pack(fill="x", padx=15, pady=(0, 15))
        
        # Stats at bottom
        stats_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        stats_frame.grid(row=4, column=0, padx=15, pady=15, sticky="ew")
        
        self.stats_label = ctk.CTkLabel(
            stats_frame,
            text="📊 0 Audio đã tạo",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"]
        )
        self.stats_label.pack(anchor="w")

    def create_section_header(self, parent, text, row):
        """Create a section header"""
        header = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_secondary"]
        )
        header.grid(row=row, column=0, padx=20, pady=(20, 10), sticky="w")

    def setup_content_area(self):
        """Setup main content area"""
        self.content = ctk.CTkFrame(self.main_container, fg_color=COLORS["bg_dark"])
        self.content.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=2)
        self.content.grid_rowconfigure(2, weight=3)
        
        # Input Card
        input_card = ModernCard(self.content)
        input_card.grid(row=1, column=0, sticky="nsew", pady=(0, 15))
        input_card.grid_columnconfigure(0, weight=1)
        input_card.grid_rowconfigure(1, weight=1)
        
        # Input header
        input_header = ctk.CTkFrame(input_card, fg_color="transparent", height=50)
        input_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 0))
        
        input_title = ctk.CTkLabel(
            input_header,
            text="📝 Nhập văn bản",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        input_title.pack(side="left")
        
        char_count = ctk.CTkLabel(
            input_header,
            text="0 ký tự",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"]
        )
        char_count.pack(side="right")
        self.char_count_label = char_count
        
        # Text input
        self.text_input = ctk.CTkTextbox(
            input_card,
            font=ctk.CTkFont(size=14),
            corner_radius=8,
            border_color=COLORS["border"],
            wrap="word"
        )
        self.text_input.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.text_input.insert("0.0", "Nhập văn bản tiếng Việt vào đây...\n\n💡 Mẹo: Bật 'Xử lý từng dòng riêng biệt' để tạo nhiều audio từ nhiều dòng văn bản cùng lúc")
        
        # Bind text change
        self.text_input.bind("<KeyRelease>", lambda e: self._update_char_count())
        
        # Generate button bar
        btn_frame = ctk.CTkFrame(input_card, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 15))
        
        self.generate_btn = ModernButton(
            btn_frame,
            text="🚀 Tạo Audio",
            command=self.generate_audio_task,
            state="disabled",
            height=48,
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.generate_btn.pack(fill="x")
        
        # Results & Log Tabs
        results_card = ModernCard(self.content)
        results_card.grid(row=2, column=0, sticky="nsew")
        results_card.grid_columnconfigure(0, weight=1)
        results_card.grid_rowconfigure(0, weight=1)
        
        self.tabview = ctk.CTkTabview(
            results_card,
            corner_radius=8,
            segmented_button_selected_color=COLORS["primary"],
            segmented_button_selected_hover_color=COLORS["primary_hover"],
            segmented_button_unselected_color=COLORS["bg_hover"],
            segmented_button_unselected_hover_color=COLORS["border"],
            text_color=COLORS["text_secondary"]
        )
        self.tabview.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        # Audio List Tab
        self.tabview.add("🎵 Danh sách Audio")
        audio_tab = self.tabview.tab("🎵 Danh sách Audio")
        audio_tab.grid_columnconfigure(0, weight=1)
        audio_tab.grid_rowconfigure(0, weight=1)
        
        # Scrollable frame for audio items
        self.audio_list_frame = ctk.CTkScrollableFrame(
            audio_tab,
            fg_color="transparent",
            corner_radius=0
        )
        self.audio_list_frame.grid(row=0, column=0, sticky="nsew")
        
        # Empty state
        self.empty_state = ctk.CTkFrame(self.audio_list_frame, fg_color="transparent")
        self.empty_state.pack(expand=True, pady=50)
        
        empty_icon = ctk.CTkLabel(
            self.empty_state,
            text="🎧",
            font=ctk.CTkFont(size=48)
        )
        empty_icon.pack()
        
        empty_text = ctk.CTkLabel(
            self.empty_state,
            text="Chưa có audio nào được tạo",
            font=ctk.CTkFont(size=14),
            text_color=COLORS["text_secondary"]
        )
        empty_text.pack(pady=(10, 0))
        
        # Log Tab
        self.tabview.add("📋 Log Hệ thống")
        log_tab = self.tabview.tab("📋 Log Hệ thống")
        log_tab.grid_columnconfigure(0, weight=1)
        log_tab.grid_rowconfigure(0, weight=1)
        
        self.log_textbox = ctk.CTkTextbox(
            log_tab,
            font=ctk.CTkFont(family="Consolas", size=12),
            state="disabled",
            wrap="word",
            corner_radius=8,
            fg_color=COLORS["bg_dark"],
            border_color=COLORS["border"]
        )
        self.log_textbox.grid(row=0, column=0, sticky="nsew")

    def _update_char_count(self):
        """Update character count label"""
        text = self.text_input.get("0.0", "end").strip()
        count = len(text)
        self.char_count_label.configure(text=f"{count} ký tự")

    def log(self, message):
        """Hàm gửi log vào textbox"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}\n")

    def process_log_queue(self):
        """Hàm lặp để đọc hàng dữ liệu Queue và viết vào giao diện"""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_textbox.configure(state="normal")
                
                # Color coding for different message types
                if "✅" in msg or "Hoàn thành" in msg:
                    color = COLORS["success"]
                elif "❌" in msg or "Lỗi" in msg or "lỗi" in msg.lower():
                    color = COLORS["error"]
                elif "⚠️" in msg or "Đang" in msg:
                    color = COLORS["warning"]
                else:
                    color = COLORS["text_primary"]
                
                self.log_textbox.insert("end", msg)
                self.log_textbox.see("end")
                self.log_textbox.configure(state="disabled")
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_log_queue)

    def _update_speed_label(self, value):
        self.speed_value_label.configure(text=f"{value:.2f}x")

    def _update_stats(self):
        """Update statistics label"""
        total = len(self.audio_results)
        if self.pending_tasks > 0:
            self.stats_label.configure(text=f"📊 {total} Audio đã tạo | {self.pending_tasks} đang xử lý")
        else:
            self.stats_label.configure(text=f"📊 {total} Audio đã tạo")

    def clear_text(self):
        self.text_input.delete("0.0", "end")
        self._update_char_count()

    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Chọn file văn bản",
            filetypes=(
                ("Text files", "*.txt"),
                ("Subtitle files", "*.srt"),
                ("All files", "*.*")
            )
        )
        if not filepath:
            return
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if filepath.lower().endswith('.srt'):
                content = self._parse_srt(content)
                
            self.text_input.delete("0.0", "end")
            self.text_input.insert("0.0", content)
            self._update_char_count()
            self.log(f"📁 Đã mở file: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Lỗi Định dạng", f"Không thể đọc file: {e}")

    def _parse_srt(self, srt_content):
        lines = []
        for line in srt_content.split('\n'):
            line = line.strip()
            if not line: continue
            if re.match(r'^\d+$', line): continue
            if re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$', line): continue
            lines.append(line)
        return ' '.join(lines)

    def _load_model(self):
        self.log("🔄 Đang khởi tạo hệ thống TTS...")
        try:
            from valtec_tts import TTS
            self.tts = TTS()
            self.speakers = self.tts.list_speakers()
            self.after(0, self._on_model_loaded)
        except Exception as e:
            self.log(f"❌ Lỗi: {str(e)}")
            self.after(0, self._on_model_error, str(e))

    def _on_model_loaded(self):
        self.speaker_combo.configure(values=self.speakers)
        if "NF" in self.speakers:
            self.speaker_combo.set("NF")
        elif self.speakers:
            self.speaker_combo.set(self.speakers[0])
            
        self.generate_btn.configure(state="normal")
        
        # Update status
        self.status_dot.configure(text_color=COLORS["success"])
        self.status_text.configure(
            text="Sẵn sàng",
            text_color=COLORS["success"]
        )
        
        self.log("✅ Hệ thống đã sẵn sàng!")

    def _on_model_error(self, err_msg):
        messagebox.showerror("Lỗi Khởi tạo", f"Không thể tải mô hình TTS:\n{err_msg}")
        self.status_dot.configure(text_color=COLORS["error"])
        self.status_text.configure(
            text="Lỗi khởi tạo",
            text_color=COLORS["error"]
        )
        self.log("❌ Khởi tạo thất bại!")

    def generate_audio_task(self):
        full_text = self.text_input.get("0.0", "end").strip()
        if not full_text:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập văn bản cần đọc!")
            return
        
        speaker = self.speaker_combo.get()
        speed = self.speed_slider.get()
        is_multiline = self.multiline_var.get()
        
        # Hide empty state
        self.empty_state.pack_forget()
        
        if is_multiline:
            # Split by lines and process each non-empty line as separate task
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            if not lines:
                messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập ít nhất một dòng văn bản!")
                return
            
            self.log(f"📋 Phát hiện {len(lines)} dòng văn bản để xử lý")
            
            self.pending_tasks += len(lines)
            for line in lines:
                task_id = self.task_counter
                self.task_counter += 1
                
                # Tạo UI item cho Task mới
                self._add_audio_item_ui(task_id, line, speaker)
                self.log(f"🔄 Bắt đầu tác vụ #{task_id}: {line[:30]}...")
                
                # Submit task
                self.executor.submit(self._do_generate, task_id, line, speaker, speed)
            
            self.log(f"✅ Đã thêm {len(lines)} tác vụ vào hàng đợi")
        else:
            # Process as single text
            task_id = self.task_counter
            self.task_counter += 1
            self.pending_tasks += 1
            
            # Tạo UI item cho Task mới
            self._add_audio_item_ui(task_id, full_text, speaker)
            self.log(f"🔄 Bắt đầu tác vụ #{task_id}: {full_text[:30]}...")
            
            # Submit task
            self.executor.submit(self._do_generate, task_id, full_text, speaker, speed)
        
        # Update stats
        self._update_stats()
        
    def _do_generate(self, task_id, text, speaker, speed):
        start_time = time.time()
        try:
            audio_data, sample_rate = self.tts.synthesize(text=text, speaker=speaker, speed=speed)
            cost_time = time.time() - start_time
            self.audio_results[task_id] = (audio_data, sample_rate, text, speaker)

            self.after(0, self._update_audio_item_ui, task_id, True, cost_time)
            self.after(0, self._task_completed)
            msg = f"✅ Tác vụ #{task_id} hoàn thành ({cost_time:.2f}s)"
            self.log(msg)
        except Exception as e:
            self.after(0, self._update_audio_item_ui, task_id, False, str(e))
            self.after(0, self._task_completed)
            self.log(f"❌ Lỗi tác vụ #{task_id}: {str(e)}")

    def _task_completed(self):
        """Called when a task completes (success or failure)"""
        self.pending_tasks = max(0, self.pending_tasks - 1)
        self._update_stats()

    def _add_audio_item_ui(self, task_id, text, speaker):
        """Thêm card hiển thị cho audio đang xử lý"""
        item_card = ModernCard(self.audio_list_frame)
        item_card.pack(fill="x", padx=5, pady=8)
        
        # Header with speaker badge
        header = ctk.CTkFrame(item_card, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(15, 10))
        
        speaker_badge = ctk.CTkLabel(
            header,
            text=f"🎙️ {speaker}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["primary_light"],
            fg_color=COLORS["primary"],
            corner_radius=6,
            width=60,
            height=24
        )
        speaker_badge.pack(side="left")

        status_label = ctk.CTkLabel(
            header,
            text="⏳ Đang xử lý...",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["warning"]
        )
        status_label.pack(side="right")
        
        # Preview text
        preview_text = text[:60] + ("..." if len(text) > 60 else "")
        info_label = ctk.CTkLabel(
            item_card,
            text=preview_text,
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_primary"],
            anchor="w",
            justify="left",
            wraplength=600
        )
        info_label.pack(fill="x", padx=15, pady=(0, 15))

        # Action buttons container (will be populated on completion)
        action_frame = ctk.CTkFrame(item_card, fg_color="transparent", height=40)
        action_frame.pack(fill="x", padx=15, pady=(0, 15))

        # Progress bar
        progress_bar = ctk.CTkProgressBar(
            item_card,
            mode="indeterminate",
            height=4,
            progress_color=COLORS["primary"]
        )
        progress_bar.pack(fill="x", padx=15, pady=(0, 15))
        progress_bar.start()

        # Lưu lại UI components
        setattr(self, f"ui_item_{task_id}", {
            "frame": item_card,
            "status_label": status_label,
            "info_label": info_label,
            "action_frame": action_frame,
            "progress_bar": progress_bar,
            "speaker_badge": speaker_badge
        })

    def _update_audio_item_ui(self, task_id, success, data):
        """Cập nhật trạng thái item khi xử lý xong"""
        ui_elements = getattr(self, f"ui_item_{task_id}", None)
        if not ui_elements:
            return

        status_label = ui_elements["status_label"]
        action_frame = ui_elements["action_frame"]
        progress_bar = ui_elements["progress_bar"]
        
        # Stop and hide progress bar
        progress_bar.stop()
        progress_bar.pack_forget()
        
        if success:
            status_label.configure(
                text=f"✅ Hoàn thành ({data:.1f}s)",
                text_color=COLORS["success"]
            )
            
            # Add action buttons
            play_btn = ctk.CTkButton(
                action_frame,
                text="▶  Nghe thử",
                width=100,
                height=36,
                corner_radius=8,
                fg_color=COLORS["success"],
                hover_color="#059669",
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda: self.play_audio(task_id)
            )
            play_btn.pack(side="left", padx=(0, 10))
            
            save_btn = ctk.CTkButton(
                action_frame,
                text="💾  Lưu",
                width=80,
                height=36,
                corner_radius=8,
                fg_color=COLORS["primary"],
                hover_color=COLORS["primary_hover"],
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda: self.save_audio(task_id)
            )
            save_btn.pack(side="left")
            
        else:
            status_label.configure(
                text="❌ Thất bại",
                text_color=COLORS["error"]
            )
            
            error_btn = ctk.CTkButton(
                action_frame,
                text="Xem lỗi",
                width=80,
                height=36,
                corner_radius=8,
                fg_color=COLORS["error"],
                hover_color="#DC2626",
                font=ctk.CTkFont(size=12),
                command=lambda: messagebox.showerror("Lỗi Audio", data)
            )
            error_btn.pack(side="left")

    def play_audio(self, task_id):
        if task_id not in self.audio_results:
            return
        audio_data, sample_rate, _, _ = self.audio_results[task_id]
        try:
            import sounddevice as sd
            sd.play(audio_data, sample_rate)
            self.log(f"▶ Đang phát audio #{task_id}...")
        except ImportError:
            messagebox.showwarning("Thiếu thư viện", "Vui lòng cài đặt 'sounddevice' để nghe thử.")
        except Exception as e:
            messagebox.showerror("Lỗi phát audio", str(e))

    def save_audio(self, task_id):
        if task_id not in self.audio_results:
            return
        
        audio_data, sample_rate, text, speaker = self.audio_results[task_id]
        
        safe_text = re.sub(r'[\\/*?:"<>|]', "", text[:20]).strip().replace(" ", "_")
        default_filename = f"{speaker}_{safe_text}.wav"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=(("WAV Audio", "*.wav"), ("All files", "*.*")),
            title="Lưu file audio",
            initialfile=default_filename
        )
        if not filepath:
            return
            
        try:
            sf.write(filepath, audio_data, sample_rate)
            msg = f"💾 Đã lưu: {os.path.basename(filepath)}"
            self.log(msg)
            messagebox.showinfo("Thành công", msg)
        except Exception as e:
            messagebox.showerror("Lỗi Lưu", f"Không thể lưu file: {e}")
            
    def destroy(self):
        """Dừng luồng trước khi thoát"""
        self.executor.shutdown(wait=False)
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        super().destroy()

if __name__ == "__main__":
    app = ValtecTTSApp()
    app.mainloop()
