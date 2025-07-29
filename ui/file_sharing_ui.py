"""
Anima File Sharing UI

This module provides a graphical interface for sharing files with Anima,
allowing users to upload images, documents, and other files to discuss
and store in Anima's memory system.
"""

import os
import sys
import time
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
from PIL import Image, ImageTk

# Add parent directory to path for Anima imports
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anima.file_sharing_ui")

# Constants
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
SUPPORTED_DOCUMENT_FORMATS = ('.pdf', '.txt', '.md', '.doc', '.docx', '.xls', '.xlsx', '.csv')
MAX_THUMBNAIL_SIZE = (150, 150)
FILE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shared_files")

# Ensure file storage directory exists
os.makedirs(FILE_STORAGE_DIR, exist_ok=True)

class FileMetadata:
    """Class to store and manage file metadata"""
    
    def __init__(self, file_path, description=""):
        self.original_path = file_path
        self.file_name = os.path.basename(file_path)
        self.description = description
        self.timestamp = datetime.now().isoformat()
        self.stored_path = self._store_file()
        self.file_type = self._determine_file_type()
        self.file_size = os.path.getsize(self.stored_path)
        self.thumbnail = None
        
    def _store_file(self):
        """Store a copy of the file in Anima's file storage"""
        stored_path = os.path.join(FILE_STORAGE_DIR, self.file_name)
        shutil.copy2(self.original_path, stored_path)
        return stored_path
        
    def _determine_file_type(self):
        """Determine the type of file based on extension"""
        ext = os.path.splitext(self.file_name)[1].lower()
        
        if ext in SUPPORTED_IMAGE_FORMATS:
            return "image"
        elif ext in SUPPORTED_DOCUMENT_FORMATS:
            return "document"
        else:
            return "other"
            
    def create_thumbnail(self):
        """Create a thumbnail for image files"""
        if self.file_type == "image":
            try:
                img = Image.open(self.stored_path)
                img.thumbnail(MAX_THUMBNAIL_SIZE)
                self.thumbnail = ImageTk.PhotoImage(img)
                return self.thumbnail
            except Exception as e:
                logger.error(f"Error creating thumbnail: {e}")
                return None
        return None
        
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "file_name": self.file_name,
            "description": self.description,
            "timestamp": self.timestamp,
            "stored_path": self.stored_path,
            "file_type": self.file_type,
            "file_size": self.file_size
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary"""
        instance = cls.__new__(cls)
        instance.file_name = data["file_name"]
        instance.description = data["description"]
        instance.timestamp = data["timestamp"]
        instance.stored_path = data["stored_path"]
        instance.file_type = data["file_type"]
        instance.file_size = data["file_size"]
        instance.original_path = data["stored_path"]
        instance.thumbnail = None
        return instance


class FileManager:
    """Manager for handling shared files and their metadata"""
    
    def __init__(self):
        self.files = []
        self.metadata_file = os.path.join(FILE_STORAGE_DIR, "file_metadata.json")
        self.load_metadata()
        
    def load_metadata(self):
        """Load file metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.files = [FileMetadata.from_dict(item) for item in data]
                logger.info(f"Loaded metadata for {len(self.files)} files")
            except Exception as e:
                logger.error(f"Error loading file metadata: {e}")
                
    def save_metadata(self):
        """Save file metadata to JSON file"""
        try:
            data = [file.to_dict() for file in self.files]
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved metadata for {len(self.files)} files")
        except Exception as e:
            logger.error(f"Error saving file metadata: {e}")
            
    def add_file(self, file_path, description=""):
        """Add a new file to the manager"""
        file_metadata = FileMetadata(file_path, description)
        self.files.append(file_metadata)
        self.save_metadata()
        return file_metadata
        
    def remove_file(self, file_metadata):
        """Remove a file from the manager"""
        if file_metadata in self.files:
            # Remove from storage
            try:
                if os.path.exists(file_metadata.stored_path):
                    os.remove(file_metadata.stored_path)
            except Exception as e:
                logger.error(f"Error removing file: {e}")
                
            # Remove from list
            self.files.remove(file_metadata)
            self.save_metadata()
            return True
        return False
        
    def get_files_by_type(self, file_type=None):
        """Get files filtered by type"""
        if file_type:
            return [f for f in self.files if f.file_type == file_type]
        return self.files


class FileSharingUI:
    """Main UI class for the file sharing interface"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Anima File Sharing")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)
        
        self.file_manager = FileManager()
        self.current_view = "all"  # all, images, documents
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the main UI components"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Header with title and buttons
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(column=0, row=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Title
        title_label = ttk.Label(header_frame, text="Anima File Sharing", font=("Arial", 16, "bold"))
        title_label.grid(column=0, row=0, sticky=tk.W)
        
        # Add file button
        self.add_button = ttk.Button(header_frame, text="Add File", command=self.add_file)
        self.add_button.grid(column=1, row=0, sticky=tk.E, padx=(0, 5))
        
        # Filter buttons
        filter_frame = ttk.Frame(header_frame)
        filter_frame.grid(column=2, row=0, sticky=tk.E)
        
        self.all_button = ttk.Button(filter_frame, text="All", command=lambda: self.filter_files("all"))
        self.all_button.grid(column=0, row=0, padx=2)
        
        self.images_button = ttk.Button(filter_frame, text="Images", command=lambda: self.filter_files("image"))
        self.images_button.grid(column=1, row=0, padx=2)
        
        self.docs_button = ttk.Button(filter_frame, text="Documents", command=lambda: self.filter_files("document"))
        self.docs_button.grid(column=2, row=0, padx=2)
        
        # File display area (canvas with scrollbar)
        self.canvas_frame = ttk.Frame(main_frame, borderwidth=1, relief="sunken")
        self.canvas_frame.grid(column=0, row=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(0, weight=1)
        
        # Canvas and scrollbar
        self.canvas = tk.Canvas(self.canvas_frame, borderwidth=0, background="#ffffff")
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.grid(column=1, row=0, sticky=(tk.N, tk.S))
        self.canvas.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame for file cards
        self.files_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.files_frame, anchor="nw", tags="files_frame")
        self.files_frame.bind("<Configure>", self.on_frame_configure)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor=tk.W)
        status_bar.grid(column=0, row=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Initial UI update
        self.update_ui()
        
    def on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def add_file(self):
        """Open file dialog to add new file"""
        file_paths = filedialog.askopenfilenames(
            title="Select File(s) to Share",
            filetypes=[
                ("Images", " ".join(["*" + fmt for fmt in SUPPORTED_IMAGE_FORMATS])),
                ("Documents", " ".join(["*" + fmt for fmt in SUPPORTED_DOCUMENT_FORMATS])),
                ("All Files", "*.*")
            ]
        )
        
        if not file_paths:
            return
            
        # For each selected file
        added_count = 0
        for file_path in file_paths:
            # Ask for description
            description = self.prompt_for_description(os.path.basename(file_path))
            
            # Add file to manager
            try:
                self.file_manager.add_file(file_path, description)
                added_count += 1
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add file: {e}")
                
        # Update UI
        self.status_var.set(f"Added {added_count} file(s)")
        self.update_ui()
        
    def prompt_for_description(self, filename):
        """Prompt user for file description"""
        dialog = tk.Toplevel(self.root)
        dialog.title("File Description")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text=f"Enter a description for '{filename}':", padding=10).pack(fill="x")
        
        description = tk.Text(dialog, height=3, width=40, wrap="word")
        description.pack(padx=10, pady=5, fill="both", expand=True)
        description.focus_set()
        
        result = [None]
        
        def on_ok():
            result[0] = description.get("1.0", "end-1c").strip()
            dialog.destroy()
            
        def on_cancel():
            dialog.destroy()
            
        button_frame = ttk.Frame(dialog)
        button_frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="right", padx=5)
        
        self.root.wait_window(dialog)
        return result[0] or ""
        
    def filter_files(self, file_type):
        """Filter displayed files by type"""
        self.current_view = file_type
        self.update_ui()
        
    def update_ui(self):
        """Update the file display based on current filter"""
        # Clear existing file cards
        for widget in self.files_frame.winfo_children():
            widget.destroy()
            
        # Get filtered files
        if self.current_view == "all":
            files = self.file_manager.files
        else:
            files = self.file_manager.get_files_by_type(self.current_view)
            
        # Update status bar
        self.status_var.set(f"Displaying {len(files)} file(s)")
        
        # Sort files by timestamp (newest first)
        files = sorted(files, key=lambda f: f.timestamp, reverse=True)
        
        # Create file cards
        for i, file_metadata in enumerate(files):
            self.create_file_card(i, file_metadata)
            
        # Update canvas scroll region
        self.files_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def create_file_card(self, index, file_metadata):
        """Create a card widget for a file"""
        row = index // 3
        col = index % 3
        
        # Create card frame
        card = ttk.Frame(self.files_frame, borderwidth=1, relief="raised", padding=5)
        card.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
        
        # File type icon or thumbnail
        if file_metadata.file_type == "image":
            # Create thumbnail in background thread to avoid UI freezing
            threading.Thread(target=self.load_thumbnail, args=(card, file_metadata), daemon=True).start()
        else:
            # Display icon based on file type
            icon_text = "📄" if file_metadata.file_type == "document" else "🗂️"
            icon = ttk.Label(card, text=icon_text, font=("Arial", 36))
            icon.grid(row=0, column=0, pady=5)
            
        # File name (truncated if too long)
        name = file_metadata.file_name
        if len(name) > 20:
            name = name[:17] + "..."
        name_label = ttk.Label(card, text=name)
        name_label.grid(row=1, column=0, sticky="w", pady=(0, 5))
        
        # Description (if any)
        if file_metadata.description:
            desc = file_metadata.description
            if len(desc) > 30:
                desc = desc[:27] + "..."
            desc_label = ttk.Label(card, text=desc, foreground="gray")
            desc_label.grid(row=2, column=0, sticky="w")
            
        # Actions frame
        actions = ttk.Frame(card)
        actions.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        
        # View button
        view_btn = ttk.Button(actions, text="View", 
                             command=lambda m=file_metadata: self.view_file(m))
        view_btn.grid(row=0, column=0, padx=2)
        
        # Share button
        share_btn = ttk.Button(actions, text="Share",
                              command=lambda m=file_metadata: self.share_with_anima(m))
        share_btn.grid(row=0, column=1, padx=2)
        
        # Delete button
        delete_btn = ttk.Button(actions, text="Delete",
                               command=lambda m=file_metadata: self.delete_file(m))
        delete_btn.grid(row=0, column=2, padx=2)
        
    def load_thumbnail(self, card, file_metadata):
        """Load and display thumbnail for image files"""
        thumbnail = file_metadata.create_thumbnail()
        if thumbnail:
            # This needs to run in the main thread
            self.root.after(0, lambda: self.set_thumbnail(card, thumbnail))
            
    def set_thumbnail(self, card, thumbnail):
        """Set the thumbnail in the UI (must be called from main thread)"""
        for widget in card.winfo_children():
            if isinstance(widget, ttk.Label) and widget.grid_info()["row"] == 0:
                widget.destroy()
                
        thumb_label = ttk.Label(card, image=thumbnail)
        thumb_label.image = thumbnail  # Keep a reference to prevent garbage collection
        thumb_label.grid(row=0, column=0, pady=5)
        
    def view_file(self, file_metadata):
        """Open the file with system default application"""
        if os.path.exists(file_metadata.stored_path):
            try:
                os.startfile(file_metadata.stored_path)
            except AttributeError:
                # os.startfile is only available on Windows, use alternatives for other OS
                import subprocess
                if sys.platform == "darwin":  # macOS
                    subprocess.call(["open", file_metadata.stored_path])
                else:  # Linux/Unix
                    subprocess.call(["xdg-open", file_metadata.stored_path])
        else:
            messagebox.showerror("Error", "File not found.")
            
    def share_with_anima(self, file_metadata):
        """Share the file with Anima AI"""
        # Try to import Anima's memory system
        try:
            from memory import memory_manager
            
            # Create memory entry for this file
            memory_entry = {
                "type": "shared_file",
                "file_path": file_metadata.stored_path,
                "file_name": file_metadata.file_name,
                "file_type": file_metadata.file_type,
                "description": file_metadata.description,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to Anima's memory
            memory_manager.add_memory(memory_entry)
            
            messagebox.showinfo("Success", f"File '{file_metadata.file_name}' shared with Anima.")
            
        except ImportError:
            # If memory system isn't available, provide file path instead
            file_info = (f"File: {file_metadata.file_name}\n"
                         f"Path: {file_metadata.stored_path}\n"
                         f"Type: {file_metadata.file_type}\n"
                         f"Description: {file_metadata.description}")
                         
            # Create a dialog with copyable text
            dialog = tk.Toplevel(self.root)
            dialog.title("File Shared")
            dialog.geometry("500x300")
            dialog.transient(self.root)
            
            ttk.Label(dialog, text="File ready to share with Anima:", padding=10).pack(fill="x")
            
            # Text area with file info
            text_area = ScrolledText(dialog, height=10, width=60)
            text_area.pack(padx=10, pady=5, fill="both", expand=True)
            text_area.insert("1.0", file_info)
            text_area.configure(state="disabled")
            
            ttk.Label(dialog, text="Copy this information to share with Anima.", padding=10).pack(fill="x")
            
            def copy_to_clipboard():
                self.root.clipboard_clear()
                self.root.clipboard_append(file_info)
                copy_btn.configure(text="Copied!")
                self.root.after(1500, lambda: copy_btn.configure(text="Copy to Clipboard"))
                
            def close_dialog():
                dialog.destroy()
                
            button_frame = ttk.Frame(dialog)
            button_frame.pack(padx=10, pady=10, fill="x")
            
            copy_btn = ttk.Button(button_frame, text="Copy to Clipboard", command=copy_to_clipboard)
            copy_btn.pack(side="left", padx=5)
            
            close_btn = ttk.Button(button_frame, text="Close", command=close_dialog)
            close_btn.pack(side="right", padx=5)
            
    def delete_file(self, file_metadata):
        """Delete a file from the system"""
        if messagebox.askyesno("Confirm Delete", 
                              f"Are you sure you want to delete '{file_metadata.file_name}'?"):
            if self.file_manager.remove_file(file_metadata):
                self.status_var.set(f"Deleted '{file_metadata.file_name}'")
                self.update_ui()
            else:
                messagebox.showerror("Error", "Failed to delete file.")


def launch_file_sharing_ui():
    """Launch the file sharing UI"""
    root = tk.Tk()
    app = FileSharingUI(root)
    
    # Set app icon if available
    try:
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "anima_icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except:
        pass  # Icon loading is not critical
        
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    launch_file_sharing_ui()
