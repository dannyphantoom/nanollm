import tkinter as tk
from tkinter import ttk, scrolledtext
import json
from datetime import datetime
from typing import Callable
import threading
from ttkthemes import ThemedTk

class ModernChatWindow:
    def __init__(self, on_send: Callable[[str], None]):
        self.root = ThemedTk(theme="arc")  # Modern theme
        self.root.title("NanoLLM Chat")
        self.root.geometry("800x600")
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("Send.TButton", 
                           padding=10, 
                           font=('Helvetica', 10))
        
        # Chat history (messages area)
        self.chat_frame = ttk.Frame(self.root)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.chat_area = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            font=('Helvetica', 10),
            bg='#ffffff',
            fg='#000000'
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)
        
        # Input area
        self.input_frame = ttk.Frame(self.root)
        self.input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.input_field = ttk.Entry(
            self.input_frame,
            font=('Helvetica', 10)
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.send_button = ttk.Button(
            self.input_frame,
            text="Send",
            style="Send.TButton",
            command=self._on_send
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Bind enter key to send
        self.input_field.bind("<Return>", lambda e: self._on_send())
        
        # Store callback
        self.on_send = on_send
        
        # Loading indicator
        self.loading = False
        
    def _on_send(self):
        message = self.input_field.get().strip()
        if message:
            # Clear input field
            self.input_field.delete(0, tk.END)
            
            # Add user message to chat
            self.add_message("You", message)
            
            # Disable input while processing
            self.input_field.config(state=tk.DISABLED)
            self.send_button.config(state=tk.DISABLED)
            
            # Start processing in separate thread
            threading.Thread(target=self._process_message, args=(message,)).start()
    
    def _process_message(self, message: str):
        try:
            # Call the callback
            self.on_send(message)
        finally:
            # Re-enable input
            self.root.after(0, self._enable_input)
    
    def _enable_input(self):
        self.input_field.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.input_field.focus()
    
    def add_message(self, sender: str, message: str):
        self.chat_area.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        # Format message
        if sender == "You":
            self.chat_area.insert(tk.END, f"\n{timestamp} {sender}:\n", "user")
        else:
            self.chat_area.insert(tk.END, f"\n{timestamp} NanoLLM:\n", "assistant")
        
        self.chat_area.insert(tk.END, f"{message}\n")
        
        # Auto-scroll to bottom
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)
    
    def start(self):
        self.root.mainloop()
    
    def stop(self):
        self.root.quit() 