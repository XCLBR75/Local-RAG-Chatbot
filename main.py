import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from agent_script import agent, llm, close_resources

# === Main Window ===
root = tk.Tk()
root.title("RAG Chatbot")
root.geometry("600x700")
root.resizable(True, True)

# === Shutdown handler ===
def on_exit():
    print("Closing app and resources...")
    try:
        close_resources()
    except Exception as e:
        print(f"Error during shutdown: {e}")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_exit)

# === Load profile pics ===
user_img = Image.open("images/user.jpg").resize((40, 40), Image.LANCZOS)
chatbot_img = Image.open("images/chatbot.jpg").resize((40, 40), Image.LANCZOS)
user_photo = ImageTk.PhotoImage(user_img)
chatbot_photo = ImageTk.PhotoImage(chatbot_img)

# === Configure layout ===
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# === Main Frame ===
main_frame = ttk.Frame(root, padding=10)
main_frame.grid(row=0, column=0, sticky="nsew")
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)

# === Scrollable Chat Display ===
scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL)
chat_display = tk.Text(
    main_frame,
    wrap=tk.WORD,
    state="disabled",
    bg="#f5f5f5",
    fg="#000000",
    font=("Arial", 12),
    yscrollcommand=scrollbar.set,
)
chat_display.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
scrollbar.config(command=chat_display.yview)
scrollbar.grid(row=0, column=1, sticky="ns")

# === Input Frame ===
input_frame = ttk.Frame(main_frame)
input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
input_frame.columnconfigure(0, weight=1)

entry = ttk.Entry(input_frame, font=("Arial", 12))
entry.grid(row=0, column=0, sticky="ew", padx=5)

send_button = ttk.Button(input_frame, text="Send", command=lambda: send_query())
send_button.grid(row=0, column=1, padx=5)

# === Bind Enter Key ===
entry.bind("<Return>", lambda event: send_query())

# === Add Message Bubble to Text ===
def add_bubble(text, sender="user"):
    chat_display.config(state="normal")

    # Force full-width outer frame 
    outer_frame = tk.Frame(chat_display, width=chat_display.winfo_width())
    outer_frame.pack(fill="x", pady=5)

    # Bubble frame (message + avatar)
    bubble_frame = ttk.Frame(outer_frame, padding=5)

    # Images
    avatar_img = user_photo if sender == "user" else chatbot_photo
    avatar_label = ttk.Label(bubble_frame, image=avatar_img)
    avatar_label.image = avatar_img  

    # Text Bubble
    text_label = tk.Label(
        bubble_frame,
        text=text,
        wraplength=400,
        justify="left",
        bg="#DCF8C6" if sender == "user" else "#E6E6E6",
        fg="#000000",
        font=("Arial", 12),
        padx=10,
        pady=5,
        relief=tk.RAISED,
    )

    if sender == "user":
        text_label.pack(side="right")
        avatar_label.pack(side="right", padx=5)
        bubble_frame.pack(side="right", anchor="e")
    else:
        avatar_label.pack(side="left", padx=5)
        text_label.pack(side="left")
        bubble_frame.pack(side="left", anchor="w")

    chat_display.window_create(tk.END, window=outer_frame)
    chat_display.insert(tk.END, "\n")
    chat_display.config(state="disabled")
    chat_display.see(tk.END)

    return bubble_frame


# === Update Bubble Text (replaces placeholder) ===
def update_bubble(bubble_frame, new_text):
    for widget in bubble_frame.winfo_children():
        if isinstance(widget, tk.Label) and widget.cget("relief") == tk.RAISED:
            widget.config(text=new_text)
            break
    chat_display.update_idletasks()
    chat_display.see(tk.END)

# === Run Agent with Retry on various errors ===
def run_agent(query, bubble_frame):
    max_retries = 1
    attempt = 0
    response = ""

    while attempt < max_retries:
        try:
            response = agent.run(query)
            if response and not response.strip().lower().startswith("error"):
                break
        except Exception as e:
            response = f"Error: {str(e)}"
        attempt += 1

    # fallback to direct LLM call
    if response.strip().lower().startswith("error") or response.strip().lower().startswith("could not parse llm output"):
        try:
            raw_response = llm.invoke(query)
            response = raw_response if isinstance(raw_response, str) else str(raw_response)
        except Exception as fallback_error:
            response = f"Total failure: {fallback_error}"

    update_bubble(bubble_frame, response)

# === Send Query ===
def send_query():
    query = entry.get().strip()
    if not query:
        return
    entry.delete(0, tk.END)
    user_bubble = add_bubble(query, sender="user")
    placeholder = add_bubble("...", sender="bot")
    threading.Thread(target=run_agent, args=(query, placeholder)).start()


# === Launch App ===
root.mainloop()



