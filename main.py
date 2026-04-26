import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame, Entry
from PIL import Image, ImageTk
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ================= MODEL =================
model = pickle.load(open("model.p", "rb"))["model"]

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=1
)

# ================= TEXT TO SPEECH =================
engine = pyttsx3.init()
engine.setProperty("rate", 120)
engine.setProperty("volume", 1.0)

#Female voice selection (final enslashing fix)
voices = engine.getProperty('voices')
for v in voices:
    if "zira" in v.name.lower() or "female" in v.name.lower():
        engine.setProperty('voice', v.id)
        break

tts_lock = threading.Lock()

def speak_text(text):
    if not text.strip():
        return
    def run():
        with tts_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

def format_for_speech(word):
    return " ".join(list(word))   

def is_number(ch):
    return ch.isdigit()

# ================= LABELS =================
labels_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',
    20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',
    26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7',34:'8',35:'9',
    36:' ',37:'.'
}
EXPECTED_FEATURES = 63

# ================= STATE =================
buffer = []
word_buffer = ""
sentence = ""
last_time = time.time()
delay = 1.5
no_hand_timer = None

# ================= UI (SAME AS ORIGINAL) =================
root = tk.Tk()
root.title("Sign Language to Speech Conversion")
root.geometry("1300x650")
root.configure(bg="#2c2f33")
root.resizable(False, False)

current_alphabet = StringVar(value="N/A")
current_word = StringVar(value="N/A")
current_sentence = StringVar(value="N/A")
typed_word = StringVar()

Label(
    root,
    text="Sign Language to Speech Conversion",
    font=("Arial", 28, "bold"),
    fg="#ffffff",
    bg="#2c2f33"
).grid(row=0, column=0, columnspan=2, pady=10)

# ---------- LEFT : CAMERA ----------
video_frame = Frame(root, bg="#2c2f33", bd=5, relief="solid", width=500, height=400)
video_frame.grid(row=1, column=0, rowspan=3, padx=20, pady=20)
video_frame.grid_propagate(False)

video_label = Label(video_frame)
video_label.pack(expand=True)

# ---------- RIGHT : INFO ----------
content_frame = Frame(root, bg="#2c2f33")
content_frame.grid(row=1, column=1, sticky="n", padx=(20,40), pady=(40,10))

Label(content_frame, text="Current Alphabet:", font=("Arial",20),
      fg="white", bg="#2c2f33").pack(anchor="w")
Label(content_frame, textvariable=current_alphabet,
      font=("Arial",24,"bold"), fg="#1abc9c", bg="#2c2f33").pack()

Label(content_frame, text="Current Word:", font=("Arial",20),
      fg="white", bg="#2c2f33").pack(anchor="w", pady=(20,5))
Label(content_frame, textvariable=current_word,
      font=("Arial",20), fg="#f39c12", bg="#2c2f33").pack()

Label(content_frame, text="Current Sentence:", font=("Arial",20),
      fg="white", bg="#2c2f33").pack(anchor="w", pady=(20,5))
Label(content_frame, textvariable=current_sentence,
      font=("Arial",20), fg="#9b59b6", bg="#2c2f33", wraplength=500).pack()

# ---------- TYPE WORD ----------
Label(content_frame, text="Type Word:", font=("Arial",18),
      fg="white", bg="#2c2f33").pack(anchor="w", pady=(25,5))

Entry(content_frame, textvariable=typed_word,
      font=("Arial",18), width=25).pack()

# ---------- BUTTONS ----------
button_frame = Frame(root, bg="#2c2f33")
button_frame.grid(row=2, column=1, pady=(20,20), padx=(10,20), sticky="n")

def reset_sentence():
    global word_buffer, sentence
    word_buffer = ""
    sentence = ""
    typed_word.set("")
    current_word.set("N/A")
    current_sentence.set("N/A")
    current_alphabet.set("N/A")

Button(button_frame, text="Reset Sentence",
       font=("Arial",16),
       bg="#e74c3c", fg="white",
       height=2, width=14,
       command=reset_sentence).grid(row=0, column=0, padx=10)

Button(button_frame, text="Speak Typed Word",
       font=("Arial",16),
       bg="#27ae60", fg="white",
       height=2, width=18,
       command=lambda: speak_text(typed_word.get())
).grid(row=0, column=1, padx=10)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

def process_frame():
    global word_buffer, sentence, last_time, no_hand_timer

    ret, frame = cap.read()
    if not ret:
        root.after(30, process_frame)
        return

    frame = cv2.flip(frame, 1)

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # -------- NO HAND DETECTED --------
    if not results.multi_hand_landmarks:

        if word_buffer != "":
            if no_hand_timer is None:
                no_hand_timer = time.time()

            elif time.time() - no_hand_timer > 2:

                speak_text(format_for_speech(word_buffer))

                if sentence == "":
                    sentence = word_buffer
                else:
                    sentence = sentence + " " + word_buffer

                current_sentence.set(sentence)

                word_buffer = ""
                current_word.set("N/A")

                no_hand_timer = None

    # -------- HAND DETECTED --------
    else:

        no_hand_timer = None

        for hand in results.multi_hand_landmarks:

            xs, ys, data = [], [], []

            for lm in hand.landmark:
                xs.append(lm.x)
                ys.append(lm.y)

            min_x = min(xs)
            min_y = min(ys)

            for lm in hand.landmark:
                data.extend([
                    lm.x - min_x,
                    lm.y - min_y,
                    lm.z
                ])

            if len(data) != 63:
                continue

            proba = model.predict_proba([np.array(data)])[0]
            prediction = np.argmax(proba)
            confidence = proba[prediction]

            char = labels_dict[int(prediction)]

            buffer.append(char)
            if len(buffer) > 30:
                buffer.pop(0)

            if buffer.count(char) > 25 and time.time() - last_time > delay:

                last_time = time.time()

                if is_number(char):
                    continue

                current_alphabet.set(f"{char} ({confidence*100:.1f}%)")

                word_buffer += char
                current_word.set(word_buffer)

            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    video_label.configure(image=img)
    video_label.image = img

    root.after(30, process_frame)

process_frame()
root.mainloop()

