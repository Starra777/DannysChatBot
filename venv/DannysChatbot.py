import cv2
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import tkinter as tk
from tkinter import ttk
import threading
import face_recognition
import os

# Initialize speech recognition, text-to-speech engine, and Google Generative AI API
r = sr.Recognizer()
engine = pyttsx3.init()

genai.configure(api_key="AIzaSyBrFbYRDjV0mqSu_ab6DWHGuacfBmrUo_A")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Function to detect faces using OpenCV
def detect_faces():
    known_face_encodings = load_registered_faces()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            if True in matches:
                cap.release()
                cv2.destroyAllWindows()
                process_input()
                return

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load known faces from labelled_images directory
def load_registered_faces():
    known_face_encodings = []
    try:
        image_paths = [os.path.join('labelled_images', f) for f in os.listdir('labelled_images')]
        for image_path in image_paths:
            img = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(img)
            if encoding:
                known_face_encodings.append(encoding[0])
    except Exception as e:
        update_text_area(f"Error loading images: {e}", 'error')
    return known_face_encodings

# Function to generate a response from Google Generative AI
def generate_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text if response else "Sorry, there was an error generating the response."
    except Exception as e:
        return f"Sorry, there was an error: {e}"

# Function to handle voice input and respond
def process_input():
    try:
        with sr.Microphone() as source:
            update_text_area("Chatbot: Listening...\n", 'chatbot')
            audio = r.listen(source)
            prompt = r.recognize_google(audio)

            # Display prompt and generate response
            update_text_area(f"You: {prompt}\n", 'user')
            response = generate_response(prompt)
            update_text_area(f"Chatbot: {response}\n", 'chatbot')

            engine.say(response)
            engine.runAndWait()

            # Show the "Prompt Again" button
            prompt_again_button.pack(pady=10)
            face_detect_button.pack_forget()

    except sr.UnknownValueError:
        update_text_area("Chatbot: Could not understand audio. Please try again.\n", 'error')
    except sr.RequestError as e:
        update_text_area(f"Chatbot: Could not request results; {e}\n", 'error')

# Function to allow prompting again
def prompt_again():
    prompt_again_button.pack_forget()
    process_input()

# Update the text area in the GUI from different threads
def update_text_area(text, tag=None):
    root.after(0, lambda: text_area.insert(tk.END, text, tag))
    root.after(0, lambda: text_area.see(tk.END))

# Create GUI window
root = tk.Tk()
root.title("Interactive Chatbot with Face Detection")
root.geometry("700x700")
root.config(bg="#2c3e50")  # Set a dark background color

# Configure a modern style for buttons and widgets
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 14), padding=10, background="#1abc9c", foreground="white")
style.map("TButton", background=[('active', '#16a085')])

# Create a scrollable text area for displaying the conversation
text_frame = tk.Frame(root)
text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

scrollbar = tk.Scrollbar(text_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

text_area = tk.Text(text_frame, wrap=tk.WORD, bg="#34495e", fg="#ecf0f1", font=("Helvetica", 12), yscrollcommand=scrollbar.set)
text_area.pack(fill=tk.BOTH, expand=True)
scrollbar.config(command=text_area.yview)

# Style text for user, chatbot, and errors
text_area.tag_configure('user', foreground="#2ecc71", font=("Helvetica", 12, "bold"))
text_area.tag_configure('chatbot', foreground="#3498db", font=("Helvetica", 12, "bold"))
text_area.tag_configure('error', foreground="#e74c3c", font=("Helvetica", 12, "italic"))

# Add buttons for face detection and prompting again
button_frame = tk.Frame(root, bg="#2c3e50")
button_frame.pack(pady=20)

face_detect_button = ttk.Button(button_frame, text="Start Face Detection", command=lambda: threading.Thread(target=detect_faces).start())
face_detect_button.pack(pady=10)

prompt_again_button = ttk.Button(button_frame, text="Prompt Again", command=prompt_again)

# Start the GUI main loop
root.mainloop()
