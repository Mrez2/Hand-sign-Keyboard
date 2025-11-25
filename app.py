import av
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit as st


# ----------------------------------
# SESSION STATE (MAIN THREAD ONLY)
# ----------------------------------
if "typed_message" not in st.session_state:
    st.session_state.typed_message = ""

if "last_added_letter" not in st.session_state:
    st.session_state.last_added_letter = ""

if "stable_count" not in st.session_state:
    st.session_state.stable_count = 0



# ----------------------------------
# MODEL (26 classes)
# ----------------------------------
class HandSignModel(nn.Module):
    def __init__(self, input_size=63, num_classes=26):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


model = HandSignModel()
model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))
model.eval()


# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils



# ----------------------------------
# SAFE VIDEO PROCESSOR (NO session_state HERE)
# ----------------------------------
class HandSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_letter = ""
        self.safe_message_copy = ""  # Safe local message copy

    def update_message(self, msg):
        """Called by main thread safely."""
        self.safe_message_copy = msg

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        predicted_letter = ""

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

                coords = []
                for lm in hand.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                if len(coords) == 63:
                    x = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        pred = model(x)
                        idx = torch.argmax(pred, dim=1).item()
                        predicted_letter = chr(idx + 65)

        self.last_letter = predicted_letter

        # SAFE drawing (using only local variable)
        cv2.putText(img, f"Sign: {self.last_letter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(img, f"Message: {self.safe_message_copy}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")



# ----------------------------------
# START WEBCAM
# ----------------------------------
ctx = webrtc_streamer(
    key="sign-keyboard",
    video_processor_factory=HandSignProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)



# ----------------------------------
# MAIN THREAD CONTROLS
# ----------------------------------
st.markdown("### ✍ Enhanced Typing System")

if ctx.video_processor:

    current_letter = ctx.video_processor.last_letter
    st.write("Detected Letter:", current_letter)

    # -----------------------------
    # AUTO-STABLE ADDING
    # -----------------------------
    if current_letter:
        if current_letter == st.session_state.last_added_letter:
            st.session_state.stable_count += 1
        else:
            st.session_state.stable_count = 0

        if st.session_state.stable_count > 15:
            st.session_state.typed_message += current_letter
            st.session_state.stable_count = 0
            st.session_state.last_added_letter = ""
        else:
            st.session_state.last_added_letter = current_letter

    # -----------------------------
    # AUTO-SPACE RULE
    # -----------------------------
    if current_letter == "A" and st.session_state.stable_count > 20:
        st.session_state.typed_message += " "
        st.session_state.stable_count = 0

    # -----------------------------
    # MANUAL BUTTONS
    # -----------------------------
    if st.button("➕ Add Letter Now"):
        st.session_state.typed_message += current_letter

    if st.button("⬅ Backspace"):
        st.session_state.typed_message = st.session_state.typed_message[:-1]

    if st.button("🧹 Clear"):
        st.session_state.typed_message = ""

    # Update safe message inside webcam processor
    ctx.video_processor.update_message(st.session_state.typed_message)



# ----------------------------------
# SHOW FINAL MESSAGE
# ----------------------------------
st.markdown("### 📝 Final Message:")
st.write(st.session_state.typed_message)
