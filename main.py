import cv2
import mediapipe as mp

# Initialize MediaPipe and OpenCV components
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    # Flip the image horizontally for a more intuitive view
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)
    
    # Initialize the gesture text
    gesture = "No Hand Detected"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Use integer indices to access landmarks
            # For reference: 
            
            # Landmark indices for finger tips and knuckles
            finger_tip_indices = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
            knuckle_indices = [5, 9, 13, 17]     # Index, Middle, Ring, Pinky
            
            thumb_tip_y = hand_landmarks.landmark[4].y
            thumb_ipp_y = hand_landmarks.landmark[3].y
            
            # Check if a finger is extended (tip y-coord is less than knuckle y-coord)
            fingers_extended = [
                hand_landmarks.landmark[tip].y < hand_landmarks.landmark[knuckle].y 
                for tip, knuckle in zip(finger_tip_indices, knuckle_indices)
            ]

            # Check for each gesture based on finger extension
            if all(fingers_extended) and (thumb_tip_y < thumb_ipp_y):
                gesture = "Open Palm"
            elif not any(fingers_extended) and (thumb_tip_y > thumb_ipp_y):
                gesture = "Fist"
            elif fingers_extended[0] and fingers_extended[1] and not fingers_extended[2] and not fingers_extended[3]:
                gesture = "Peace Sign"
            elif not any(fingers_extended) and (thumb_tip_y < thumb_ipp_y):
                gesture = "Thumbs Up"
            else:
                gesture = "Unknown Gesture"

            # Draw the hand landmarks on the image
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    # Display the result
    cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the image
    cv2.imshow('Hand Gesture Recognition', image)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()