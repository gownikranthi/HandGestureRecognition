<<<<<<< HEAD
#Real-Time Static Hand Gesture Recognition

### 1. Technology Justification

This project utilizes **OpenCV** and **MediaPipe** to perform real-time hand gesture recognition.

-   **OpenCV** is a robust computer vision library used for handling the webcam video stream, flipping the image for a mirrored view, and displaying the output with overlays. Its simple API for video I/O makes it a perfect choice for this task.
-   **MediaPipe Hands** is a highly efficient machine learning solution developed by Google. It was chosen for its ability to accurately detect and track 21 key hand landmarks in real-time. This saves us from having to train a custom model and provides precise landmark data crucial for our gesture recognition logic.

This combination allows for a performant and reliable solution that meets the real-time requirements of the assignment.

### 2. Gesture Logic Explanation

The gesture recognition logic is a **rules-based algorithm** that analyzes the 21 hand landmarks provided by MediaPipe. We determine the state of each finger (extended or curled) by comparing the vertical position (y-coordinate) of its fingertip landmark against its corresponding knuckle landmark.

-   **Open Palm:** All four fingers (index, middle, ring, pinky) are extended. The thumb is also extended.
-   **Fist:** All four fingers are curled, and the thumb is also curled.
-   **Peace Sign:** The index and middle fingers are extended, while the ring and pinky fingers are curled.
-   **Thumbs Up:** The thumb is extended, while all other fingers are curled.

This approach is simple, effective, and robust to different hand sizes and positions in the camera frame.

### 3. Setup and Execution Instructions

Follow these steps to set up the environment and run the application:

1.  Clone this repository.
2.  Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the main Python script:
    ```bash
    python main.py
    ```
4.  Press 'q' to exit the application window.

### 4. Demonstration

A short video demonstration of the application in action.

