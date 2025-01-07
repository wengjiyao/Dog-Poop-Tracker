from collections import deque
from PIL import Image
import time
from dog_poop_classifier import DogPoopClassifier

class PoopingDetector:
    def __init__(self, N):
        self.N = N  # The threshold for consecutive detections
        self.queue = deque(maxlen=N)
        self.current_state = False  # Tracks whether the dog is currently pooping or not
        self.cooldown = 0  # Tracks cooldown period

    def check_if_pooping(self, dog_roi):
        # Check if in cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            print("Detector in cooldown.")
            return self.current_state

        # Retrieve the singleton classifier
        classifier = DogPoopClassifier("models/best_mobilenetv2.9466.pth")

        # Convert the NumPy array to a Pillow Image
        dog_image = Image.fromarray(dog_roi)

        # Classify
        is_pooping = classifier.is_dog_pooping(dog_image)
        self.queue.append(is_pooping)

        # Count occurrences of True (pooping) in the queue
        true_count = self.queue.count(True)
        false_count = self.queue.count(False)

        if true_count == self.N and not self.current_state:
            self.current_state = True
            self.cooldown = 25 * 5  # Set cooldown to 5 second (assuming 25 fps)
            print("Dog is pooping.")
            return True
        elif false_count == self.N and self.current_state:
            self.current_state = False
            print("Dog is NOT pooping.")
            return False

        # Maintain the current state if the threshold is not met
        print("Dog is pooping." if self.current_state else "Dog is NOT pooping.")
        return self.current_state

# Example usage
# detector = PoopingDetector(N=3)
# while video_streaming:
#     dog_roi = get_dog_roi()  # Function to retrieve the region of interest
#     detector.check_if_pooping(dog_roi)
