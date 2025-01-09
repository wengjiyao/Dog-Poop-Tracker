from collections import deque
from PIL import Image
from dog_pooping_detector.classifier import DogPoopClassifier

class PoopingDetector:
    """
    Detects if a dog is currently pooping based on consecutive classifications.
    """

    def __init__(self, N: int = 4):
        """
        Args:
            consecutive_threshold (int): The number of consecutive frames
                that must be classified as 'pooping' or 'not pooping' to
                switch the state.
        """
        self.N = N
        self.queue = deque(maxlen=self.N)
        self.current_state = False  # Tracks whether the dog is currently pooping or not
        self.cooldown = 0  # Tracks a cooldown period before re-checking

    def check_if_pooping(self, dog_roi) -> bool:
        """
        Given a dog's region of interest (ROI) as a NumPy array,
        determine if the dog is pooping.

        Args:
            dog_roi (np.ndarray): A cropped image of the dog (RGB).

        Returns:
            bool: True if dog is pooping, False otherwise.
        """
        # If in cooldown, decrement and return current state
        if self.cooldown > 0:
            self.cooldown -= 1
            print("Detector in cooldown.")
            return self.current_state

        # Retrieve the singleton classifier (adjust path to your model if needed)
        classifier = DogPoopClassifier(model_path="models/best_mobilenetv2.pth")

        # Convert the NumPy array to a Pillow Image
        dog_image = Image.fromarray(dog_roi)

        # Run classification
        is_pooping = classifier.is_dog_pooping(dog_image)
        self.queue.append(is_pooping)

        # Count occurrences of True/False in the recent queue
        true_count = self.queue.count(True)
        false_count = self.queue.count(False)

        print(f"true_count={true_count}")
        print(f"false_count={false_count}")

        # If all frames in the queue are "poop" and state was previously False => switch to True + cooldown
        if true_count == self.N and not self.current_state:
            self.current_state = True
            self.cooldown = 25 * 5  # ~5 seconds if your frame rate is ~25 FPS
            print("Dog is pooping.")
            return True
        # If all frames in the queue are "not poop" and state was previously True => switch to False
        elif false_count == self.N and self.current_state:
            self.current_state = False
            print("Dog is NOT pooping.")
            return False

        # Otherwise maintain the current state
        if self.current_state:
            print("Dog is pooping.")
        else:
            print("Dog is NOT pooping.")
        return self.current_state
