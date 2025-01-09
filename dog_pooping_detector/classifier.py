import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class DogPoopClassifier:
    """
    Singleton classifier for detecting 'poop' vs. 'notpoop' in images.
    """

    _instance = None

    def __new__(cls, model_path="../models/best_mobilenetv2.pth", device=None):
        if cls._instance is None:
            cls._instance = super(DogPoopClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path="../models/best_mobilenetv2.pth", device=None):
        # Ensure that we only initialize once in a singleton pattern
        if getattr(self, "_initialized", False):
            return

        self._initialized = True

        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.num_classes = 2  # e.g. 0=not poop, 1=poop

        # Load MobileNetV2 base
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Adjust final layer to match our 2-class problem
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, self.num_classes)

        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define image transforms
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def is_dog_pooping(self, cropped_image: Image.Image) -> bool:
        """
        Given a single cropped dog image, predict whether the dog is pooping.
        """
        # Preprocess
        input_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Predicted class
        _, predicted_idx = torch.max(outputs, 1)
        return (predicted_idx.item() == 1)
