import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class DogPoopClassifier:
    """
    Singleton classifier for detecting 'poop' vs. 'notpoop' in images.

    Usage:
        classifier1 = DogPoopClassifier(model_path="best_mobilenetv2.pth")
        classifier2 = DogPoopClassifier(model_path="some_other_model.pth")

        # Both classifier1 and classifier2 will reference the same object
        # (hence only the first .__init__ loads the model).
    """
    _instance = None

    def __new__(cls, model_path="best_mobilenetv2.pth", device=None):
        # If an instance doesn’t already exist, create it
        if cls._instance is None:
            cls._instance = super(DogPoopClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path="best_mobilenetv2.pth", device=None):
        """
        Initialize the singleton classifier by:
         - Loading MobileNetV2 (2-class final layer).
         - Loading the trained weights from model_path.
         - Defining image transforms for preprocessing.
        """
        # The following check ensures that initialization code
        # only runs once in a singleton pattern.
        if getattr(self, "_initialized", False):
            return  # already initialized, skip re-initializing

        # Mark as initialized
        self._initialized = True

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.num_classes = 2  # 2 classes: notpoop (0), poop (1)

        # Load a MobileNetV2 model structure
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Replace the final classifier layer to match the number of classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, self.num_classes)

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing (ImageNet normalization)
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
        Returns:
            True  if the classifier predicts 'poop' (class index 1)
            False otherwise
        """
        # Apply the transform
        input_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Determine predicted class (0=notpoop, 1=poop)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_idx = predicted_idx.item()
        return (predicted_idx == 1)
