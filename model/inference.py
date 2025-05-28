import torch
from torchvision import transforms
from PIL import Image
from model import PlantDiseaseModel

class DiseasePredictor:
    def __init__(self, model_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PlantDiseaseModel(num_classes=len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'disease': self.class_names[predicted_class],
            'confidence': confidence
        }