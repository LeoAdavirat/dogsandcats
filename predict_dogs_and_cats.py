import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Function to predict a single image


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def predict_image(img_path, model, device):
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img).unsqueeze(0).to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            output = model(img).squeeze()
            prediction = output > 0.5
        print("Cho phép chó vào." if prediction else "Không được phép vào.")
    except Exception as e:
        print(f"Error processing image: {e}")


# Main execution
if __name__ == "__main__":
    from pathlib import Path
    import random

    def list_all(path: str | Path, include_dirs: bool = False):
        path = Path(path)
        if include_dirs:
            return [p for p in path.rglob("*")]
        else:
            return [p for p in path.rglob("*") if p.is_file()]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(
        'dogs_vs_cats_pytorch_model.pth', map_location=device))

    folder_path = "kagglecatsanddogs_3367a/PetImages/"

    files = list_all(folder_path)

    image_path = random.choice(files).as_posix()

    print(image_path)
    predict_image(image_path, model, device)
