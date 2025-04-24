import torch
from torchvision import transforms
from PIL import Image
from train_dogs_and_cats import ConvNet

# Function to predict a single image


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
