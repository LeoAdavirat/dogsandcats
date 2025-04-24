import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Step 1: Prepare the data
data_dir = 'kagglecatsanddogs_3367a/PetImages'
train_dir = 'train'
test_dir = 'test'

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# Function to validate images
def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Verify image integrity
        img.close()
        img = Image.open(file_path)
        img.transpose(Image.FLIP_LEFT_RIGHT)  # Test image processing
        img.close()
        return True
    except Exception:
        return False


# Collect and split data
for category in ['Cat', 'Dog']:
    category_dir = os.path.join(data_dir, category)
    images = [
        os.path.join(category_dir, img) for img in os.listdir(category_dir)
        if os.path.isfile(os.path.join(category_dir, img)) and is_valid_image(os.path.join(category_dir, img))
    ]
    print(f"Valid {category} images: {len(images)}")
    train_images, test_images = train_test_split(
        images, test_size=0.2, random_state=42)

    # Copy images to respective directories
    for img in train_images:
        dest = os.path.join(train_dir, category)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(img, dest)
    for img in test_images:
        dest = os.path.join(test_dir, category)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(img, dest)


# Step 2: Define custom dataset
class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Cat', 'Dog']
        self.image_paths = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if is_valid_image(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


# Step 3: Define the CNN model
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


# Step 4: Data transforms
train_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Step 5: Create datasets and dataloaders
train_dataset = CatsDogsDataset(train_dir, transform=train_transforms)
test_dataset = CatsDogsDataset(test_dir, transform=test_transforms)

# Step 6: Hyperparameter tuning
learning_rates = [1e-3, 1e-4, 0.01, 1e-5]
batch_sizes = [64, 32]
best_val_acc = 0
best_model_state = None
best_params = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for lr in learning_rates:
    for bs in batch_sizes:
        print(f"Training with lr={lr}, batch_size={bs}")
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        model = ConvNet().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(10):  # Reduced epochs for tuning
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).float()
                optimizer.zero_grad()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total
            print(
                f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_params = {'lr': lr, 'batch_size': bs}

print(
    f"Best parameters: {best_params}, Best Validation Accuracy: {best_val_acc:.4f}")

# Step 7: Train the best model for more epochs
model = ConvNet().to(device)
model.load_state_dict(best_model_state)
train_loader = DataLoader(
    train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=best_params['batch_size'], shuffle=False)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

# Full training
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    print(
        f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

# Step 8: Evaluate the model
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs = model(images).squeeze()
        predicted = (outputs > 0.5).float()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Report metrics
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Test accuracy
test_acc = np.mean(y_true == y_pred)
print(f"Test Accuracy: {test_acc:.4f}")

# Step 9: Function to predict new images


def predict_image(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = test_transforms(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(img).squeeze()
            prediction = output > 0.5
        if prediction:
            print("Cho phép chó vào.")
        else:
            print("Không được phép vào.")
    except Exception as e:
        print(f"Error processing image: {e}")

# Example usage (uncomment and replace with actual path)
# predict_image('path/to/new/image.jpg')


# Save the model
torch.save(model.state_dict(), 'dogs_vs_cats_pytorch_model.pth')

if __name__ == "__main__":
    print("Model training, tuning, and evaluation completed.")
    # Uncomment the line below and provide a path to test the prediction function
    # predict_image('path/to/new/image.jpg')
