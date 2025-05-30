import os
import torch
import torch.nn as nn
from flask import Flask, request, render_template_string
from torchvision import transforms
from PIL import Image

app = Flask(__name__)


# Model Definition
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


# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)
model.load_state_dict(torch.load(
    'dogs_vs_cats_pytorch_model.pth', map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Dog vs Cat Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        .container { max-width: 600px; margin: auto; }
        input[type=file] { margin: 20px; }
        .result { margin-top: 20px; font-size: 1.2em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dog vs Cat Classifier</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <input type="submit" value="Upload and Predict">
        </form>
        {% if result %}
        <div class="result">{{ result }}</div>
        {% endif %}
    </div>
</body>
</html>
'''


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    result = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file uploaded', 400
        file = request.files['image']
        if file.filename == '':
            return 'No file selected', 400

        try:
            # Process image
            img = Image.open(file).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = model(img).squeeze()
                prediction = output > 0.5
            result = "Cho phép chó vào." if prediction else "Không được phép vào."
        except Exception as e:
            result = f"Error processing image: {e}"

    return render_template_string(HTML_TEMPLATE, result=result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
