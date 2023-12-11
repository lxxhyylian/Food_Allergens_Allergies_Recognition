
import streamlit as st
import subprocess
with open("./requirements.txt", "r") as f:
    requirements = f.read().splitlines()
for requirement in requirements:
    try:
        subprocess.run(["pip", "show", requirement], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        subprocess.run(["pip", "install", requirement], check=True)

import torch
from skimage.io import imread as imread
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


with open('./data/allergens.txt', 'r') as file:
    allergens = [line.strip() for line in file]
with open('./data/allergies.txt', 'r') as file:
    allergies = [line.strip() for line in file]

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FineTunedDenseNet(nn.Module):
    def __init__(self, num_classes=len(allergens)):
        super(FineTunedDenseNet, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.conv1x1 = nn.Conv2d(densenet.classifier.in_features, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AllergenModel(FineTunedDenseNet):
    def __init__(self, num_classes=len(allergens)):
        super(AllergenModel, self).__init__(num_classes=num_classes)

class AllergyModel(FineTunedDenseNet):
    def __init__(self, num_classes=len(allergies)):
        super(AllergyModel, self).__init__(num_classes=num_classes)

allergen_model = AllergenModel()
allergy_model = AllergyModel()

allergen_model.load_state_dict(torch.load('./FineTunedDenseNet_allergens_model_1e-4.pth', map_location='cpu'))
allergy_model.load_state_dict(torch.load('./FineTunedDenseNet_allergies_model.pth', map_location='cpu'))
allergen_model.cpu()
allergy_model.cpu()

class CombinedModel(nn.Module):
    def __init__(self, allergen_model, allergy_model):
        super(CombinedModel, self).__init__()
        self.allergen_model = allergen_model
        self.allergy_model = allergy_model

    def forward(self, x):
        allergen_output = self.allergen_model(x)
        allergy_output = self.allergy_model(x)
        return allergen_output, allergy_output

combined_model = CombinedModel(allergen_model, allergy_model)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("lxhyylian-Food Allergens and Allergies Recognition")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_tensor = transform(img).unsqueeze(0)
    threshold = 0.5
    combined_model.eval()
    with torch.no_grad():
        allergen_output, allergy_output = combined_model(img_tensor)

    pred_allergen_title = ', '.join(['{} ({:2.1f}%)'.format(allergens[j], 100 * torch.sigmoid(allergen_output[0, j]).item())
                            for j, v in enumerate(allergen_output.squeeze())
                            if torch.sigmoid(v) > threshold])
    pred_allergy_title = ', '.join(['{} ({:2.1f}%)'.format(allergies[j], 100 * torch.sigmoid(allergy_output[0, j]).item())
                            for j, v in enumerate(allergy_output.squeeze())
                            if torch.sigmoid(v) > threshold])


    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f'Predicted Allergens:\n{pred_allergen_title}\n\nPredicted Allergies:\n{pred_allergy_title}')
