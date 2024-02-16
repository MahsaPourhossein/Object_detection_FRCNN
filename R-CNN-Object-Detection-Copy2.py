#!/usr/bin/env python
# coding: utf-8

# # Object Detection task with Fast R-CNN on new set of dataset
# ***
# ***

# # Load the data and plot  examples of images with its bounding-boxes

# In[1]:


import os
import csv
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

# Folder path containing images
image_folder = 'updated_images3'

# CSV file containing bounding box information
csv_file = 'updated_bounding_boxes3.csv'

# Number of images to plot (up to 10)
num_images_to_plot = 10

# Initialize lists to store image paths and corresponding bounding boxes
image_paths = []
bounding_boxes = []

# Read bounding box information from the CSV file
with open(csv_file, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)  # Skip the header row
    for row in csv_reader:
        image_name, xmin, ymin, xmax, ymax = row
        image_path = os.path.join(image_folder, image_name)
        bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
        image_paths.append(image_path)
        bounding_boxes.append(bbox)

# Randomly select up to 10 images and their corresponding bounding boxes
random.seed(42)  # Set a seed for reproducibility
selected_indices = random.sample(range(len(image_paths)), min(num_images_to_plot, len(image_paths)))

# Create a larger figure for the plots
plt.figure(figsize=(15, 8))  # Adjust the figsize as needed (width, height)

# Plot the selected images with bounding boxes
for i, index in enumerate(selected_indices):
    image_path = image_paths[index]
    bbox = bounding_boxes[index]

    # Open the image
    with Image.open(image_path) as img:
        # Draw the bounding box
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline="red", width=2)

        # Plot the image
        plt.subplot(2, 5, i + 1)  # Create a 2x5 grid of subplots
        plt.imshow(img)
        plt.title(f"Image {i+1}")
        plt.axis('off')

plt.tight_layout()
plt.show()


# # Data preprocessing

# In[3]:


import os
import csv
import random
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Custom dataset class
class CardiacImagesDataset(Dataset):
    def __init__(self, image_info, transform=None):
        self.image_info = image_info  # List of dictionaries
        self.transform = transform

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_data = self.image_info[idx]
        image = Image.open(image_data['path']).convert("RGB")
        boxes = torch.tensor([image_data['boxes']], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)  # Class label for all instances is '1'

        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            image = self.transform(image)

        return image, target

# Define the transforms
transform = T.Compose([
    T.ToTensor(),  # Convert the PIL Image to a tensor
    T.Resize((224, 224)),  # Resize the image for the model
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Read the CSV file and prepare the image_info list
csv_file = 'updated_bounding_boxes3.csv'  # Update this path
image_folder = 'updated_images3'  # Update this path

image_info = []
with open(csv_file, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header
    for row in csv_reader:
        image_path = os.path.join(image_folder, row[0])
        bbox = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]
        image_info.append({'path': image_path, 'boxes': bbox, 'labels': 1})

# Split the dataset into train, validation, and test sets
trainval_info, test_info = train_test_split(image_info, test_size=0.15, random_state=42)
train_info, val_info = train_test_split(trainval_info, test_size=0.15 / 0.85, random_state=42)

# Initialize the datasets
train_dataset = CardiacImagesDataset(train_info, transform=transform)
val_dataset = CardiacImagesDataset(val_info, transform=transform)
test_dataset = CardiacImagesDataset(test_info, transform=transform)

# You can now use these datasets to create DataLoaders for your training loop


# To proceed with training, you will need to set up DataLoaders for each dataset (training, validation, and testing) and then create a training loop to optimize your model. Here's how you can do that:
# 
# First, set up the DataLoaders:

# In[4]:


from torch.utils.data import DataLoader

# Set up the DataLoaders for the train, validation, and test datasets
batch_size = 32  # Adjust the batch size according to your system's capabilities

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


# # Building the R-CNN Object detection model

# Model Initialization
# You've already initialized a Faster R-CNN model with a ResNet backbone. If you're using a pre-trained model, remember it's trained on RGB images. Since your images are grayscale converted to RGB, this should work, but be aware that pre-training might be less effective.

# In[5]:


import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN


#########  Define stage 1 of Fast R-CNN: RPN  #########

# 1.Define Backbone of RPN

# Define the backbone network (ResNet-50) as a feature extraction network for RPN stage
backbone = torchvision.models.resnet50(pretrained=True)

backbone_out_channels = backbone.fc.in_features #The extracted features by Resnet50

backbone.fc = torch.nn.Identity()  # Remove the classification head because we need featuremaps

# Add a Feature Pyramid Network (FPN) over the backbone
backbone = BackboneWithFPN(backbone, return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}, 
                           in_channels_list=[
                               backbone.inplanes // 8,
                               backbone.inplanes // 4,
                               backbone.inplanes // 2,
                               backbone.inplanes,
                           ],
                           out_channels=256)


# 2.Define the anchor box generator of the RPN


# This should match the number of feature maps
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

# Define the anchor generator with the correct sizes and aspect ratios
rpn_anchor_generator = AnchorGenerator(
    sizes=anchor_sizes,
    aspect_ratios=aspect_ratios
)

#########  Define stage 2 of Fast R-CNN: ROI Pooling  #########

# Define the ROI (Region of Interest) align module
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'], output_size=7, sampling_ratio=2
)



# Create the Faster R-CNN model
model = FasterRCNN(
    backbone,
    num_classes=2,  # Include background as a class
    rpn_anchor_generator=rpn_anchor_generator,
    box_roi_pool=roi_pooler
)


# pip install tqdm

# In[7]:


from tqdm import tqdm
import torch.optim as optim

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# And a learning rate scheduler (optional)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10  # Define the number of training epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Wrap the train_loader with tqdm to create a progress bar
    for images, targets in tqdm(train_loader, total=len(train_loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        loss_dict = model(images, targets)  # Forward pass
        losses = sum(loss for loss in loss_dict.values())  # Sum the losses
        
        losses.backward()  # Backpropagation
        optimizer.step()  # Optimize
        
        running_loss += losses.item()
    
    # Update the learning rate
    lr_scheduler.step()
    
    # Print the loss every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")


# In[8]:



# Start of validation loop
with torch.no_grad():
    model.eval()  # Ensure model is in evaluation mode

    for images, targets in val_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)  # Get model predictions
            
        # Optional: Print some predictions for inspection
        if len(predictions) > 0:
            print("Sample prediction:", predictions[0])

    # Note: You'll want to replace this with actual evaluation logic
    print(f"Finished validation for epoch {epoch+1}")


# # Saveing of created bounding-boxes and scores in a csv file

# In[22]:


# Assuming predictions are made in the validation loop or a separate prediction loop
# Define the CSV file where predictions will be saved
predictions_csv = 'predicted_bounding_boxes.csv'

# Open the CSV file in write mode
with open(predictions_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_id', 'bbox', 'score'])  # Writing the header

    # Iterate through predictions and save them
    for image_id, prediction in enumerate(predictions):
        boxes = prediction['boxes'].cpu().numpy()  # Convert to numpy array
        scores = prediction['scores'].cpu().numpy()  # Convert to numpy array

        for box, score in zip(boxes, scores):
            writer.writerow([image_id, box.tolist(), score])


# # Plot prediction on test data

# In[26]:


# Create a larger figure for the plots
# Increase the figsize values to make the plot bigger
plt.figure(figsize=(20, 10))  # Adjust the figsize as needed (width, height)

# Plot the selected images with bounding boxes
for i, index in enumerate(selected_indices):
    image_path = image_paths[index]
    bbox = bounding_boxes[index]

    # Get predicted bounding boxes (assuming you have this function and model ready)
    predicted_boxes = get_predicted_boxes(model, image_path, transform)

    # Open the image
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        # Draw the real bounding box
        draw.rectangle(bbox, outline="red", width=3)  # Increase width for better visibility
        
        # Draw predicted bounding boxes
        for pbox in predicted_boxes:
            draw.rectangle(pbox, outline="blue", width=3)  # Use a different color and increase width for predicted boxes

        # Plot the image
        plt.subplot(2, 5, i + 1)  # Create a 2x5 grid of subplots
        plt.imshow(img)
        plt.title(f"Image {i+1}", fontsize=16)  # You can also adjust the fontsize for better readability
        plt.axis('off')

plt.tight_layout()
plt.show()


# ***
# ***
# 
# # Evaluation of Fast R-CNN with calculation of Intersection Over Union (IOU)
# 
# ***
# ***

# In[ ]:


import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou

# Assuming you have a test_loader with your unseen data
model.eval()  # Set the model to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

ious = []

with torch.no_grad():
    for images, targets in test_loader:
        images = list(img.to(device) for img in images)
        output = model(images)
        
        for i, out in enumerate(output):
            if out['boxes'].shape[0] > 0:
                pred_box = out['boxes'].cpu().numpy()[0]  # Get the first predicted box
            else:
                # Handle the case with no predictions
                print(f"No predictions for image {i+1}, assigning IoU = 0")
                ious.append(0)
                continue  # Skip to the next image

            true_box = targets[i]['boxes'].cpu().numpy()[0]  # Assuming one box per image
            iou = calculate_iou(pred_box, true_box)
            ious.append(iou)

# Calculate average IoU across the dataset
average_iou = sum(ious) / len(ious) if ious else 0
print(f"Average IoU: {average_iou:.4f}")

# Optionally, calculate precision, recall, F1 score based on a specific IoU threshold
iou_threshold = 0.5
correct_detections = [iou >= iou_threshold for iou in ious]
precision = sum(correct_detections) / len(correct_detections) if correct_detections else 0
print(f"Precision at IoU >= {iou_threshold}: {precision:.4f}")

