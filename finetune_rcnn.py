import os
import json
import torch
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class CamoNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # Assuming annotations['images'] contains image info
        self.imgs = {img['id']: img['file_name'] for img in self.annotations['images']}
    
    def convert_coco_bbox_to_pytorch(self, bbox):
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        return [x_min, y_min, x_max, y_max]

    def __getitem__(self, idx):
        # Assuming annotations['annotations'] contains bbox info
        img_id = self.annotations['annotations'][idx]['image_id']
        img_path = os.path.join(self.root, self.imgs[img_id])
        img = read_image(img_path)
        img = img.float() / 255

        # Bounding boxes
        bbox = self.annotations['annotations'][idx]['bbox']
        bbox = self.convert_coco_bbox_to_pytorch(bbox)
        boxes = torch.as_tensor([bbox], dtype=torch.float32)

        # Assuming all instances have the same label (e.g., camouflaged animal)
        labels = torch.ones((1,), dtype=torch.int64)

        image_id = img_id#torch.tensor([img_id])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYWH", canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
        #return len(self.annotations['annotations'])

# Usage example:
dataset = CamoNetDataset(root='./data/camouflaged_animals/images', annotation_file='./data/camouflaged_animals/annotations.json')
dataset_test = CamoNetDataset(root='./data/camouflaged_animals/images', annotation_file='./data/camouflaged_animals/annotations.json')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image

def show_image_with_boxes(dataset, idx):
    img, target = dataset[idx]
    pil_img = to_pil_image(img)
    fig, ax = plt.subplots(1)
    ax.imshow(pil_img)

    boxes = target["boxes"]  # Directly access the bounding box tensor
    for box in boxes:
        x_min, y_min, width, height = box
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

# Assuming you have an instance of your dataset called `camo_net_dataset`
#show_image_with_boxes(dataset, 0)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (animal) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

import utils
from engine import train_one_epoch, evaluate

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-40])
print(f"dataset len: ", len(dataset))
dataset_test = torch.utils.data.Subset(dataset_test, indices[-40:])
print(f"dataset_test len: ", len(dataset_test))

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn
)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = 5

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

