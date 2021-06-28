# use pretrained models in torchvision
from torchvision import models

# in models there are some basic models
# print(dir(models))
# download pretrained resnet model
resnet = models.resnet101(pretrained=True)

# define transform procedure to transform image
from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# import image
from PIL import Image
img = Image.open('cat copy.JPG')
# transform the image
img_t = preprocess(img)

import torch
batch_t = torch.unsqueeze(img_t, 0)

resnet.eval()
out = resnet(batch_t)

with open('imagenet1000_clsidx_to_labels.txt') as f:
        labels = [line.strip() for line in f.readlines()]

# get max index in row dim=1
_, index = torch.max(out, dim=1)
# because return is an tensor index, tensor([281]), so we use index[0] to get

# import softmax
from torch.nn import functional as F
percentage = F.softmax(out, dim=1)[0]*100
predict, probability = labels[index[0]], percentage[index[0]].item()
print("ResNet predict the image is a " + str(predict) +"  and probability is "+str(probability))

# give first five prediction
_, indices = torch.sort(out, descending=True)
first_5 = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
print("first_5 are " +str(first_5))