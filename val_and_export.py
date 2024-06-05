# VALIDATION FOR MOBILE FORMERS

# pip install timm==0.3.4
  # do this for each time you start up a fresh container
# pip install onnxruntime-gpu==1.12.0

# last recorded run (77.3%) final accuracy


import torch
import torchvision.transforms as transforms

import mobile_former as models
from torchvision import datasets, transforms

transform = transforms.Compose([
  transforms.CenterCrop((244, 244)),
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  ])

batch_size = 1

valset = datasets.ImageFolder('/dump/swssd/data_set/imagenet/jpeg/val', transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

"""
Your network here:
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Force the model to run its tensors on the GPU


#model = models.MobileFormer(block_args=[]) #IMPORT MobileFormer model, load weights later on
model = models.mobile_former_508m(pretrained=True)
model.to(device) # run the model on the GPU


'''
Load pre-trained model to train further...
Comment out if you wish to train from scratch
'''
nn_model = '/dump/swdump202/c-mzhu/MobileFormer/mobile-former-508m.pth.tar'
checkpoint = torch.load(nn_model)
model.load_state_dict(checkpoint['state_dict'])
model.half()
print("Now loading for Validation...", nn_model)


"""
Define your number of epochs here:
"""
num_epochs = 1 #How many times the model iterates through the entire dataset

for epoch in range(num_epochs):
  model.eval()
  num_correct = 0
  num_samples = 0

  # Do Validation:
  for batch, (images, labels) in enumerate(valloader):
    images, labels = images.to(device), labels.to(device) # Data needs to be on the GPU

    #batch, channels, width, height = images.shape
    #print(batch, channels, width, height)
    
    logits = model(images.to(torch.float16))

    _,predicted = torch.max(logits, 1)
    num_correct += (predicted == labels).sum().item()
    num_samples += labels.size(0)

    training_accuracy = 100 * num_correct / num_samples
    print(f"Sample: {num_samples}, Initial Training Accuracy: {training_accuracy:.2f}%")
  print(f"Epoch{epoch + 1} - Validation Accuracy: {training_accuracy:.2f}%")

print("Testing Done!...\n")
data = torch.randn(batch_size, 3, 375, 375, requires_grad=True, device="cuda") # dummy input
torch.onnx.export(
    model, 
    data.to(torch.float16),
    'MobileFormer508m_export.onnx', 
    export_params=True, 
    do_constant_folding=True, 
    input_names=['input'],
    output_names=['output'],
    )
print("Exporting Done!...\n")