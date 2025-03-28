# from ultralytics.models.rtdetr import RTDETR
from ultralytics import RTDETR
from ultralytics.nn.modules.transformer import convert_ln_to_dyt
import torch
import wandb
import os

### CONFIGS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login(key = "9097b6348907fd8bad133bde5c71d9e0c08fde45")
wandb.init(project="RTDETR_mew_exp")

# model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-x.yaml')
model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml')
model.model.to(device)


# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()

model = convert_ln_to_dyt(model)
# Iterate through layers
# print("Model Layers and Parameters:\n")
# for name, module in model.named_children():
#     pass
    # print(f"Layer Name: {name}, Layer Type: {module}")

# Define a callback to log losses at the end of each training batch
def log_losses(trainer):
    # Access the loss dictionary
    loss_items = trainer.loss_items
    
    # Log each loss component
    wandb.log({
        "train/box_loss": loss_items[0],
        "train/cls_loss": loss_items[1],
        "train/dfl_loss": loss_items[2]
    }, step=trainer.epoch)

    torch.cuda.empty_cache()

# Register the callback with the YOLO model
model.add_callback('on_train_batch_end', log_losses)

# Train the model with the specified configuration and sync to W&B
metrics = model.train(
    epochs=1,
    data="coco128.yaml",
    optimizer='SOAP',
    project='rtdetr_new_exp',
    save=True,
    imgsz = 640
)

metrics = model.val()

print(metrics.box.map)  # map50-95


print(metrics.box.map50)  # map50


print(metrics.box.map75) # map75


print(metrics.box.maps)  # a list contains map50-95 of each category

# Finish the W&B run
wandb.finish()