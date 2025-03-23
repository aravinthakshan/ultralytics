from ultralytics.models.rtdetr import RTDETR
# from ultralytics import YOLO
from ultralytics.engine.model import Model
import torch
import wandb
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project="RTDETR_mew_exp")
# Load the custom model configuration
model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-x.yaml')
model.model.to(device)

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
Result_Final_model = model.train(
    epochs=70,
    optimizer='SOAP',
    project='rtdetr_new_exp',
    save=True,
    imgsz = 640
)
# Finish the W&B run
wandb.finish()