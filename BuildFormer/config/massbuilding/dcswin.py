from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.mass_dataset import *
from geoseg.models.DCSwin import dcswin_small
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 70
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 4
lr = 1e-3
weight_decay = 2.5e-4
backbone_lr = 1e-4
backbone_weight_decay = 2.5e-4
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "dcswin-small-1024-ms-512crop-e70"
weights_path = "model_weights/massbuilding//{}".format(weights_name)
test_weights_name = "dcswin-small-1024-ms-512crop-e70"
log_name = 'massbuilding//{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 3
save_last = False
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None
#  define the network
net = dcswin_small(num_classes=num_classes)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = False

# define the dataloader
train_dataset = MassBuildDataset(mosaic_ratio=0.25, transform=get_training_transform())
val_dataset = MassBuildDataset(mode='val', img_dir='val_images', mask_dir='val_masks', transform=get_validation_transform())
test_dataset = MassBuildDataset(mode='val', img_dir='test_images', mask_dir='test_masks', transform=get_test_transform())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

