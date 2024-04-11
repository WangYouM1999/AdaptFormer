from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.mass_dataset import *
from geoseg.models.FTUNetFormer import FTUNetFormer
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 105
ignore_index = 255
train_batch_size = 4
val_batch_size = 4
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

# me define
data_name = "massbuilding"
diff_save_path = "/home/wym/projects/BiuldFormer/fig_results/" + data_name + "/diff"
sava_last_name = "last"

weights_name = "ftunetformer"
weights_path = "model_weights/massbuilding/{}".format(weights_name)
test_weights_name = "last"
log_name = 'massbuilding/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = [1]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None
#  define the network
net = FTUNetFormer(num_classes=num_classes, decode_channels=256)

# define the loss
loss = FTUNetFormerLoss(ignore_index=ignore_index)
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
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

