import timm
import torch

print(timm.__version__)
models = timm.list_models()
print("Timm库支持的模型列表：")
print(models)
model = timm.create_model('vit_small_patch8_224')
print(model.default_cfg)

# pretrained_cfg = timm.models.create_model('swsl_resnet18').default_cfg
# pretrained_cfg = timm.models.create_model('resnet50').default_cfg
# pretrained_cfg['file'] = './pretrain_weights/resnet50.pth'
# print(pretrained_cfg)

# model = timm.models.create_model('resnet50', pretrained=True, pretrained_cfg=pretrained_cfg, features_only=True,
#                                  output_stride=32,
#                                  out_indices=(1, 2, 3, 4))
# swin_tiny_patch4_window7_224', 'swinv2_base_window8_256\tf_efficientnet_b7
model = timm.create_model('vit_small_patch8_224', features_only=True, output_stride=32,
                                out_indices=(1, 2, 3, 4), pretrained=False)
encoder_channels = model.feature_info.channels()
print(encoder_channels)

input = torch.rand(8, 3, 1024, 1024)

res1, res2, res3, res4 = model(input)
print(res1.size())
print(res2.size())
print(res3.size())
print(res4.size())

