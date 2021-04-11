# 在mmdet框架中使用config配置文件构建网络模型


```python
from   mmdet.models.builder import build_detector
from   mmcv import Config
import torch
import mmcv
```


```python
from   os.path import join,realpath,relpath
import glob
config_dpath = 'configs'+"/yolo"
config_fpaths = list(glob.glob(join(config_dpath,'*.py')))
config_fpaths = [p for p in config_fpaths if p.find('_base_') == -1]
config_names = [relpath(p, config_dpath) for p in config_fpaths]
```


```python
config_names
```




    ['yolov3 - 副本.py',
     'yolov3.py',
     'yolov3_d53_320_273e_coco.py',
     'yolov3_d53_mstrain-416_273e_coco.py',
     'yolov3_d53_mstrain-608_273e_coco.py']



### 1. 解析yolov3配置文件


```python
for config_fname in config_names[1:2]:
    config_fpath = join(config_dpath, config_fname)
    config_mod = Config.fromfile(config_fpath)
print(config_fname,'\n',config_fpath,'\n',config_mod)
```

    yolov3.py 
     configs/yolo\yolov3.py 
     Config (path: configs/yolo\yolov3.py): {'model': {'type': 'YOLOV3', 'pretrained': 'open-mmlab://darknet53', 'backbone': {'type': 'Darknet', 'depth': 53, 'out_indices': (3, 4, 5)}, 'neck': {'type': 'YOLOV3Neck', 'num_scales': 3, 'in_channels': [1024, 512, 256], 'out_channels': [512, 256, 128]}, 'bbox_head': {'type': 'YOLOV3Head', 'num_classes': 80, 'in_channels': [512, 256, 128], 'out_channels': [1024, 512, 256], 'anchor_generator': {'type': 'YOLOAnchorGenerator', 'base_sizes': [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]], 'strides': [32, 16, 8]}, 'bbox_coder': {'type': 'YOLOBBoxCoder'}, 'featmap_strides': [32, 16, 8], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_conf': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_xy': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 2.0, 'reduction': 'sum'}, 'loss_wh': {'type': 'MSELoss', 'loss_weight': 2.0, 'reduction': 'sum'}}, 'train_cfg': {'assigner': {'type': 'GridAssigner', 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.5, 'min_pos_iou': 0}}, 'test_cfg': {'nms_pre': 1000, 'min_bbox_size': 0, 'score_thr': 0.05, 'conf_thr': 0.005, 'nms': {'type': 'nms', 'iou_threshold': 0.45}, 'max_per_img': 100}}, 'dataset_type': 'CocoDataset', 'data_root': 'F:/datasets/COCO2017/', 'img_norm_cfg': {'mean': [0, 0, 0], 'std': [255.0, 255.0, 255.0], 'to_rgb': True}, 'test_pipeline': [{'type': 'LoadImageFromFile', 'to_float32': True}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'type': 'PhotoMetricDistortion'}, {'type': 'Resize', 'img_scale': [(320, 320), (608, 608)], 'keep_ratio': True}], 'train_pipeline': [{'type': 'LoadImageFromFile', 'to_float32': True}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'type': 'PhotoMetricDistortion'}, {'type': 'Resize', 'img_scale': [(320, 320), (608, 608)], 'keep_ratio': True}], 'data': {'samples_per_gpu': 8, 'workers_per_gpu': 4, 'train': {'type': 'CocoDataset', 'ann_file': 'F:/datasets/COCO2017/annotations/instances_train2017.json', 'img_prefix': 'F:/datasets/COCO2017/train2017/', 'pipeline': [{'type': 'LoadImageFromFile', 'to_float32': True}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'type': 'PhotoMetricDistortion'}, {'type': 'Resize', 'img_scale': [(320, 320), (608, 608)], 'keep_ratio': True}]}, 'val': {'type': 'CocoDataset', 'ann_file': 'F:/datasets/COCO2017/annotations/instances_val2017.json', 'img_prefix': 'F:/datasets/COCO2017/val2017/', 'pipeline': [{'type': 'LoadImageFromFile', 'to_float32': True}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'type': 'PhotoMetricDistortion'}, {'type': 'Resize', 'img_scale': [(320, 320), (608, 608)], 'keep_ratio': True}]}, 'test': {'type': 'CocoDataset', 'ann_file': 'F:/datasets/COCO2017/annotations/instances_val2017.json', 'img_prefix': 'F:/datasets/COCO2017/val2017/', 'pipeline': [{'type': 'LoadImageFromFile', 'to_float32': True}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'type': 'PhotoMetricDistortion'}, {'type': 'Resize', 'img_scale': [(320, 320), (608, 608)], 'keep_ratio': True}]}}}
    


```python
for k,v in  config_mod.items():
    print(k,":",v)
```

    model : {'type': 'YOLOV3', 'pretrained': 'open-mmlab://darknet53', 'backbone': {'type': 'Darknet', 'depth': 53, 'out_indices': (3, 4, 5)}, 'neck': {'type': 'YOLOV3Neck', 'num_scales': 3, 'in_channels': [1024, 512, 256], 'out_channels': [512, 256, 128]}, 'bbox_head': {'type': 'YOLOV3Head', 'num_classes': 80, 'in_channels': [512, 256, 128], 'out_channels': [1024, 512, 256], 'anchor_generator': {'type': 'YOLOAnchorGenerator', 'base_sizes': [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]], 'strides': [32, 16, 8]}, 'bbox_coder': {'type': 'YOLOBBoxCoder'}, 'featmap_strides': [32, 16, 8], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_conf': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_xy': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 2.0, 'reduction': 'sum'}, 'loss_wh': {'type': 'MSELoss', 'loss_weight': 2.0, 'reduction': 'sum'}}, 'train_cfg': {'assigner': {'type': 'GridAssigner', 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.5, 'min_pos_iou': 0}}, 'test_cfg': {'nms_pre': 1000, 'min_bbox_size': 0, 'score_thr': 0.05, 'conf_thr': 0.005, 'nms': {'type': 'nms', 'iou_threshold': 0.45}, 'max_per_img': 100}}
    dataset_type : CocoDataset
    data_root : data/coco/
    img_norm_cfg : {'mean': [0, 0, 0], 'std': [255.0, 255.0, 255.0], 'to_rgb': True}
    test_pipeline : []
    train_pipeline : []
    data : {'samples_per_gpu': 8, 'workers_per_gpu': 4, 'train': {'type': 'CocoDataset', 'ann_file': 'data/coco/annotations/instances_train2017.json', 'img_prefix': 'data/coco/train2017/', 'pipeline': []}, 'val': {'type': 'CocoDataset', 'ann_file': 'data/coco/annotations/instances_val2017.json', 'img_prefix': 'data/coco/val2017/', 'pipeline': []}, 'test': {'type': 'CocoDataset', 'ann_file': 'data/coco/annotations/instances_val2017.json', 'img_prefix': 'data/coco/val2017/', 'pipeline': []}}
    


```python
config_mod.model
```




    {'type': 'YOLOV3',
     'pretrained': 'open-mmlab://darknet53',
     'backbone': {'type': 'Darknet', 'depth': 53, 'out_indices': (3, 4, 5)},
     'neck': {'type': 'YOLOV3Neck',
      'num_scales': 3,
      'in_channels': [1024, 512, 256],
      'out_channels': [512, 256, 128]},
     'bbox_head': {'type': 'YOLOV3Head',
      'num_classes': 80,
      'in_channels': [512, 256, 128],
      'out_channels': [1024, 512, 256],
      'anchor_generator': {'type': 'YOLOAnchorGenerator',
       'base_sizes': [[(116, 90), (156, 198), (373, 326)],
        [(30, 61), (62, 45), (59, 119)],
        [(10, 13), (16, 30), (33, 23)]],
       'strides': [32, 16, 8]},
      'bbox_coder': {'type': 'YOLOBBoxCoder'},
      'featmap_strides': [32, 16, 8],
      'loss_cls': {'type': 'CrossEntropyLoss',
       'use_sigmoid': True,
       'loss_weight': 1.0,
       'reduction': 'sum'},
      'loss_conf': {'type': 'CrossEntropyLoss',
       'use_sigmoid': True,
       'loss_weight': 1.0,
       'reduction': 'sum'},
      'loss_xy': {'type': 'CrossEntropyLoss',
       'use_sigmoid': True,
       'loss_weight': 2.0,
       'reduction': 'sum'},
      'loss_wh': {'type': 'MSELoss', 'loss_weight': 2.0, 'reduction': 'sum'}},
     'train_cfg': {'assigner': {'type': 'GridAssigner',
       'pos_iou_thr': 0.5,
       'neg_iou_thr': 0.5,
       'min_pos_iou': 0}},
     'test_cfg': {'nms_pre': 1000,
      'min_bbox_size': 0,
      'score_thr': 0.05,
      'conf_thr': 0.005,
      'nms': {'type': 'nms', 'iou_threshold': 0.45},
      'max_per_img': 100}}




```python
if 'pretrained' in config_mod.model:
    config_mod.model['pretrained'] = None
```

### 2.构建检测器（使用构建函数和hook）


```python
from mmdet.models import build_detector
model=build_detector(config_mod.model)
```


```python
print(model)
```

    YOLOV3(
      (backbone): Darknet(
        (conv1): ConvModule(
          (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv_res_block1): Sequential(
          (conv): ConvModule(
            (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (res0): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
        (conv_res_block2): Sequential(
          (conv): ConvModule(
            (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (res0): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res1): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
        (conv_res_block3): Sequential(
          (conv): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (res0): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res1): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res2): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res3): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res4): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res5): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res6): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res7): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
        (conv_res_block4): Sequential(
          (conv): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (res0): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res1): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res2): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res3): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res4): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res5): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res6): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res7): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
        (conv_res_block5): Sequential(
          (conv): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (res0): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res1): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res2): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (res3): ResBlock(
            (conv1): ConvModule(
              (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
            (conv2): ConvModule(
              (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
      )
      (neck): YOLOV3Neck(
        (detect1): DetectionBlock(
          (conv1): ConvModule(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv3): ConvModule(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv4): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv5): ConvModule(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (conv1): ConvModule(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (detect2): DetectionBlock(
          (conv1): ConvModule(
            (conv): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv3): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv4): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv5): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (conv2): ConvModule(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (detect3): DetectionBlock(
          (conv1): ConvModule(
            (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv3): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv4): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv5): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
      (bbox_head): YOLOV3Head(
        (loss_cls): CrossEntropyLoss()
        (loss_conf): CrossEntropyLoss()
        (loss_xy): CrossEntropyLoss()
        (loss_wh): MSELoss()
        (convs_bridge): ModuleList(
          (0): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (1): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (convs_pred): ModuleList(
          (0): Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
          (2): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    


```python
from mmdet.models import DETECTORS
DETECTORS.module_dict
```




    {'SingleStageDetector': mmdet.models.detectors.single_stage.SingleStageDetector,
     'ATSS': mmdet.models.detectors.atss.ATSS,
     'TwoStageDetector': mmdet.models.detectors.two_stage.TwoStageDetector,
     'CascadeRCNN': mmdet.models.detectors.cascade_rcnn.CascadeRCNN,
     'CornerNet': mmdet.models.detectors.cornernet.CornerNet,
     'DETR': mmdet.models.detectors.detr.DETR,
     'FastRCNN': mmdet.models.detectors.fast_rcnn.FastRCNN,
     'FasterRCNN': mmdet.models.detectors.faster_rcnn.FasterRCNN,
     'FCOS': mmdet.models.detectors.fcos.FCOS,
     'FOVEA': mmdet.models.detectors.fovea.FOVEA,
     'FSAF': mmdet.models.detectors.fsaf.FSAF,
     'GFL': mmdet.models.detectors.gfl.GFL,
     'GridRCNN': mmdet.models.detectors.grid_rcnn.GridRCNN,
     'HybridTaskCascade': mmdet.models.detectors.htc.HybridTaskCascade,
     'MaskRCNN': mmdet.models.detectors.mask_rcnn.MaskRCNN,
     'MaskScoringRCNN': mmdet.models.detectors.mask_scoring_rcnn.MaskScoringRCNN,
     'NASFCOS': mmdet.models.detectors.nasfcos.NASFCOS,
     'PAA': mmdet.models.detectors.paa.PAA,
     'PointRend': mmdet.models.detectors.point_rend.PointRend,
     'RepPointsDetector': mmdet.models.detectors.reppoints_detector.RepPointsDetector,
     'RetinaNet': mmdet.models.detectors.retinanet.RetinaNet,
     'RPN': mmdet.models.detectors.rpn.RPN,
     'SCNet': mmdet.models.detectors.scnet.SCNet,
     'SparseRCNN': mmdet.models.detectors.sparse_rcnn.SparseRCNN,
     'TridentFasterRCNN': mmdet.models.detectors.trident_faster_rcnn.TridentFasterRCNN,
     'VFNet': mmdet.models.detectors.vfnet.VFNet,
     'YOLACT': mmdet.models.detectors.yolact.YOLACT,
     'YOLOV3': mmdet.models.detectors.yolo.YOLOV3}




```python
# model=build_detector({'type':'YOLOV5','backbone':{}, 'neck':{},'bbox_head':{}})
```


```python
m=DETECTORS.get(config_mod.model['type'])
# print(m(backbone={},neck={},bbox_head={}))
```

### 2.1 构建主干网络


```python
from mmdet.models import build_backbone,BACKBONES
BACKBONES.module_dict
```




    {'Darknet': mmdet.models.backbones.darknet.Darknet,
     'ResNet': mmdet.models.backbones.resnet.ResNet,
     'ResNetV1d': mmdet.models.backbones.resnet.ResNetV1d,
     'DetectoRS_ResNet': mmdet.models.backbones.detectors_resnet.DetectoRS_ResNet,
     'DetectoRS_ResNeXt': mmdet.models.backbones.detectors_resnext.DetectoRS_ResNeXt,
     'HourglassNet': mmdet.models.backbones.hourglass.HourglassNet,
     'HRNet': mmdet.models.backbones.hrnet.HRNet,
     'ResNeXt': mmdet.models.backbones.resnext.ResNeXt,
     'RegNet': mmdet.models.backbones.regnet.RegNet,
     'Res2Net': mmdet.models.backbones.res2net.Res2Net,
     'ResNeSt': mmdet.models.backbones.resnest.ResNeSt,
     'SSDVGG': mmdet.models.backbones.ssd_vgg.SSDVGG,
     'TridentResNet': mmdet.models.backbones.trident_resnet.TridentResNet}



2.1.1 Use  backbone  builder


```python
backbone_config=config_mod.model['backbone']
print(backbone_config)
```

    {'type': 'Darknet', 'depth': 53, 'out_indices': (3, 4, 5)}
    


```python
backone=build_backbone(config_mod.model['backbone'])
print(backone)
```

    Darknet(
      (conv1): ConvModule(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (conv_res_block1): Sequential(
        (conv): ConvModule(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (res0): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
      (conv_res_block2): Sequential(
        (conv): ConvModule(
          (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (res0): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res1): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
      (conv_res_block3): Sequential(
        (conv): ConvModule(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (res0): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res1): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res2): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res3): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res4): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res5): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res6): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res7): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
      (conv_res_block4): Sequential(
        (conv): ConvModule(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (res0): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res1): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res2): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res3): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res4): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res5): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res6): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res7): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
      (conv_res_block5): Sequential(
        (conv): ConvModule(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (res0): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res1): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res2): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res3): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
    )
    

2.1.2 Use code hook


```python
bo=BACKBONES.get(backbone_config['type'])
bc=backbone_config.pop('type')
print(bc)
```

    Darknet
    


```python
backbone_config
```




    {'depth': 53, 'out_indices': (3, 4, 5)}




```python
bo(**backbone_config)
```




    Darknet(
      (conv1): ConvModule(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (conv_res_block1): Sequential(
        (conv): ConvModule(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (res0): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
      (conv_res_block2): Sequential(
        (conv): ConvModule(
          (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (res0): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res1): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
      (conv_res_block3): Sequential(
        (conv): ConvModule(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (res0): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res1): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res2): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res3): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res4): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res5): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res6): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res7): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
      (conv_res_block4): Sequential(
        (conv): ConvModule(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (res0): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res1): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res2): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res3): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res4): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res5): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res6): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res7): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
      (conv_res_block5): Sequential(
        (conv): ConvModule(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (res0): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res1): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res2): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (res3): ResBlock(
          (conv1): ConvModule(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
          (conv2): ConvModule(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
    )



### 2.2 Builder  necks 


```python
from mmdet.models import build_neck,NECKS
NECKS.module_dict
```




    {'BFP': mmdet.models.necks.bfp.BFP,
     'ChannelMapper': mmdet.models.necks.channel_mapper.ChannelMapper,
     'FPG': mmdet.models.necks.fpg.FPG,
     'FPN': mmdet.models.necks.fpn.FPN,
     'FPN_CARAFE': mmdet.models.necks.fpn_carafe.FPN_CARAFE,
     'HRFPN': mmdet.models.necks.hrfpn.HRFPN,
     'NASFPN': mmdet.models.necks.nas_fpn.NASFPN,
     'NASFCOS_FPN': mmdet.models.necks.nasfcos_fpn.NASFCOS_FPN,
     'PAFPN': mmdet.models.necks.pafpn.PAFPN,
     'RFP': mmdet.models.necks.rfp.RFP,
     'YOLOV3Neck': mmdet.models.necks.yolo_neck.YOLOV3Neck}




```python
neck = build_neck(config_mod.model['neck'])
```


```python
neck
```




    YOLOV3Neck(
      (detect1): DetectionBlock(
        (conv1): ConvModule(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv2): ConvModule(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv3): ConvModule(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv4): ConvModule(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv5): ConvModule(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (conv1): ConvModule(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (detect2): DetectionBlock(
        (conv1): ConvModule(
          (conv): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv2): ConvModule(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv3): ConvModule(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv4): ConvModule(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv5): ConvModule(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (conv2): ConvModule(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (detect3): DetectionBlock(
        (conv1): ConvModule(
          (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv2): ConvModule(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv3): ConvModule(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv4): ConvModule(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv5): ConvModule(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
    )




```python
neck_config=config_mod.model['neck']
neck_config.pop('type')
print(neck_config)
```

    {'num_scales': 3, 'in_channels': [1024, 512, 256], 'out_channels': [512, 256, 128]}
    


```python
NECKS.get('YOLOV3Neck')(**neck_config)
```




    YOLOV3Neck(
      (detect1): DetectionBlock(
        (conv1): ConvModule(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv2): ConvModule(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv3): ConvModule(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv4): ConvModule(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv5): ConvModule(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (conv1): ConvModule(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (detect2): DetectionBlock(
        (conv1): ConvModule(
          (conv): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv2): ConvModule(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv3): ConvModule(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv4): ConvModule(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv5): ConvModule(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (conv2): ConvModule(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (detect3): DetectionBlock(
        (conv1): ConvModule(
          (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv2): ConvModule(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv3): ConvModule(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv4): ConvModule(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (conv5): ConvModule(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
    )



#### 2.3 构建检测头 head


```python
from  mmdet.models import build_head,HEADS
HEADS.module_dict
```




    {'AnchorFreeHead': mmdet.models.dense_heads.anchor_free_head.AnchorFreeHead,
     'AnchorHead': mmdet.models.dense_heads.anchor_head.AnchorHead,
     'ATSSHead': mmdet.models.dense_heads.atss_head.ATSSHead,
     'RPNHead': mmdet.models.dense_heads.rpn_head.RPNHead,
     'StageCascadeRPNHead': mmdet.models.dense_heads.cascade_rpn_head.StageCascadeRPNHead,
     'CascadeRPNHead': mmdet.models.dense_heads.cascade_rpn_head.CascadeRPNHead,
     'CornerHead': mmdet.models.dense_heads.corner_head.CornerHead,
     'CentripetalHead': mmdet.models.dense_heads.centripetal_head.CentripetalHead,
     'EmbeddingRPNHead': mmdet.models.dense_heads.embedding_rpn_head.EmbeddingRPNHead,
     'FCOSHead': mmdet.models.dense_heads.fcos_head.FCOSHead,
     'FoveaHead': mmdet.models.dense_heads.fovea_head.FoveaHead,
     'RetinaHead': mmdet.models.dense_heads.retina_head.RetinaHead,
     'FreeAnchorRetinaHead': mmdet.models.dense_heads.free_anchor_retina_head.FreeAnchorRetinaHead,
     'FSAFHead': mmdet.models.dense_heads.fsaf_head.FSAFHead,
     'GuidedAnchorHead': mmdet.models.dense_heads.guided_anchor_head.GuidedAnchorHead,
     'GARetinaHead': mmdet.models.dense_heads.ga_retina_head.GARetinaHead,
     'GARPNHead': mmdet.models.dense_heads.ga_rpn_head.GARPNHead,
     'GFLHead': mmdet.models.dense_heads.gfl_head.GFLHead,
     'NASFCOSHead': mmdet.models.dense_heads.nasfcos_head.NASFCOSHead,
     'PAAHead': mmdet.models.dense_heads.paa_head.PAAHead,
     'PISARetinaHead': mmdet.models.dense_heads.pisa_retinanet_head.PISARetinaHead,
     'SSDHead': mmdet.models.dense_heads.ssd_head.SSDHead,
     'PISASSDHead': mmdet.models.dense_heads.pisa_ssd_head.PISASSDHead,
     'RepPointsHead': mmdet.models.dense_heads.reppoints_head.RepPointsHead,
     'RetinaSepBNHead': mmdet.models.dense_heads.retina_sepbn_head.RetinaSepBNHead,
     'SABLRetinaHead': mmdet.models.dense_heads.sabl_retina_head.SABLRetinaHead,
     'TransformerHead': mmdet.models.dense_heads.transformer_head.TransformerHead,
     'VFNetHead': mmdet.models.dense_heads.vfnet_head.VFNetHead,
     'YOLACTHead': mmdet.models.dense_heads.yolact_head.YOLACTHead,
     'YOLACTSegmHead': mmdet.models.dense_heads.yolact_head.YOLACTSegmHead,
     'YOLACTProtonet': mmdet.models.dense_heads.yolact_head.YOLACTProtonet,
     'YOLOV3Head': mmdet.models.dense_heads.yolo_head.YOLOV3Head,
     'BBoxHead': mmdet.models.roi_heads.bbox_heads.bbox_head.BBoxHead,
     'ConvFCBBoxHead': mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.ConvFCBBoxHead,
     'Shared2FCBBoxHead': mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.Shared2FCBBoxHead,
     'Shared4Conv1FCBBoxHead': mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.Shared4Conv1FCBBoxHead,
     'DIIHead': mmdet.models.roi_heads.bbox_heads.dii_head.DIIHead,
     'DoubleConvFCBBoxHead': mmdet.models.roi_heads.bbox_heads.double_bbox_head.DoubleConvFCBBoxHead,
     'SABLHead': mmdet.models.roi_heads.bbox_heads.sabl_head.SABLHead,
     'SCNetBBoxHead': mmdet.models.roi_heads.bbox_heads.scnet_bbox_head.SCNetBBoxHead,
     'CascadeRoIHead': mmdet.models.roi_heads.cascade_roi_head.CascadeRoIHead,
     'StandardRoIHead': mmdet.models.roi_heads.standard_roi_head.StandardRoIHead,
     'DoubleHeadRoIHead': mmdet.models.roi_heads.double_roi_head.DoubleHeadRoIHead,
     'DynamicRoIHead': mmdet.models.roi_heads.dynamic_roi_head.DynamicRoIHead,
     'GridRoIHead': mmdet.models.roi_heads.grid_roi_head.GridRoIHead,
     'HybridTaskCascadeRoIHead': mmdet.models.roi_heads.htc_roi_head.HybridTaskCascadeRoIHead,
     'FCNMaskHead': mmdet.models.roi_heads.mask_heads.fcn_mask_head.FCNMaskHead,
     'CoarseMaskHead': mmdet.models.roi_heads.mask_heads.coarse_mask_head.CoarseMaskHead,
     'FeatureRelayHead': mmdet.models.roi_heads.mask_heads.feature_relay_head.FeatureRelayHead,
     'FusedSemanticHead': mmdet.models.roi_heads.mask_heads.fused_semantic_head.FusedSemanticHead,
     'GlobalContextHead': mmdet.models.roi_heads.mask_heads.global_context_head.GlobalContextHead,
     'GridHead': mmdet.models.roi_heads.mask_heads.grid_head.GridHead,
     'HTCMaskHead': mmdet.models.roi_heads.mask_heads.htc_mask_head.HTCMaskHead,
     'MaskPointHead': mmdet.models.roi_heads.mask_heads.mask_point_head.MaskPointHead,
     'MaskIoUHead': mmdet.models.roi_heads.mask_heads.maskiou_head.MaskIoUHead,
     'SCNetMaskHead': mmdet.models.roi_heads.mask_heads.scnet_mask_head.SCNetMaskHead,
     'SCNetSemanticHead': mmdet.models.roi_heads.mask_heads.scnet_semantic_head.SCNetSemanticHead,
     'MaskScoringRoIHead': mmdet.models.roi_heads.mask_scoring_roi_head.MaskScoringRoIHead,
     'PISARoIHead': mmdet.models.roi_heads.pisa_roi_head.PISARoIHead,
     'PointRendRoIHead': mmdet.models.roi_heads.point_rend_roi_head.PointRendRoIHead,
     'SCNetRoIHead': mmdet.models.roi_heads.scnet_roi_head.SCNetRoIHead,
     'SparseRoIHead': mmdet.models.roi_heads.sparse_roi_head.SparseRoIHead,
     'TridentRoIHead': mmdet.models.roi_heads.trident_roi_head.TridentRoIHead}




```python
config_mod.model['bbox_head']
```




    {'type': 'YOLOV3Head',
     'num_classes': 80,
     'in_channels': [512, 256, 128],
     'out_channels': [1024, 512, 256],
     'anchor_generator': {'type': 'YOLOAnchorGenerator',
      'base_sizes': [[(116, 90), (156, 198), (373, 326)],
       [(30, 61), (62, 45), (59, 119)],
       [(10, 13), (16, 30), (33, 23)]],
      'strides': [32, 16, 8]},
     'bbox_coder': {'type': 'YOLOBBoxCoder'},
     'featmap_strides': [32, 16, 8],
     'loss_cls': {'type': 'CrossEntropyLoss',
      'use_sigmoid': True,
      'loss_weight': 1.0,
      'reduction': 'sum'},
     'loss_conf': {'type': 'CrossEntropyLoss',
      'use_sigmoid': True,
      'loss_weight': 1.0,
      'reduction': 'sum'},
     'loss_xy': {'type': 'CrossEntropyLoss',
      'use_sigmoid': True,
      'loss_weight': 2.0,
      'reduction': 'sum'},
     'loss_wh': {'type': 'MSELoss', 'loss_weight': 2.0, 'reduction': 'sum'},
     'train_cfg': {'assigner': {'type': 'GridAssigner',
       'pos_iou_thr': 0.5,
       'neg_iou_thr': 0.5,
       'min_pos_iou': 0}},
     'test_cfg': {'nms_pre': 1000,
      'min_bbox_size': 0,
      'score_thr': 0.05,
      'conf_thr': 0.005,
      'nms': {'type': 'nms', 'iou_threshold': 0.45},
      'max_per_img': 100}}




```python
bbox_head=build_head(config_mod.model['bbox_head'])
print(bbox_head)
```

    YOLOV3Head(
      (loss_cls): CrossEntropyLoss()
      (loss_conf): CrossEntropyLoss()
      (loss_xy): CrossEntropyLoss()
      (loss_wh): MSELoss()
      (convs_bridge): ModuleList(
        (0): ConvModule(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (1): ConvModule(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (2): ConvModule(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (convs_pred): ModuleList(
        (0): Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    


```python
head=HEADS.get('YOLOV3Head')
```


```python
head_config = config_mod.model['bbox_head']
print(head_config)
```

    {'type': 'YOLOV3Head', 'num_classes': 80, 'in_channels': [512, 256, 128], 'out_channels': [1024, 512, 256], 'anchor_generator': {'type': 'YOLOAnchorGenerator', 'base_sizes': [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]], 'strides': [32, 16, 8]}, 'bbox_coder': {'type': 'YOLOBBoxCoder'}, 'featmap_strides': [32, 16, 8], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_conf': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_xy': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 2.0, 'reduction': 'sum'}, 'loss_wh': {'type': 'MSELoss', 'loss_weight': 2.0, 'reduction': 'sum'}, 'train_cfg': {'assigner': {'type': 'GridAssigner', 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.5, 'min_pos_iou': 0}}, 'test_cfg': {'nms_pre': 1000, 'min_bbox_size': 0, 'score_thr': 0.05, 'conf_thr': 0.005, 'nms': {'type': 'nms', 'iou_threshold': 0.45}, 'max_per_img': 100}}
    


```python
head_config.pop('type')
```




    'YOLOV3Head'




```python
head(**head_config)
```




    YOLOV3Head(
      (loss_cls): CrossEntropyLoss()
      (loss_conf): CrossEntropyLoss()
      (loss_xy): CrossEntropyLoss()
      (loss_wh): MSELoss()
      (convs_bridge): ModuleList(
        (0): ConvModule(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (1): ConvModule(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (2): ConvModule(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (convs_pred): ModuleList(
        (0): Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
      )
    )



### 3. 构建数据集 


```python
from mmdet.datasets import build_dataset,build_dataloader,DATASETS,PIPELINES
DATASETS.module_dict
```




    {'CustomDataset': mmdet.datasets.custom.CustomDataset,
     'CocoDataset': mmdet.datasets.coco.CocoDataset,
     'CityscapesDataset': mmdet.datasets.cityscapes.CityscapesDataset,
     'ConcatDataset': mmdet.datasets.dataset_wrappers.ConcatDataset,
     'RepeatDataset': mmdet.datasets.dataset_wrappers.RepeatDataset,
     'ClassBalancedDataset': mmdet.datasets.dataset_wrappers.ClassBalancedDataset,
     'DeepFashionDataset': mmdet.datasets.deepfashion.DeepFashionDataset,
     'LVISV05Dataset': mmdet.datasets.lvis.LVISV05Dataset,
     'LVISDataset': mmdet.datasets.lvis.LVISV05Dataset,
     'LVISV1Dataset': mmdet.datasets.lvis.LVISV1Dataset,
     'XMLDataset': mmdet.datasets.xml_style.XMLDataset,
     'VOCDataset': mmdet.datasets.voc.VOCDataset,
     'WIDERFaceDataset': mmdet.datasets.wider_face.WIDERFaceDataset}




```python
PIPELINES.module_dict
```




    {'Compose': mmdet.datasets.pipelines.compose.Compose,
     'AutoAugment': mmdet.datasets.pipelines.auto_augment.AutoAugment,
     'Shear': mmdet.datasets.pipelines.auto_augment.Shear,
     'Rotate': mmdet.datasets.pipelines.auto_augment.Rotate,
     'Translate': mmdet.datasets.pipelines.auto_augment.Translate,
     'ColorTransform': mmdet.datasets.pipelines.auto_augment.ColorTransform,
     'EqualizeTransform': mmdet.datasets.pipelines.auto_augment.EqualizeTransform,
     'BrightnessTransform': mmdet.datasets.pipelines.auto_augment.BrightnessTransform,
     'ContrastTransform': mmdet.datasets.pipelines.auto_augment.ContrastTransform,
     'ToTensor': mmdet.datasets.pipelines.formating.ToTensor,
     'ImageToTensor': mmdet.datasets.pipelines.formating.ImageToTensor,
     'Transpose': mmdet.datasets.pipelines.formating.Transpose,
     'ToDataContainer': mmdet.datasets.pipelines.formating.ToDataContainer,
     'DefaultFormatBundle': mmdet.datasets.pipelines.formating.DefaultFormatBundle,
     'Collect': mmdet.datasets.pipelines.formating.Collect,
     'WrapFieldsToLists': mmdet.datasets.pipelines.formating.WrapFieldsToLists,
     'InstaBoost': mmdet.datasets.pipelines.instaboost.InstaBoost,
     'LoadImageFromFile': mmdet.datasets.pipelines.loading.LoadImageFromFile,
     'LoadImageFromWebcam': mmdet.datasets.pipelines.loading.LoadImageFromWebcam,
     'LoadMultiChannelImageFromFiles': mmdet.datasets.pipelines.loading.LoadMultiChannelImageFromFiles,
     'LoadAnnotations': mmdet.datasets.pipelines.loading.LoadAnnotations,
     'LoadProposals': mmdet.datasets.pipelines.loading.LoadProposals,
     'FilterAnnotations': mmdet.datasets.pipelines.loading.FilterAnnotations,
     'MultiScaleFlipAug': mmdet.datasets.pipelines.test_time_aug.MultiScaleFlipAug,
     'Resize': mmdet.datasets.pipelines.transforms.Resize,
     'RandomFlip': mmdet.datasets.pipelines.transforms.RandomFlip,
     'Pad': mmdet.datasets.pipelines.transforms.Pad,
     'Normalize': mmdet.datasets.pipelines.transforms.Normalize,
     'RandomCrop': mmdet.datasets.pipelines.transforms.RandomCrop,
     'SegRescale': mmdet.datasets.pipelines.transforms.SegRescale,
     'PhotoMetricDistortion': mmdet.datasets.pipelines.transforms.PhotoMetricDistortion,
     'Expand': mmdet.datasets.pipelines.transforms.Expand,
     'MinIoURandomCrop': mmdet.datasets.pipelines.transforms.MinIoURandomCrop,
     'Corrupt': mmdet.datasets.pipelines.transforms.Corrupt,
     'Albu': mmdet.datasets.pipelines.transforms.Albu,
     'RandomCenterCropPad': mmdet.datasets.pipelines.transforms.RandomCenterCropPad,
     'CutOut': mmdet.datasets.pipelines.transforms.CutOut,
     'GtBoxBasedCrop': mmdet.datasets.pipelines.transforms.GtBoxBasedCrop}




```python
config_mod.data['train']
```




    {'type': 'CocoDataset',
     'ann_file': 'F:/datasets/COCO2017/annotations/instances_train2017.json',
     'img_prefix': 'F:/datasets/COCO2017/train2017/',
     'pipeline': [{'type': 'LoadImageFromFile', 'to_float32': True},
      {'type': 'LoadAnnotations', 'with_bbox': True},
      {'type': 'PhotoMetricDistortion'},
      {'type': 'Resize',
       'img_scale': [(320, 320), (608, 608)],
       'keep_ratio': True}]}




```python
cocodataset=build_dataset(config_mod.data['train'])
```

    loading annotations into memory...
    Done (t=42.56s)
    creating index...
    index created!
    


```python
dataloader=build_dataloader(cocodataset,1,1)
```


```python
for batch in dataloader:
    print(batch)
    break
```


```python

```
