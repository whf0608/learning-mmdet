# Build model from config in mmdete 


```python
from   mmdet.models.builder import build_detector
from   mmcv import Config
import torch
import mmcv
```


```python
!ls ./configs/yolo
```

    __pycache__  yolov3_d53_320_273e_coco.py	  yolov3.py
    README.md    yolov3_d53_mstrain-416_273e_coco.py  yolov5.py
    yolov3_1.py  yolov3_d53_mstrain-608_273e_coco.py



```python
from   tools.utils.config import  _get_config_directory
from   os.path import join,realpath,relpath
import glob
config_dpath = _get_config_directory()+"/yolo"
config_fpaths = list(glob.glob(join(config_dpath,'*.py')))
config_fpaths = [p for p in config_fpaths if p.find('_base_') == -1]
config_names = [relpath(p, config_dpath) for p in config_fpaths]
```


```python
config_names
```




    ['yolov5.py',
     'yolov3_d53_mstrain-608_273e_coco.py',
     'yolov3_d53_320_273e_coco.py',
     'yolov3.py',
     'yolov3_1.py',
     'yolov3_d53_mstrain-416_273e_coco.py']




```python
!cat ./configs/yolo/yolov5.py
```

    _base_ = '../_base_/default_runtime.py'
    # model settings
    model = dict(
        type='YOLOV5',
        pretrained='',
        backbone=dict(type='CSPDarknet',gd=1.0,gw=1.0,out_layers=[9,6,4]),
        neck=dict(
            type='YOLOV3Neck',
            num_scales=3,
            in_channels=[1024, 512, 256],
            out_channels=[512, 256, 128]),
        bbox_head=dict(
            type='YOLOV3Head',
            num_classes=80,
            in_channels=[512, 256, 128],
            out_channels=[1024, 512, 256],
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0,
                reduction='sum')
                ))
    
    # training and testing settings
    


### 1. Pares config yolov5.py 


```python
for config_fname in config_names[0:1]:
    config_fpath = join(config_dpath, config_fname)
    config_mod = Config.fromfile(config_fpath)
print(config_fname,'\n',config_fpath,'\n',config_mod)
```

    yolov5.py 
     /home/whf/Desktop/mmdetection/configs/yolo/yolov5.py 
     Config (path: /home/whf/Desktop/mmdetection/configs/yolo/yolov5.py): {'checkpoint_config': {'interval': 1}, 'log_config': {'interval': 50, 'hooks': [{'type': 'TextLoggerHook'}]}, 'dist_params': {'backend': 'nccl'}, 'log_level': 'INFO', 'load_from': None, 'resume_from': None, 'workflow': [('train', 1)], 'model': {'type': 'YOLOV5', 'pretrained': '', 'backbone': {'type': 'CSPDarknet', 'gd': 1.0, 'gw': 1.0, 'out_layers': [9, 6, 4]}, 'neck': {'type': 'YOLOV3Neck', 'num_scales': 3, 'in_channels': [1024, 512, 256], 'out_channels': [512, 256, 128]}, 'bbox_head': {'type': 'YOLOV3Head', 'num_classes': 80, 'in_channels': [512, 256, 128], 'out_channels': [1024, 512, 256], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}}}}



```python
for k,v in  config_mod.items():
    print(k,":",v)
```

    checkpoint_config : {'interval': 1}
    log_config : {'interval': 50, 'hooks': [{'type': 'TextLoggerHook'}]}
    dist_params : {'backend': 'nccl'}
    log_level : INFO
    load_from : None
    resume_from : None
    workflow : [('train', 1)]
    model : {'type': 'YOLOV5', 'pretrained': '', 'backbone': {'type': 'CSPDarknet', 'gd': 1.0, 'gw': 1.0, 'out_layers': [9, 6, 4]}, 'neck': {'type': 'YOLOV3Neck', 'num_scales': 3, 'in_channels': [1024, 512, 256], 'out_channels': [512, 256, 128]}, 'bbox_head': {'type': 'YOLOV3Head', 'num_classes': 80, 'in_channels': [512, 256, 128], 'out_channels': [1024, 512, 256], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}}}



```python
config_mod.model
```




    {'type': 'YOLOV5',
     'pretrained': '',
     'backbone': {'type': 'CSPDarknet',
      'gd': 1.0,
      'gw': 1.0,
      'out_layers': [9, 6, 4]},
     'neck': {'type': 'YOLOV3Neck',
      'num_scales': 3,
      'in_channels': [1024, 512, 256],
      'out_channels': [512, 256, 128]},
     'bbox_head': {'type': 'YOLOV3Head',
      'num_classes': 80,
      'in_channels': [512, 256, 128],
      'out_channels': [1024, 512, 256],
      'loss_cls': {'type': 'CrossEntropyLoss',
       'use_sigmoid': True,
       'loss_weight': 1.0,
       'reduction': 'sum'}}}




```python
if 'pretrained' in config_mod.model:
    config_mod.model['pretrained'] = None
```

### 2. Builder detector


```python
from mmdet.models import build_detector
model=build_detector(config_mod.model)
```


```python
print(model)
```

    YOLOV5(
      (backbone): CSPDarknet(
        (model): Sequential(
          (0): Focus(
            (conv): Conv(
              (conv): Conv2d(12, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
          )
          (1): Conv(
            (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (2): C3(
            (cv1): Conv(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (2): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
            )
          )
          (3): Conv(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (4): C3(
            (cv1): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (2): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (3): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (4): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (5): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (6): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (7): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (8): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
            )
          )
          (5): Conv(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (6): C3(
            (cv1): Conv(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (2): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (3): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (4): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (5): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (6): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (7): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (8): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
            )
          )
          (7): Conv(
            (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (8): SPP(
            (cv1): Conv(
              (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): ModuleList(
              (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
              (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
              (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
            )
          )
          (9): C3(
            (cv1): Conv(
              (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (2): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
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
     'VFNet': mmdet.models.detectors.vfnet.VFNet,
     'YOLACT': mmdet.models.detectors.yolact.YOLACT,
     'YOLOV3': mmdet.models.detectors.yolo.YOLOV3,
     'YOLOV4': mmdet.models.detectors.yolo.YOLOV4,
     'YOLOV5': mmdet.models.detectors.yolo.YOLOV5}




```python
# model=build_detector({'type':'YOLOV5','backbone':{}, 'neck':{},'bbox_head':{}})
```


```python
m=DETECTORS.get(config_mod.model['type'])
# print(m(backbone={},neck={},bbox_head={}))
```

### 2.1 Builder Backbone


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
     'SSDVGG': mmdet.models.backbones.ssd_vgg.SSDVGG,
     'CSPDarknet': mmdet.models.backbones.cspdarknet.CSPDarknet,
     'VGG16': mmdet.models.backbones.VGG16,
     'VGG9': mmdet.models.backbones.VGG9}



2.1.1 Use  backbone  builder


```python
backbone_config=config_mod.model['backbone']
print(backbone_config)
```

    {'type': 'CSPDarknet', 'gd': 1.0, 'gw': 1.0, 'out_layers': [9, 6, 4]}



```python
backone=build_backbone(config_mod.model['backbone'])
print(backone)
```

    CSPDarknet(
      (model): Sequential(
        (0): Focus(
          (conv): Conv(
            (conv): Conv2d(12, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (1): Conv(
          (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (2): C3(
          (cv1): Conv(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv3): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): Sequential(
            (0): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (1): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (2): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
          )
        )
        (3): Conv(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (4): C3(
          (cv1): Conv(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv3): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): Sequential(
            (0): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (1): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (2): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (3): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (4): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (5): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (6): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (7): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (8): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
          )
        )
        (5): Conv(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (6): C3(
          (cv1): Conv(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv3): Conv(
            (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): Sequential(
            (0): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (1): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (2): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (3): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (4): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (5): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (6): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (7): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (8): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
          )
        )
        (7): Conv(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (8): SPP(
          (cv1): Conv(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): ModuleList(
            (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
            (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
            (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
          )
        )
        (9): C3(
          (cv1): Conv(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv3): Conv(
            (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): Sequential(
            (0): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (1): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (2): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
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

    CSPDarknet



```python
backbone_config
```




    {'gd': 1.0, 'gw': 1.0, 'out_layers': [9, 6, 4]}




```python
bo(**backbone_config)
```




    CSPDarknet(
      (model): Sequential(
        (0): Focus(
          (conv): Conv(
            (conv): Conv2d(12, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (1): Conv(
          (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (2): C3(
          (cv1): Conv(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv3): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): Sequential(
            (0): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (1): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (2): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
          )
        )
        (3): Conv(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (4): C3(
          (cv1): Conv(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv3): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): Sequential(
            (0): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (1): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (2): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (3): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (4): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (5): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (6): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (7): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (8): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
          )
        )
        (5): Conv(
          (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (6): C3(
          (cv1): Conv(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv3): Conv(
            (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): Sequential(
            (0): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (1): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (2): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (3): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (4): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (5): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (6): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (7): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (8): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
          )
        )
        (7): Conv(
          (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (8): SPP(
          (cv1): Conv(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): ModuleList(
            (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
            (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
            (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
          )
        )
        (9): C3(
          (cv1): Conv(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv3): Conv(
            (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): Sequential(
            (0): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (1): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (2): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
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
     'FPN': mmdet.models.necks.fpn.FPN,
     'FPN_CARAFE': mmdet.models.necks.fpn_carafe.FPN_CARAFE,
     'HRFPN': mmdet.models.necks.hrfpn.HRFPN,
     'NASFPN': mmdet.models.necks.nas_fpn.NASFPN,
     'NASFCOS_FPN': mmdet.models.necks.nasfcos_fpn.NASFCOS_FPN,
     'PAFPN': mmdet.models.necks.pafpn.PAFPN,
     'RFP': mmdet.models.necks.rfp.RFP,
     'YOLOV3Neck': mmdet.models.necks.yolo_neck.YOLOV3Neck}




```python

```
