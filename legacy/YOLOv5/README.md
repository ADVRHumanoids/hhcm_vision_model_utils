# Legacy YOLOv5 Training

⚠️ **DEPRECATED**: This folder contains legacy YOLOv5 training code that is no longer maintained.

## Deprecation Notice

**Please use the current YOLO training tools instead:**
- **For object detection**: See [../../training/detection/](../../training/detection/)
- **For instance segmentation**: See [../../training/segmentation/yolo/](../../training/segmentation/yolo/)

The current implementation uses:
- **YOLO11** (latest Ultralytics version)
- Native Ultralytics API (no wrapper needed)
- Ray Tune + Optuna hyperparameter optimization
- Multi-GPU support
- Comprehensive logging and checkpointing

## Legacy Script

### TrainYolo.py

**Purpose**: Wrapper script for training YOLOv5 models (deprecated)

This script was a simple wrapper around the original YOLOv5 training script from Ultralytics. It has been replaced by direct use of the modern Ultralytics API with YOLO11.

**Arguments**:
- `--train_file`: Path to YOLOv5 train.py script (required)
- `--data_file`: YOLO dataset YAML file (required)
- `--batch`: Batch size (default: 1)
- `--epochs`: Training epochs (default: 3)
- `--weights`: Model weights variant (default: yolov5s6)
- `--img_size`: Image size (default: 1280)
- `--data_name`: Dataset name for output naming (default: laserSpots)

**Example Usage** (deprecated):
```bash
python TrainYolo.py \
    --train_file /path/to/yolov5/train.py \
    --data_file dataset.yaml \
    --batch 16 \
    --epochs 100 \
    --weights yolov5m6
```

## Advantages of Current Implementation

**YOLOv5 (Legacy)** → **YOLO11 (Current)**:
- ❌ External dependency → ✅ Native API
- ❌ Manual configuration → ✅ Built-in defaults
- ❌ No hyperparameter tuning → ✅ Ray Tune + Optuna
- ❌ Basic logging → ✅ TensorBoard + comprehensive metrics
- ❌ Manual multi-GPU setup → ✅ Automatic multi-GPU
- ❌ YOLOv5 architecture → ✅ YOLO11 (improved accuracy/speed)

## Why This Was Deprecated

1. **Direct API Access**: Modern Ultralytics provides direct Python API
2. **Better Architecture**: YOLO11 offers improved performance over YOLOv5
3. **Integrated Features**: Built-in hyperparameter optimization
4. **Maintainability**: No need for external YOLOv5 repository
5. **Segmentation Support**: Native instance segmentation in YOLO11

## References

- **Current YOLO Training**: [../../training/segmentation/yolo/](../../training/segmentation/yolo/)
- **Ultralytics Documentation**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **YOLOv5 Repository**: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
