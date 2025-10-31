# YOLO Segmentation Training

This folder contains training, evaluation, and hyperparameter optimization scripts for YOLOv11 segmentation models. Supports basic training workflows and advanced hyperparameter search using Ray Tune with Weights & Biases integration.

## Scripts

### train_yolov11.py

Basic training script for YOLOv11 segmentation models with simple command-line interface.

**Arguments**:
- `--weights` (str, default: "yolo11m-seg") - YOLO model variant to use as starting weights
- `--data_path` (str, default: "/home/tori/YOLO/data/") - Root directory for dataset files
- `--data_file` (str, default: "coco_lasers_nicla_combo_640") - Dataset YAML file name (without extension)
- `--batch` (int, default: 16) - Training batch size
- `--epochs` (int, default: 500) - Number of training epochs
- `--img_size` (int, default: 640) - Input image size for training

**Example**:
```bash
python3 train_yolov11.py \
    --weights yolo11m-seg \
    --data_path /path/to/data/ \
    --data_file my_dataset \
    --batch 16 \
    --epochs 100 \
    --img_size 640
```

**Output**: Creates timestamped run directory in `runs/segment/` with trained model weights and training logs.

### tune_yolo11_seg.py

Hyperparameter optimization script using Ray Tune with Optuna search and ASHA scheduling.

📖 **For detailed explanation of Ray Tune, Optuna, ASHA, and W&B logging, see: [HYPERPARAMETER_TUNING_GUIDE.md](./HYPERPARAMETER_TUNING_GUIDE.md)**

**Arguments**:
- `--model` (str, default: "yolo11m-seg.pt") - YOLO model variant
- `--data` (str, required) - Path to dataset YAML file
- `--project` (str, default: "yolo_tune") - W&B project name
- `--name` (str, default: "hp_search") - W&B run name prefix
- `--num_samples` (int, default: 20) - Number of hyperparameter trials
- `--max_epochs` (int, default: 50) - Maximum epochs per trial
- `--gpus_per_trial` (float, default: 1.0) - GPU allocation per trial
- `--cpus_per_trial` (int, default: 4) - CPU cores per trial

**Example**:
```bash
python3 tune_yolo11_seg.py \
    --model yolo11m-seg.pt \
    --data /path/to/dataset.yaml \
    --project my_optimization \
    --name experiment_1 \
    --num_samples 50 \
    --max_epochs 30
```

**Hyperparameters Optimized**:
- Learning rate (lr0, lrf)
- Optimizer parameters (momentum, weight_decay)
- Loss weights (box, cls, dfl)
- Data augmentation (HSV, rotation, translation, scale, mosaic, mixup, copy-paste)
- Training parameters (warmup_epochs, batch size)

**Output**:
- Ray Tune results in `~/ray_results/`
- W&B logs for each trial
- Best hyperparameters reported at completion

### eval.py

Evaluation and visualization script for YOLOv11 segmentation models.

**Arguments**:
- `--model` (str, required) - Path to trained YOLO model (.pt file)
- `--data` (str, required) - Path to dataset YAML file
- `--split` (str, default: "val") - Dataset split to evaluate (val, test, train)
- `--num_samples` (int, optional) - Number of random images to visualize (if not specified, evaluates all)

**Example**:
```bash
# Evaluate on validation set with visualization
python3 eval.py \
    --model runs/segment/exp/weights/best.pt \
    --data /path/to/dataset.yaml \
    --split val \
    --num_samples 10
```

**Output**:
- Displays ground truth and predicted masks side-by-side
- Color-coded segmentation masks with class labels
- OpenCV windows for interactive visualization

## Dependencies

- ultralytics (YOLO)
- torch
- ray[tune] (for hyperparameter optimization)
- optuna (search algorithm)
- wandb (experiment tracking)
- opencv-python (cv2)
- matplotlib
- numpy
- pyyaml

Install with:
```bash
pip install ultralytics torch "ray[tune]" optuna wandb opencv-python matplotlib numpy pyyaml
```

## YOLO Model Variants

Supported YOLOv11 segmentation models (specify via `--weights` or `--model`):
- `yolo11n-seg` - Nano (fastest, least accurate)
- `yolo11s-seg` - Small
- `yolo11m-seg` - Medium (balanced)
- `yolo11l-seg` - Large
- `yolo11x-seg` - Extra Large (slowest, most accurate)

## Dataset Format

Expected YOLO dataset structure referenced in YAML file:

```
dataset/
├── dataset.yaml          # Dataset configuration
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/         # YOLO format segmentation labels
    ├── val/
    └── test/
```

**dataset.yaml** format:
```yaml
path: /path/to/dataset  # Dataset root
train: images/train
val: images/val
test: images/test

names:
  0: class1
  1: class2
  # ...
```

## Training Workflows

### Basic Training
Use `train_yolov11.py` for quick training with default or custom hyperparameters:
1. Prepare YOLO format dataset with YAML config
2. Run training script with desired arguments
3. Monitor training in terminal and `runs/segment/` directory
4. Best weights saved automatically

### Hyperparameter Optimization
Use `tune_yolo11_seg.py` for systematic hyperparameter search:
1. Set up Weights & Biases account and login (`wandb login`)
2. Prepare dataset and configure search space (edit script if needed)
3. Run tuning script with desired number of trials
4. Monitor experiments in W&B dashboard
5. Extract best hyperparameters from Ray Tune results
6. Use best hyperparameters for final training run

### Evaluation
Use `eval.py` to visualize and assess model performance:
1. Train model using either workflow above
2. Run evaluation script with model path and dataset
3. Visually inspect ground truth vs predictions
4. Assess segmentation quality for debugging/improvement

## Integration with Weights & Biases

The tuning script logs comprehensive metrics to W&B:
- **Segmentation metrics**: Precision, Recall, mAP50, mAP50-95, F1
- **Box metrics**: Precision, Recall, mAP50, mAP50-95, F1
- **Loss components**: Box loss, Segmentation loss, Classification loss, DFL loss
- **Training metrics**: Per-epoch loss breakdown

Access logs at: https://wandb.ai/{your-entity}/{project-name}

**For detailed explanation of all logged metrics and what they mean, see the [Hyperparameter Tuning Guide](./HYPERPARAMETER_TUNING_GUIDE.md#weights--biases-logging)**

## Tips

- Start with `train_yolov11.py` for initial baseline training
- Use smaller model variants (nano, small) for faster iteration
- Increase batch size for larger GPUs (proportional to VRAM)
- Use `tune_yolo11_seg.py` only after confirming baseline training works
- Monitor GPU memory usage with `nvidia-smi` during training
- For production models, retrain with best hyperparameters from tuning for full epoch count

## Additional Documentation

- **[HYPERPARAMETER_TUNING_GUIDE.md](./HYPERPARAMETER_TUNING_GUIDE.md)**: Comprehensive technical guide covering:
  - Ray Tune orchestration and resource management
  - Optuna Bayesian optimization algorithm
  - ASHA early stopping scheduler
  - Trial stopper configuration (for advanced use)
  - Complete search space customization guide
  - Detailed W&B metrics explanation with typical values
  - Troubleshooting common issues
  - Step-by-step customization examples


## Future Implementations

- [ ] Load Suggested initial hyperparameters from YAML
- [ ] Load hyperparameters search space from YAML
- [ ] In evaluation script, if ground truth annotations are available, display metrics fo each image/loaded dataset