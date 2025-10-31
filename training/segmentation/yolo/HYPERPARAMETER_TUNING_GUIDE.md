# Hyperparameter Tuning Technical Guide

**Script**: `tune_yolo11_seg.py`
**Purpose**: Automated hyperparameter optimization for YOLOv11 segmentation models
**Framework**: Ray Tune + Optuna + ASHA Scheduler + Trial Plateau Stopper + Weights & Biases

---

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Search Space Configuration](#search-space-configuration)
4. [Weights & Biases Logging](#weights--biases-logging)
5. [Customization Guide](#customization-guide)
6. [Troubleshooting](#troubleshooting)

---

## Overview

This script performs **automated hyperparameter optimization** using a combination of powerful tools:

- **Ray Tune**: Distributed hyperparameter tuning framework
- **Optuna**: Bayesian optimization search algorithm with initial point seeding
- **ASHA Scheduler**: Early stopping for unpromising trials
- **Trial Plateau Stopper**: Detects convergence and stops plateaued trials
- **Weights & Biases**: Experiment tracking and visualization

### Workflow

```
┌─────────────┐
│  Ray Tune   │  Orchestrates the entire tuning process
│  (Master)   │  • Manages trial distribution
└──────┬──────┘  • Tracks results
       │         • Selects best configurations
       ↓
┌─────────────┐
│   Optuna    │  Suggests hyperparameter combinations
│  (Sampler)  │  • Starts with known good points (optional)
└──────┬──────┘  • Uses Bayesian optimization
       │         • Learns from previous trials
       ↓
┌─────────────┐
│    ASHA     │  Decides which trials to stop early
│ (Scheduler) │  • Compares trials against each other
└──────┬──────┘  • Terminates poor performers
       │
       ↓
┌─────────────┐
│   Plateau   │  Detects when trials have converged
│   Stopper   │  • Monitors training F1 stability
└──────┬──────┘  • Stops if no improvement
       │
       ↓
┌─────────────┐
│ train_yolo()│  Executes individual training trials
│   (Worker)  │  • Trains YOLO with given hyperparameters
└──────┬──────┘  • Reports metrics per epoch
       │         • Logs to W&B with train+val metrics
       ↓
┌─────────────┐
│   Weights   │  Tracks and visualizes experiments
│  & Biases   │  • Stores metrics, curves, models
└─────────────┘  • Enables comparison across trials
```

---

## Core Components

### 1. Ray Tune

**What it is**: Distributed hyperparameter tuning framework from Ray ecosystem.

**Role in this script**:
- **Orchestrates** the entire tuning process
- **Distributes** trials across available compute resources
- **Tracks** all trial results and metrics
- **Selects** the best hyperparameter configuration

**Key concepts**:
```python
# Initialize Ray with resource allocation
ray.init(num_gpus=1, num_cpus=8)

# Define trainable function with resources
trainable_with_cpu_gpu = tune.with_resources(
    train_yolo,
    {"gpu": (1 / args.concurrent)}  # Fraction of GPU per trial
)

# Configure tuner
tuner = tune.Tuner(
    trainable_with_cpu_gpu,
    tune_config=tune.TuneConfig(
        scheduler=asha_scheduler,
        search_alg=algo,
        num_samples=50,
    ),
    param_space=search_space,
    run_config=tune.RunConfig(
        stop=stopper,
        callbacks=[CleanupCallback()],
    ),
)
```

**Important parameters to modify**:
- `num_gpus`: Number of GPUs available (line 382)
- `num_cpus`: Number of CPU cores available (line 382)
- `{"gpu": (1 / args.concurrent)}`: GPU fraction per trial (line 665)
  - If concurrent=2 and 1 GPU → each trial gets 0.5 GPU
- `num_samples`: Total number of trials (line 671)

### 2. Optuna with Initial Point Seeding

**What it is**: Bayesian optimization library using Tree-structured Parzen Estimator (TPE).

**Role in this script**:
- **Starts** with known good hyperparameter configurations (optional)
- **Suggests** which hyperparameter combinations to try next
- **Learns** from previous trial results to make better suggestions
- **Balances** exploration (trying new areas) vs exploitation (refining good areas)

**How it works**:
1. **Initial Phase**: Evaluates provided `points_to_evaluate` first
2. **Learning Phase**: Builds a probabilistic model of hyperparameter → performance relationship
3. **Optimization Phase**: Suggests hyperparameters that maximize expected improvement

**Initial Search Space (Warm Start)** (lines 643-646):

```python
optuna_search = OptunaSearch(
    metric="metrics/F1(M)",
    mode="max",
    points_to_evaluate=[initial_space, initial_space_b8, initial_space_b2]
)
```

**What this does**:
- **First 3 trials** evaluate the provided initial configurations
- Optuna learns from these "warm start" points
- Subsequent trials explore around these known-good regions
- **Faster convergence** compared to pure random search

**Benefits**:
- **Jump-start optimization** with previously found good hyperparameters
- **Avoid wasting trials** on obviously bad configurations
- **Reproducible experiments** - can restart optimization from same baseline
- **Transfer learning** - use hyperparameters from related datasets

**How to customize** (lines 432-607 define initial configurations):
```python
# Define initial configuration based on previous experiments
my_initial_config = {
    "lr0": 0.0001,        # From previous successful run
    "lrf": 0.3,
    "momentum": 0.9,
    "weight_decay": 0.0002,
    "warmup_epochs": 2,
    # ... include ALL hyperparameters in search_space
    "batch": 16,
}

# Can provide 1-5 initial points
optuna_search = OptunaSearch(
    metric="metrics/F1(M)",
    mode="max",
    points_to_evaluate=[my_initial_config]  # List of dicts
)
```

**Best practices**:
- Include 1-3 initial configurations (more = longer startup)
- Use results from previous successful runs on similar datasets
- All hyperparameters must be specified (match search_space keys)
- Can use default YOLO hyperparameters as baseline

**Key characteristics**:
- **Intelligent sampling**: Later trials benefit from earlier results
- **Handles various hyperparameter types**: Continuous, discrete, categorical
- **Robust to noise**: Can handle stochastic training processes
- **Efficient**: Typically finds good hyperparameters in fewer trials than grid/random search

**When to use Optuna vs other search algorithms**:
- ✅ **Use Optuna** when: You have limited compute budget, want intelligent exploration
- ❌ **Use Random Search** when: Very noisy objectives, initial exploration phase
- ❌ **Use Grid Search** when: Few hyperparameters, need exhaustive coverage

### 3. ASHA Scheduler

**What it is**: Asynchronous Successive Halving Algorithm - an early stopping scheduler.

**Role in this script**:
- **Monitors** trial performance throughout training
- **Terminates** poorly performing trials early (compared to other trials)
- **Saves** computational resources by not training bad configurations to completion
- **Promotes** promising trials to continue training

**How it works**:

```
Epoch 0  ─────────────────────────────────────────────
          Trial 1, 2, 3, 4, 5, 6, 7, 8 all start

Epoch 25 ─────────────────────────────────────────────  ← Grace Period
          All trials evaluated
          Bottom 50% stopped (reduction_factor=2)
          ✗ Trial 2, 4, 5, 7 stopped
          ✓ Trial 1, 3, 6, 8 continue

Epoch 50 ─────────────────────────────────────────────
          Remaining trials evaluated
          Bottom 50% of survivors stopped
          ✗ Trial 3, 8 stopped
          ✓ Trial 1, 6 continue

Epoch 75+─────────────────────────────────────────────
          Best trials train to completion
```

**Current configuration** (lines 648-656):
```python
asha_scheduler = ASHAScheduler(
    time_attr='epoch',
    metric='metrics/F1(M)',
    mode='max',
    max_t=search_space["training_epochs"] + 1,  # 251 epochs so YOLO handles the stopping and trial is not pruned
    grace_period=25,      # Minimum epochs before first evaluation
    reduction_factor=2,   # Keep top 50% at each rung
    brackets=1,           # Suggested from docs
)
```

**Parameters explained**:
- **`grace_period=25`**: Every trial gets at least 25 epochs to prove itself
  - 10% of 250 max epochs
  - Enough time for model to show potential
- **`reduction_factor=2`**: At each rung, keep top 50% of trials
  - Balanced pruning strategy
- **`max_t=251`**: Maximum epochs a trial can run
- **`brackets=1`**: Single bracket (standard ASHA)

**Benefits**:
- **10-20x speedup** compared to running all trials to completion
- **More exploration** with same budget (can try more hyperparameter combinations)
- **Automatic** - no manual intervention needed

**To modify stopping behavior**:
- Increase `grace_period` if model needs more epochs to stabilize (e.g., 40)
- Decrease `reduction_factor` for more aggressive early stopping (e.g., 3 = top 33%)
- Increase `reduction_factor` to give trials more chances (e.g., 1.5 = top 67%)

### 4. Trial Plateau Stopper (Convergence Detection)

**What it is**: Stops trials when training has plateaued (no improvement).

**Difference from ASHA**:
- **ASHA**: Compares trials against each other (relative performance)
- **Plateau Stopper**: Monitors individual trial convergence (absolute criterion)

**Current implementation** (lines 657-662):
```python
stopper = TrialPlateauStopper(
    metric="train_seg_metrics/F1",  # Monitors TRAINING F1 (not validation)
    std=0.01,                        # Standard deviation threshold
    num_results=5,                   # Look at last 5 epochs
    grace_period=10,                 # Minimum 10 epochs before evaluating
)
```

**How it works**:
1. Waits for at least `grace_period` (10) epochs
2. Monitors last `num_results` (5) epochs of training F1
3. Calculates standard deviation across those 5 epochs
4. If std < 0.01 → **training has plateaued** → stop trial
5. Works **independently** from ASHA

**Example**:
```
Epoch 10-14: F1 = [0.75, 0.76, 0.77, 0.78, 0.79]  → std=0.016 → continue
Epoch 50-54: F1 = [0.85, 0.851, 0.849, 0.850, 0.851] → std=0.0008 → STOP!
```

> **Note** <br>
I am not sure training metrics should be used for this kind of stop. <br>
Ideally we should check the validation loss, but in my case was never satisfying the criteria so I decided to prune the trial when the model had reached the perfect fitting (ideally, but never happened).


**Why monitor training F1 instead of validation F1**:
- **Training metrics** are more stable (less noisy)
- Better indicator of whether model is still learning
- Detects **convergence** rather than overfitting

**Benefits**:
- **Saves time** on trials that have converged early
- **Prevents wasting resources** after learning has saturated
- **Complements ASHA**: ASHA prunes poor trials, Plateau Stopper prunes converged trials
- **Automatic convergence detection**

**Parameters to modify**:
```python
stopper = TrialPlateauStopper(
    metric="train_seg_metrics/F1",
    std=0.01,          # Stricter: 0.005, More lenient: 0.02
    num_results=5,     # Longer window: 10, Shorter: 3
    grace_period=10,   # More warmup: 20, Less: 5
)
```

**When to adjust**:
- **Increase `std`** (0.02) if trials stop too early while still improving
- **Decrease `std`** (0.005) if trials waste time after convergence
- **Increase `num_results`** (10) to require longer plateau before stopping
- **Increase `grace_period`** (20) if model has slow start

**To disable**:
```python
# In run_config
run_config=tune.RunConfig(
    stop=None,  # Remove stopper, rely only on ASHA
    callbacks=[CleanupCallback()],
)
```

---

## Search Space Configuration

The search space defines which hyperparameters to tune and their ranges. Located in `tune_yolo11_seg.py` starting at line 294.

### Hyperparameter Categories

#### 1. Learning Rate Dynamics

```python
"lr0": tune.loguniform(1e-5, 1e-1),     # Initial learning rate
"lrf": tune.uniform(0.01, 1.0),         # Final LR factor
"momentum": tune.uniform(0.6, 0.98),    # SGD/Adam momentum
"weight_decay": tune.uniform(0.0, 0.005),  # L2 regularization
"optimizer": tune.choice(["AdamW"]),    # Fixed to AdamW
```

**What they do**:
- `lr0`: Starting learning rate - higher = faster convergence but risk overshooting
- `lrf`: Learning rate decay - lower = more decay by end (lr_final = lr0 * lrf)
- `momentum`: Accelerates optimizer in relevant direction
- `weight_decay`: Prevents overfitting via L2 regularization

**When to modify**:
- **Widen `lr0` range** if uncertain about optimal learning rate
- **Try other optimizers** by adding to choice: `["AdamW", "SGD", "Adam"]`

#### 2. Warmup Strategy

```python
"warmup_epochs": tune.randint(0, 5),  # Random integer 0-4
"warmup_momentum": tune.uniform(0.00, 0.95),
```

**What they do**:
- Gradually increase learning rate over first N epochs
- Prevents unstable gradients early in training
- Lower momentum during warmup for stability

#### 3. Loss Component Weights

```python
"box": tune.uniform(5.0, 8.0),   # Or fixed with tune.choice([6.36])
"cls": tune.uniform(1.0, 3.0),   # Or fixed with tune.choice([1.42])
"dropout": tune.uniform(0.0, 0.1),
```

**What they do**:
- Balance importance of different loss components
- Higher weight = model focuses more on that aspect

#### 4. Data Augmentation (Tunable in temporary script)

```python
"fliplr": tune.uniform(0.0, 0.5),      # Horizontal flip probability
"hsv_h": tune.uniform(0.0, 0.5),       # Hue shift range
"hsv_s": tune.uniform(0.0, 0.5),       # Saturation shift
"hsv_v": tune.uniform(0.0, 0.5),       # Value shift
"degrees": tune.uniform(0.0, 10.0),    # Rotation range in degrees
"translate": tune.uniform(0.0, 0.25),  # Translation range
"scale": tune.uniform(0.0, 0.5),       # Scale range
"perspective": tune.uniform(0.0, 0.1), # Perspective transform
"mosaic": tune.uniform(0.7, 1.0),      # Mosaic probability
"copy_paste": tune.uniform(0.0, 0.3),  # Copy-paste probability
"cutmix": tune.uniform(0.0, 0.3),      # Cutmix probability
```

**Note**: Some augmentations are disabled in original script but enabled in temporary version.

#### 5. Batch Size

```python
"batch": tune.choice([16, 32, 64])  # Batch size options
```

**What it does**:
- Number of images processed together
- Larger batch = more stable gradients, faster training, more memory
- Smaller batch = more noise, better generalization, less memory

**When to modify**:
- **Add smaller batches** (8, 12) if GPU memory limited
- **Add larger batches** (128, 256) if you have high-end GPU
- **Remove options** that cause OOM errors

### Distribution Types

Ray Tune supports various distribution types for search:

```python
# Continuous uniform: [a, b]
tune.uniform(0.0, 1.0)       # Linear scale

# Log-uniform: Better for learning rates
tune.loguniform(1e-6, 1e-2)  # Logarithmic scale

# Integer uniform
tune.randint(1, 100)         # Random integers

# Categorical choice
tune.choice(["a", "b", "c"]) # Select from list

# Grid search (evaluates all)
tune.grid_search([1, 2, 3])  # Not recommended with Optuna
```

**Best practices**:
- Use `loguniform` for learning rates, weight decay (span multiple orders of magnitude)
- Use `uniform` for probabilities, weights (0-1 range)
- Use `choice` for categorical (optimizer types, true/false)
- Avoid `grid_search` with Optuna (defeats Bayesian optimization)

---

## Weights & Biases Logging

W&B tracks two types of information: **metrics** (per epoch) and **artifacts** (files/models).

### Metric Groups

#### 1. Segmentation Metrics (Validation) (`seg_metrics/`)

```python
"seg_metrics/F1": metrics.get("metrics/F1(M)")              # F1 score for masks
"seg_metrics/precision": metrics.get("metrics/precision(M)") # Mask precision
"seg_metrics/recall": metrics.get("metrics/recall(M)")       # Mask recall
"seg_metrics/map50_95": metrics.get("metrics/mAP50-95(M)")  # mAP at IoU 0.50-0.95
"seg_metrics/map50": metrics.get("metrics/mAP50(M)")        # mAP at IoU 0.50
```

**What they mean**:
- **F1**: Harmonic mean of precision and recall (0-1, higher better) - **OPTIMIZED METRIC**
- **Precision**: Of predicted masks, what fraction are correct?
- **Recall**: Of ground truth masks, what fraction are detected?
- **mAP50**: Mean Average Precision at 50% IoU threshold
- **mAP50-95**: Average of mAP at IoU thresholds 0.50 to 0.95

#### 2. Box Metrics (Validation) (`box_metrics/`)

```python
"box_metrics/F1": metrics.get("metrics/F1(B)")
"box_metrics/precision": metrics.get("metrics/precision(B)")
"box_metrics/recall": metrics.get("metrics/recall(B)")
"box_metrics/map50_95": metrics.get("metrics/mAP50-95(B)")
"box_metrics/map50": metrics.get("metrics/mAP50(B)")
```

Same metrics but for bounding boxes only. Usually higher than mask metrics.

#### 3. Training Metrics
Same as `Segmentation Metrics` and `Box Metrics` but evaluated on the `train` split of the dataset (the one the model uses to adjust weights).

```python
"train_seg_metrics/F1": train_f1_seg                  # Training F1 for masks
"train_seg_metrics/precision": train_mask_precision
"train_seg_metrics/recall": train_mask_recall
"train_seg_metrics/map50_95": ...
"train_seg_metrics/map50": ...

"train_box_metrics/F1": train_f1_box                  # Training F1 for boxes
"train_box_metrics/precision": train_box_precision
"train_box_metrics/recall": train_box_recall
"train_box_metrics/map50_95": ...
"train_box_metrics/map50": ...
```

**What they mean**:
- **Same metrics as validation** but computed on training data
- Helps detect **overfitting** by comparing train vs validation
- **Used by Plateau Stopper** to detect convergence

**Overfitting detection**:
```
Good:     train_F1=0.85, val_F1=0.82  (3% gap - normal)
Warning:  train_F1=0.90, val_F1=0.75  (15% gap - overfitting)
Severe:   train_F1=0.95, val_F1=0.60  (35% gap - severe overfitting)
```

**In W&B dashboard**:
- Create custom chart: plot `train_seg_metrics/F1` vs `seg_metrics/F1`
- Monitor gap over epochs
- Increasing gap = overfitting

#### 4. Validation Loss Components (`val/`)

```python
"val/box_loss": metrics.get("val/box_loss")      # Bounding box regression loss
"val/seg_loss": metrics.get("val/seg_loss")      # Segmentation mask loss
"val/cls_loss": metrics.get("val/cls_loss")      # Classification loss
"val/dfl_loss": metrics.get("val/dfl_loss")      # Distribution Focal Loss
```

**What they mean**:
- **box_loss**: How well bounding boxes match ground truth
  - Lower = better localization
- **seg_loss**: How well segmentation masks match ground truth
  - Lower = better mask quality
- **cls_loss**: How well object classes are classified
  - Lower = better classification
- **dfl_loss**: Distribution Focal Loss for bounding box refinement
  - Advanced loss for fine-grained localization

**Monitoring during training**:
- All losses should **decrease** over epochs
- If losses **plateau early**, may need different learning rate
- If losses **oscillate**, batch size or learning rate may be too high
- If **one loss dominates** others, adjust loss weights (box, cls parameters)

#### 5. Training Loss Components (`train/`)

```python
"train/box": float(arr[i])   # Training box loss
"train/seg": float(arr[i])   # Training seg loss
"train/cls": float(arr[i])   # Training cls loss
"train/dfl": float(arr[i])   # Training DFL loss
```

**What they mean**:
- Same as validation losses but on training set
- Usually lower than validation losses (model has seen training data)
- **Large gap** between train and val losses = overfitting

### Artifacts Logged

#### 1. Model Artifacts

```python
# Best model from trial
model_artifact = wandb.Artifact(
    name=f"best_model_trial_{run_id}",
    type="model"
)
model_artifact.add_file(str(best_model_path), name="best.pt")
```

**Contains**: Best model weights from training (lowest validation loss)

**How to download**:
```python
import wandb
run = wandb.init()
artifact = run.use_artifact('your-project/best_model_trial_xxx:latest')
artifact_dir = artifact.download()
```

#### 2. Training Curves

```python
# Various training visualizations
curve_files = [
    "BoxF1_curve.png",              # Box F1 over epochs
    "BoxP_curve.png",               # Box Precision over epochs
    "BoxPR_curve.png",              # Box Precision-Recall curve
    "BoxR_curve.png",               # Box Recall over epochs
    "MaskF1_curve.png",             # Mask F1 over epochs
    "MaskP_curve.png",              # Mask Precision over epochs
    "MaskPR_curve.png",             # Mask Precision-Recall curve
    "MaskR_curve.png",              # Mask Recall over epochs
    "confusion_matrix.png",         # Class confusion matrix
    "confusion_matrix_normalized.png",  # Normalized confusion matrix
    "results.png",                  # All metrics combined
    "results.csv",                  # Metrics in CSV format
    "labels.jpg"                    # Dataset label distribution
]
```

**Use cases**:
- **F1 curves**: Monitor convergence and detect overfitting
- **PR curves**: Understand precision-recall tradeoff at different thresholds
- **Confusion matrix**: See which classes are confused with each other
- **results.png**: Quick overview of all metrics


---

## Customization Guide

### Change the Optimization Objective

**Current**: Optimize segmentation F1 score (`metrics/F1(M)`)

**To change to box F1**:
```python
optuna_search = OptunaSearch(
    metric="metrics/F1(B)",  # Changed
    mode="max"
)

asha_scheduler = ASHAScheduler(
    metric='metrics/F1(B)',  # Changed
    mode='max',
    # ...
)
```

### Modify Initial Points

```python
# Add your own initial configurations
my_config_1 = {
    "lr0": 0.0001,
    # ... all hyperparameters
}

my_config_2 = {
    "lr0": 0.0002,
    # ... all hyperparameters
}

optuna_search = OptunaSearch(
    metric="metrics/F1(M)",
    mode="max",
    points_to_evaluate=[my_config_1, my_config_2]
)
```

### Adjust Trial Stopping Criteria

**Make ASHA more aggressive** (stop poor trials earlier):
```python
asha_scheduler = ASHAScheduler(
    grace_period=15,        # Reduce from 25
    reduction_factor=3,     # Increase from 2 (keep top 33%)
    # ...
)
```

**Make Plateau Stopper more sensitive**:
```python
stopper = TrialPlateauStopper(
    metric="train_seg_metrics/F1",
    std=0.005,        # Stricter (was 0.01)
    num_results=3,    # Shorter window (was 5)
    grace_period=10,
)
```

---

## Troubleshooting

### Issue: Trials Stopping Too Early (ASHA)

**Solutions**:
```python
asha_scheduler = ASHAScheduler(
    grace_period=40,        # Increase
    reduction_factor=2,     # Keep same or reduce
)
```

### Issue: Trials Not Converging (Plateau Stopper Too Strict)

**Solutions**:
```python
stopper = TrialPlateauStopper(
    std=0.02,         # More lenient (was 0.01)
    num_results=10,   # Longer window (was 5)
    grace_period=20,  # More warmup (was 10)
)
```

### Issue: CUDA Out of Memory

**Solutions**:
1. Reduce batch sizes in search space:
   ```python
   "batch": tune.choice([2, 4])  # Remove larger batches
   ```

2. Reduce concurrent trials:
   ```bash
   python tune_yolo11_seg.py --concurrent 1
   ```

3. Use smaller model:
   ```bash
   python tune_yolo11_seg.py --model yolo11n-seg.pt
   ```

---

## Quick Start Checklist

- [ ] Install dependencies: `pip install ray[tune] optuna wandb ultralytics`
- [ ] Login to W&B: `wandb login`
- [ ] Prepare YOLO dataset with data.yaml
- [ ] Define hyperparameters search space
- [ ] (Optional) Define initial hyperparameter configurations
- [ ] Run the script (example: `python tune_yolo11_seg.py --data path/to/data.yaml --project my_project --concurrent 2`)
- [ ] Monitor in W&B dashboard
- [ ] Extract best hyperparameters
- [ ] Retrain final model with best config


---

## Additional Resources

- **Ray Tune Documentation**: https://docs.ray.io/en/latest/tune/index.html
- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Weights & Biases Docs**: https://docs.wandb.ai/
- **YOLO Ultralytics Docs**: https://docs.ultralytics.com/
- **TrialPlateauStopper API**: https://docs.ray.io/en/latest/tune/api/stoppers.html
