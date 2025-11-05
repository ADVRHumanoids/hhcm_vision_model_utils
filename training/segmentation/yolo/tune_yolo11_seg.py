#!/usr/bin/env python3
"""
Hyperparameter optimization script for YOLOv11 segmentation using Ray Tune.

Performs hyperparameter search for YOLOv11 segmentation models using Ray Tune
with Optuna search algorithm and ASHA scheduling. Integrates with Weights & Biases
for experiment tracking and provides custom callbacks for detailed metric logging.

Created by: Alessio Lovato
Modified by: Alessio Lovato, 31-10-2025
"""

import os
import gc
import torch
import argparse
import wandb
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ultralytics import YOLO

# Custom callback to capture training metrics per epoch
def on_train_epoch_end(trainer):
    """
    Custom callback for logging training metrics at end of each epoch.

    Args:
        trainer: Ultralytics YOLO trainer object with training state
    """
    metrics = {}
    # Log training loss components if available
    try:
        tloss = getattr(trainer, "tloss", None)
        loss_names = getattr(trainer, "loss_names", None)
        if tloss is not None and loss_names is not None:
            # Convert tloss tensor to a NumPy array or list of floats
            try:
                loss_vals = tloss.cpu().numpy()

            except:
                try:
                    loss_vals = list(tloss)
                except:
                    loss_vals = None
            if loss_vals is not None:
                import numpy as _np
                # Ensure it's an array for summing
                arr = _np.array(loss_vals, dtype=float).ravel()
                # Log each loss component (box, seg, cls, dfl)
                for i, name in enumerate(loss_names):
                    if i < arr.size:
                        metrics[f"train/{name}"] = float(arr[i])
    except Exception:
        pass

    if metrics:
        wandb.log(metrics, step=trainer.epoch + 1)

def patched_on_fit_epoch_end(trainer):
    """
    Patched callback for Ray Tune integration with YOLO training.

    Replaces default on_fit_epoch_end to compute and report custom metrics
    including F1 scores for both box and segmentation tasks. Reports metrics
    to Ray Tune for hyperparameter optimization.

    Args:
        trainer: Ultralytics YOLO trainer object with training metrics

    Original function: https://docs.ultralytics.com/reference/utils/callbacks/raytune/#ultralytics.utils.callbacks.raytune.on_fit_epoch_end
    """
        
    metrics = trainer.metrics
    # Compute F1 score for segmentation
    mask_precision = metrics.get("metrics/precision(M)", 0.0)
    mask_recall = metrics.get("metrics/recall(M)", 0.0)
    box_precision = metrics.get("metrics/precision(B)", 0.0)
    box_recall = metrics.get("metrics/recall(B)", 0.0)

    # Compute F1 for segmentation with safe division
    if mask_precision + mask_recall > 0:
        f1_seg = 2 * (mask_precision * mask_recall) / (mask_precision + mask_recall)
    else:
        f1_seg = 0.0

    # Compute F1 for bounding boxes with safe division
    if box_precision + box_recall > 0:
        f1_box = 2 * (box_precision * box_recall) / (box_precision + box_recall)
    else:
        f1_box = 0.0
    
    # Add F1 to reported metrics
    metrics["F1(M)"] = f1_seg
    metrics["F1(B)"] = f1_box

    tune.report({**metrics, **{"epoch": trainer.epoch + 1}})

    # Prepare W&B metrics
    wandb_metrics = {
        "seg_metrics/F1": metrics.get("F1(M)", 0.0),
        "seg_metrics/precision": metrics.get("metrics/precision(M)", 0.0),
        "seg_metrics/recall": metrics.get("metrics/recall(M)", 0.0),
        "seg_metrics/map50_95": metrics.get("metrics/mAP50-95(M)", 0.0),
        "seg_metrics/map50": metrics.get("metrics/mAP50(M)", 0.0),
        "box_metrics/precision": metrics.get("metrics/precision(B)", 0.0),
        "box_metrics/recall": metrics.get("metrics/recall(B)", 0.0),
        "box_metrics/map50_95": metrics.get("metrics/mAP50-95(B)", 0.0),
        "box_metrics/map50": metrics.get("metrics/mAP50(B)", 0.0),
        "box_metrics/F1": metrics.get("F1(B)", 0.0),
        "val/box_loss": metrics.get("val/box_loss", 0.0),
        "val/seg_loss": metrics.get("val/seg_loss", 0.0),
        "val/cls_loss": metrics.get("val/cls_loss", 0.0),
        "val/dfl_loss": metrics.get("val/dfl_loss", 0.0),
    }
    wandb.log(wandb_metrics, step=trainer.epoch + 1)


def train_yolo(config: dict):
    """
    Execute a single hyperparameter optimization trial for YOLO training.

    Initializes Weights & Biases run, trains YOLO model with given hyperparameters,
    logs metrics to W&B, and reports best F1 score to Ray Tune for optimization.
    Includes custom callbacks for detailed metric tracking and memory management.

    Args:
        config (dict): Hyperparameter configuration dictionary containing:
            - model_name (str): YOLO model variant (e.g., 'yolo11m-seg.pt')
            - data_yaml (str): Path to dataset YAML file
            - epochs (int): Number of training epochs
            - imgsz (int): Input image size
            - batch (int): Batch size
            - lr0 (float): Initial learning rate
            - lrf (float): Final learning rate factor
            - momentum (float): SGD momentum
            - weight_decay (float): Weight decay
            - warmup_epochs (int): Warmup epochs
            - box (float): Box loss weight
            - cls (float): Classification loss weight
            - dfl (float): DFL loss weight
            - hsv_h (float): HSV hue augmentation
            - hsv_s (float): HSV saturation augmentation
            - hsv_v (float): HSV value augmentation
            - degrees (float): Rotation augmentation range
            - translate (float): Translation augmentation
            - scale (float): Scale augmentation
            - shear (float): Shear augmentation
            - perspective (float): Perspective augmentation
            - flipud (float): Vertical flip probability
            - fliplr (float): Horizontal flip probability
            - mosaic (float): Mosaic augmentation probability
            - mixup (float): Mixup augmentation probability
            - copy_paste (float): Copy-paste augmentation probability

    Returns:
        None: Reports metrics to Ray Tune via tune.report()
    """

    # Initialize Weights & Biases run
    run_id = tune.get_context().get_trial_id()
    wandb.init(
        project=config.get("wandb_project", "yolo_f1_tuning"),
        name=f"trial_{run_id}",
        config=config
    )

    model = YOLO('yolo11n-seg.pt')
    model.add_callback("on_fit_epoch_end", patched_on_fit_epoch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)


    # Temporary directory for YOLO logs
    results = model.train(
        data=config["data"],
        epochs=400,
        optimizer=config["optimizer"],
        batch=config["batch"],
        imgsz=480,
        lr0=config["lr0"],
        lrf=config["lrf"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        # warmup_epochs=config["warmup_epochs"],
        warmup_epochs=5,
        warmup_momentum=config["warmup_momentum"],
        box=config["box"],
        cls=config["cls"],
        # pose=config["pose"],
        pose=0.0,
        dropout=config["dropout"],
        patience=15,
        amp=False,
        plots=True,
        verbose=False,
        save=True,
        project=config["wandb_project"],
        name=f"trial_{run_id}",
    )
    

    # Log best model and training artifacts to W&B
    try:
        # Check if W&B session is still active, reinitialize if needed
        if wandb.run is None:
            print("W&B session closed, reinitializing...")
            wandb.init(
                project=config.get("wandb_project", "yolo_f1_tuning"),
                name=f"trial_{run_id}",
                config=config,
                resume="allow",
                id=run_id
            )
        
        # Log best model
        best_model_path = results.save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"Logging best model to W&B: {best_model_path}")
            model_artifact = wandb.Artifact(
                name=f"best_model_trial_{run_id}",
                type="model",
                description="Best YOLO model from training",
            )
            model_artifact.add_file(str(best_model_path), name="best.pt")
            wandb.log_artifact(model_artifact)
            print(f"✓ Best model logged to W&B: {best_model_path}")
        
        # Log training curves and visualizations
        curves_artifact = wandb.Artifact(
            name=f"training_curves_trial_{run_id}",
            type="curves",
            description="Training curves and visualizations from YOLO training",
        )
        
        # List of curve files to log
        curve_files = [
            "BoxF1_curve.png", "BoxP_curve.png", "BoxPR_curve.png", "BoxR_curve.png",
            "MaskF1_curve.png", "MaskP_curve.png", "MaskPR_curve.png", "MaskR_curve.png",
            "confusion_matrix.png", "confusion_matrix_normalized.png",
            "results.png", "results.csv", "labels.jpg"
        ]
        
        curves_logged = 0
        for curve_file in curve_files:
            curve_path = results.save_dir / curve_file
            if curve_path.exists():
                curves_artifact.add_file(str(curve_path), name=curve_file)
                curves_logged += 1
        
        # Log args.yaml if exists
        args_path = results.save_dir / "args.yaml"
        if args_path.exists():
            curves_artifact.add_file(str(args_path), name="args.yaml")
            curves_logged += 1
        
        if curves_logged > 0:
            wandb.log_artifact(curves_artifact)
            print(f"✓ Training curves and visualizations logged ({curves_logged} files)")
        else:
            print("⚠ No curve files found to log")
            
    except Exception as e:
        print(f"❌ Error logging artifacts: {e}")
        print(f"W&B run status: {wandb.run}")


    wandb.finish()

    # Cleanup GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to YOLO data YAML file')
    parser.add_argument('--project', type=str, required=True, help='W&B project name')
    args = parser.parse_args()

    # Resolve data path
    data_path = os.path.abspath(os.path.expanduser(args.data))
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' does not exist.")
        exit(1)

    # Init Ray
    ray.init(num_gpus=1, num_cpus=8)


    # Define your search space
    search_space = {
        # Learning rate dynamics
        "lr0": tune.loguniform(1e-6, 1e-2), 
        "lrf": tune.uniform(0.05, 1),
        "momentum": tune.uniform(0.7, 0.98),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "optimizer": tune.choice(["AdamW", "SGD", "RMSProp"]),
        # Warmup strategy
            # "warmup_epochs": tune.choice([5]),
        "warmup_momentum": tune.uniform(0.05, 0.3),
        # Weights for loss components
        "box": tune.uniform(4.0, 10.0),
        "cls": tune.uniform(0.5, 3.0),
            # "pose": tune.choice([0.0]),
        "dropout": tune.uniform(0.0, 0.4),
        # Augmentations disabled
        "flipud": tune.choice([0.0]),
        "fliplr": tune.choice([0.0]),
        "hsv_h": tune.choice([0.0]),
        "hsv_s": tune.choice([0.0]),
        "hsv_v": tune.choice([0.0]),
        "degrees": tune.choice([0.0]),
        "translate": tune.choice([0.0]),
        "scale": tune.choice([0.0]),
        "shear": tune.choice([0.0]),
        "perspective": tune.choice([0.0]),
        "mosaic": tune.choice([0.0]),
        "mixup": tune.choice([0.0]),
        "copy_paste": tune.choice([0.0]),
        "cutmix": tune.choice([0.0]),
        # Batch size
        "batch": tune.choice([16, 32, 64]),
        # Project settings
        "wandb_project": args.project,
        "data": data_path
    }


    # Optuna + ASHA configuration
    optuna_search = OptunaSearch(metric="F1(M)", mode="max")
    scheduler = ASHAScheduler(grace_period=20, reduction_factor=2)

    # Run tuning
    trainable_with_cpu_gpu = tune.with_resources(train_yolo, {"cpu": 2, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_cpu_gpu,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            metric="F1(M)",
            mode="max",
            num_samples=400,
            max_concurrent_trials=1,  # safe, avoids OOM
        ),
        param_space=search_space,
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="F1(M)", mode="max")
    print("\n=== Best Trial Summary ===")
    print(f"Best F1 mask: {best_result.metrics['F1(M)']:.4f}")
    print(f"Best Config: {best_result.config}")
