import pathlib
from typing import Optional
from bsort.data import get_all_image_paths, split_dataset, load_image, load_label, extract_roi
from bsort.model import ColorThresholdModel
from bsort.utils.metrics import calculate_metrics, get_confusion_matrix
from bsort.utils.logging import setup_logger, init_wandb
from bsort.utils.visualization import plot_confusion_matrix, plot_class_distribution, log_plots_to_wandb

logger = setup_logger()

def train(data_dir: str, test_size: float = 0.2, use_wandb: bool = False):
    """
    Runs the training/evaluation pipeline.
    
    Since we are using a heuristic model, 'training' essentially means
    evaluating the fixed model on the dataset to establish a baseline.
    """
    logger.info(f"Starting training pipeline with data_dir={data_dir}")
    
    if use_wandb:
        init_wandb(config={"model": "ColorThreshold", "test_size": test_size})
    
    # 1. Load Data
    data_path = pathlib.Path(data_dir)
    all_images = get_all_image_paths(data_path)
    
    if not all_images:
        logger.error("No images found in data directory!")
        return
        
    train_files, val_files = split_dataset(all_images, test_size=test_size)
    logger.info(f"Data split: {len(train_files)} training, {len(val_files)} validation")
    
    # 2. Initialize Model
    model = ColorThresholdModel()
    
    # 3. Evaluate (on Validation set)
    logger.info("Evaluating on validation set...")
    y_true = []
    y_pred = []
    
    class_names = {0: 'Light Blue', 1: 'Dark Blue', 2: 'Others'}
    
    # Track class distribution for bias check
    val_class_counts = {0: 0, 1: 0, 2: 0}
    
    for img_path in val_files:
        img = load_image(img_path)
        if img is None: continue
            
        label_path = img_path.with_suffix('.txt')
        labels = load_label(label_path)
        
        for label in labels:
            cls_id = int(label[0])
            roi = extract_roi(img, label)
            
            if roi is None: continue
            
            pred = model.predict(roi)
            
            y_true.append(cls_id)
            y_pred.append(pred)
            
            if cls_id in val_class_counts:
                val_class_counts[cls_id] += 1

    # 4. Metrics & Logging
    if not y_true:
        logger.warning("No objects found in validation set.")
        return

    metrics = calculate_metrics(y_true, y_pred)
    logger.info(f"Validation Metrics: {metrics}")
    
    cm = get_confusion_matrix(y_true, y_pred)
    
    # Visualizations
    cm_plot = plot_confusion_matrix(cm, list(class_names.values()))
    dist_plot = plot_class_distribution(val_class_counts, class_names, title="Validation Set Distribution")
    
    if use_wandb:
        import wandb
        wandb.log(metrics)
        log_plots_to_wandb({"confusion_matrix": cm_plot, "class_distribution": dist_plot})
        
    logger.info("Training pipeline completed successfully.")
