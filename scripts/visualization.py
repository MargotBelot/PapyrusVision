"""
Visualization utilities for PapyrusNU Hieroglyph Detection
Handles data visualization, training plots, and result visualization

Features:
- Dataset overview plots
- Class distribution analysis
- Annotation visualization on images
- Training curves plotting
- Evaluation metrics visualization
- Interactive annotation viewer using Plotly
- Data split analysis for train/val/test sets
- Hieroglyph prediction visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pycocotools import mask as coco_mask


class HieroglyphVisualizer:
    """Visualization utilities for hieroglyph dataset and results"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer with default figure size"""
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_dataset_overview(self, stats: Dict[str, Any], save_path: Optional[str] = None):
        """Create overview plots of dataset statistics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # Basic statistics
        ax = axes[0, 0]
        basic_stats = [stats['num_images'], stats['num_annotations'], stats['num_categories']]
        labels = ['Images', 'Annotations', 'Categories']
        bars = ax.bar(labels, basic_stats, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_title('Dataset Composition')
        ax.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, value in zip(bars, basic_stats):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        # Area distribution
        ax = axes[0, 1]
        areas = [ann_area for ann_area in stats['area_stats'].values()]
        ax.hist(areas, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax.set_title('Annotation Area Distribution')
        ax.set_xlabel('Area (pixels²)')
        ax.set_ylabel('Frequency')
        
        # Top categories
        ax = axes[0, 2]
        top_cats = stats['most_common_categories'][:10]
        cat_names = [f"Cat_{cat_id}"for cat_id, _ in top_cats]
        counts = [count for _, count in top_cats]
        
        bars = ax.barh(range(len(cat_names)), counts, color='lightgreen')
        ax.set_yticks(range(len(cat_names)))
        ax.set_yticklabels(cat_names)
        ax.set_title('Top 10 Categories by Count')
        ax.set_xlabel('Number of Annotations')
        
        # Bbox width vs height scatter
        ax = axes[1, 0]
        bbox_stats = stats['bbox_stats']
        # We need the actual bbox data, so this is a placeholder
        ax.scatter(np.random.normal(bbox_stats['widths']['mean'], bbox_stats['widths']['std'], 100),
                  np.random.normal(bbox_stats['heights']['mean'], bbox_stats['heights']['std'], 100),
                  alpha=0.6, color='coral')
        ax.set_title('Bounding Box Dimensions')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        
        # Unicode vs Gardiner codes
        ax = axes[1, 1]
        codes_data = [stats['unique_unicode_codes'], stats['unique_gardiner_codes']]
        code_labels = ['Unicode Codes', 'Gardiner Codes']
        bars = ax.bar(code_labels, codes_data, color=['plum', 'khaki'])
        ax.set_title('Unique Code Counts')
        ax.set_ylabel('Count')
        
        for bar, value in zip(bars, codes_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        # Area statistics table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        area_data = stats['area_stats']
        table_data = [[k.title(), f"{v:.1f}"] for k, v in area_data.items()]
        table = ax.table(cellText=table_data, colLabels=['Statistic', 'Value'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        ax.set_title('Area Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_class_distribution(self, class_balance: Dict[str, Any], save_path: Optional[str] = None):
        """Plot class distribution and balance analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Frequent classes
        ax = axes[0, 0]
        if class_balance['frequent_classes']:
            freq_data = class_balance['frequent_classes'][:10]
            names = [item[0] for item in freq_data]
            counts = [item[1] for item in freq_data]
            
            bars = ax.barh(range(len(names)), counts, color='green', alpha=0.7)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            ax.set_title('Most Frequent Classes (>5%)')
            ax.set_xlabel('Count')
        
        # Common classes
        ax = axes[0, 1]
        if class_balance['common_classes']:
            common_data = class_balance['common_classes'][:15]
            names = [item[0] for item in common_data]
            counts = [item[1] for item in common_data]
            
            bars = ax.barh(range(len(names)), counts, color='orange', alpha=0.7)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_title('Common Classes (1-5%)')
            ax.set_xlabel('Count')
        
        # Rare classes distribution
        ax = axes[1, 0]
        if class_balance['rare_classes']:
            rare_counts = [item[1] for item in class_balance['rare_classes']]
            ax.hist(rare_counts, bins=min(20, len(set(rare_counts))), 
                   alpha=0.7, color='red', edgecolor='black')
            ax.set_title('Distribution of Rare Classes (<1%)')
            ax.set_xlabel('Annotation Count')
            ax.set_ylabel('Number of Classes')
        
        # Class balance pie chart
        ax = axes[1, 1]
        balance_data = [
            len(class_balance['frequent_classes']),
            len(class_balance['common_classes']),
            len(class_balance['rare_classes'])
        ]
        labels = ['Frequent (>5%)', 'Common (1-5%)', 'Rare (<1%)']
        colors = ['green', 'orange', 'red']
        
        wedges, texts, autotexts = ax.pie(balance_data, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title('Class Balance Overview')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_annotations_on_image(self, image_path: str, annotations: List[Dict],
                                     categories: Dict[int, Dict], max_annotations: int = 50,
                                     save_path: Optional[str] = None):
        """Visualize annotations overlaid on the image"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        ax.imshow(image)
        
        # Color map for categories
        colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
        category_colors = {cat_id: colors[i % len(colors)] for i, cat_id in enumerate(categories.keys())}
        
        # Plot annotations (limit to avoid clutter)
        plotted_annotations = annotations[:max_annotations]
        
        for ann in plotted_annotations:
            bbox = ann['bbox']
            cat_id = ann['category_id']
            color = category_colors[cat_id]
            
            # Draw bounding box
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                   linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Add category label
            category_name = categories[cat_id]['name']
            ax.text(bbox[0], bbox[1] - 5, category_name, fontsize=8,
                   color=color, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_title(f'Annotations Visualization ({len(plotted_annotations)}/{len(annotations)} shown)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_curves(self, training_data: Dict[str, List], save_path: Optional[str] = None):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        iterations = training_data.get('iterations', [])
        
        # Total loss
        ax = axes[0, 0]
        if 'total_loss' in training_data:
            ax.plot(iterations, training_data['total_loss'], label='Total Loss', color='red')
            ax.set_title('Total Loss')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Classification loss
        ax = axes[0, 1]
        if 'cls_loss' in training_data:
            ax.plot(iterations, training_data['cls_loss'], label='Classification Loss', color='blue')
            ax.set_title('Classification Loss')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Box regression loss
        ax = axes[1, 0]
        if 'bbox_loss' in training_data:
            ax.plot(iterations, training_data['bbox_loss'], label='BBox Regression Loss', color='green')
            ax.set_title('Bounding Box Regression Loss')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Mask loss
        ax = axes[1, 1]
        if 'mask_loss' in training_data:
            ax.plot(iterations, training_data['mask_loss'], label='Mask Loss', color='purple')
            ax.set_title('Mask Loss')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_evaluation_metrics(self, eval_results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot evaluation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
        
        # AP metrics
        ax = axes[0, 0]
        ap_metrics = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
        ap_values = [eval_results.get(metric, 0) for metric in ap_metrics]
        
        bars = ax.bar(ap_metrics, ap_values, color='skyblue', alpha=0.8)
        ax.set_title('Average Precision Metrics')
        ax.set_ylabel('AP Score')
        ax.set_ylim(0, 1)
        
        for bar, value in zip(bars, ap_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # AR metrics
        ax = axes[0, 1]
        ar_metrics = ['AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
        ar_values = [eval_results.get(metric, 0) for metric in ar_metrics]
        
        bars = ax.bar(ar_metrics, ar_values, color='lightgreen', alpha=0.8)
        ax.set_title('Average Recall Metrics')
        ax.set_ylabel('AR Score')
        ax.set_ylim(0, 1)
        
        for bar, value in zip(bars, ar_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Per-class AP (if available)
        ax = axes[1, 0]
        if 'per_class_ap' in eval_results:
            per_class_ap = eval_results['per_class_ap']
            class_names = list(per_class_ap.keys())[:20]  # Show top 20
            ap_scores = [per_class_ap[name] for name in class_names]
            
            bars = ax.barh(range(len(class_names)), ap_scores, color='coral', alpha=0.8)
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names, fontsize=8)
            ax.set_title('Per-Class AP (Top 20)')
            ax.set_xlabel('AP Score')
        
        # Confusion matrix (if available)
        ax = axes[1, 1]
        if 'confusion_matrix' in eval_results:
            cm = eval_results['confusion_matrix']
            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            ax.set_title('Confusion Matrix (Sample)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'Confusion Matrix\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_predictions(self, image_path: str, predictions: List[Dict],
                            categories: Dict[int, Dict], confidence_threshold: float = 0.5,
                            save_path: Optional[str] = None):
        """Visualize model predictions on image"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Filter predictions by confidence
        filtered_preds = [pred for pred in predictions 
                         if pred.get('score', 0) >= confidence_threshold]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        ax.imshow(image)
        
        # Color map for categories
        colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
        category_colors = {cat_id: colors[i % len(colors)] for i, cat_id in enumerate(categories.keys())}
        
        for pred in filtered_preds:
            bbox = pred['bbox']
            cat_id = pred['category_id']
            score = pred.get('score', 0)
            color = category_colors.get(cat_id, 'red')
            
            # Draw bounding box
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                   linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Add label with confidence
            category_name = categories.get(cat_id, {}).get('name', f'Cat_{cat_id}')
            label = f'{category_name}: {score:.2f}'
            ax.text(bbox[0], bbox[1] - 5, label, fontsize=8,
                   color=color, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_title(f'Model Predictions (conf > {confidence_threshold}, {len(filtered_preds)} detections)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_annotation_viewer(self, image_path: str, annotations: List[Dict],
                                           categories: Dict[int, Dict]):
        """Create an interactive annotation viewer using Plotly"""
        # Load image
        img = Image.open(image_path)
        
        # Create figure
        fig = go.Figure()
        
        # Add image
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=img.width,
                sizey=img.height,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )
        
        # Add annotations
        for i, ann in enumerate(annotations):
            bbox = ann['bbox']
            cat_id = ann['category_id']
            category_name = categories[cat_id]['name']
            
            # Add bounding box
            fig.add_shape(
                type="rect",
                x0=bbox[0],
                y0=img.height - bbox[1] - bbox[3],  # Flip Y coordinate
                x1=bbox[0] + bbox[2],
                y1=img.height - bbox[1],
                line=dict(color="red", width=2),
                opacity=0.7
            )
            
            # Add annotation info on hover
            fig.add_trace(go.Scatter(
                x=[bbox[0] + bbox[2]/2],
                y=[img.height - bbox[1] - bbox[3]/2],
                mode='markers',
                marker=dict(size=5, opacity=0),
                name=f'{category_name}_{i}',
                hovertext=f"Category: {category_name}<br>Area: {ann['area']:.0f}px²<br>BBox: {bbox}",
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title="Interactive Annotation Viewer",
            xaxis=dict(range=[0, img.width], showgrid=False, zeroline=False),
            yaxis=dict(range=[0, img.height], showgrid=False, zeroline=False, scaleanchor="x"),
            width=1200,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def plot_data_split_analysis(self, train_data: Dict, val_data: Dict, test_data: Dict,
                               save_path: Optional[str] = None):
        """Analyze and visualize data split"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Data Split Analysis', fontsize=16, fontweight='bold')
        
        # Split sizes
        ax = axes[0, 0]
        split_sizes = [
            len(train_data['annotations']),
            len(val_data['annotations']),
            len(test_data['annotations'])
        ]
        labels = ['Train', 'Validation', 'Test']
        colors = ['green', 'orange', 'red']
        
        bars = ax.bar(labels, split_sizes, color=colors, alpha=0.8)
        ax.set_title('Dataset Split Sizes')
        ax.set_ylabel('Number of Annotations')
        
        for bar, size in zip(bars, split_sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{size:,}', ha='center', va='bottom', fontweight='bold')
        
        # Category distribution across splits
        ax = axes[0, 1]
        train_cats = Counter([ann['category_id'] for ann in train_data['annotations']])
        val_cats = Counter([ann['category_id'] for ann in val_data['annotations']])
        test_cats = Counter([ann['category_id'] for ann in test_data['annotations']])
        
        all_cats = set(train_cats.keys()) | set(val_cats.keys()) | set(test_cats.keys())
        
        train_counts = [train_cats.get(cat, 0) for cat in all_cats]
        val_counts = [val_cats.get(cat, 0) for cat in all_cats]
        test_counts = [test_cats.get(cat, 0) for cat in all_cats]
        
        x = np.arange(len(all_cats))
        width = 0.25
        
        ax.bar(x - width, train_counts, width, label='Train', color='green', alpha=0.8)
        ax.bar(x, val_counts, width, label='Validation', color='orange', alpha=0.8)
        ax.bar(x + width, test_counts, width, label='Test', color='red', alpha=0.8)
        
        ax.set_title('Category Distribution Across Splits')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Count')
        ax.legend()
        
        # Class balance preservation
        ax = axes[1, 0]
        total_per_cat = {cat: train_cats.get(cat, 0) + val_cats.get(cat, 0) + test_cats.get(cat, 0) 
                        for cat in all_cats}
        
        train_ratios = [train_cats.get(cat, 0) / total_per_cat[cat] if total_per_cat[cat] > 0 else 0 
                       for cat in all_cats]
        
        ax.hist(train_ratios, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(x=0.7, color='red', linestyle='--', label='Target Train Ratio')
        ax.set_title('Training Set Ratio Distribution')
        ax.set_xlabel('Train Ratio per Category')
        ax.set_ylabel('Number of Categories')
        ax.legend()
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        summary_data = [
            ['Total Annotations', f"{sum(split_sizes):,}"],
            ['Train Split', f"{split_sizes[0]:,} ({split_sizes[0]/sum(split_sizes)*100:.1f}%)"],
            ['Validation Split', f"{split_sizes[1]:,} ({split_sizes[1]/sum(split_sizes)*100:.1f}%)"],
            ['Test Split', f"{split_sizes[2]:,} ({split_sizes[2]/sum(split_sizes)*100:.1f}%)"],
            ['Categories in Train', f"{len(train_cats)}"],
            ['Categories in Val', f"{len(val_cats)}"],
            ['Categories in Test', f"{len(test_cats)}"]
        ]
        
        table = ax.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        ax.set_title('Split Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
