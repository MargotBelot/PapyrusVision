"""
Evaluation utilities for PapyrusNU Hieroglyph Detection
Handles model evaluation, metrics computation, and performance analysis

Features:
- Hieroglyph detection evaluation
- Classification accuracy
- Per-class metrics
- Failure case analysis
- Unicode accuracy computation
- Integration with Detectron2 evaluators
"""

import numpy as np
import json
import torch
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset
        from detectron2.data import build_detection_test_loader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


class HieroglyphEvaluator:
    """Evaluation utilities for hieroglyph detection model"""
    
    def __init__(self, categories: Dict[int, Dict], unicode_mapping: Dict[str, str] = None):
        """ Initialize evaluator with category information"""
        self.categories = categories
        self.unicode_mapping = unicode_mapping or {}
        self.category_names = {cat_id: cat_info['name'] for cat_id, cat_info in categories.items()}
        
    def evaluate_predictions(self, predictions: List[Dict], ground_truth: List[Dict],
                           iou_threshold: float = 0.5) -> Dict[str, Any]:
        """Evaluation of model predictions"""
        results = {}
        
        # Convert to format suitable for evaluation
        pred_by_image = defaultdict(list)
        gt_by_image = defaultdict(list)
        
        for pred in predictions:
            pred_by_image[pred['image_id']].append(pred)
        
        for gt in ground_truth:
            gt_by_image[gt['image_id']].append(gt)
        
        # Compute metrics
        results.update(self._compute_detection_metrics(pred_by_image, gt_by_image, iou_threshold))
        results.update(self._compute_classification_metrics(pred_by_image, gt_by_image, iou_threshold))
        results.update(self._compute_per_class_metrics(pred_by_image, gt_by_image, iou_threshold))
        
        return results
    
    def _compute_detection_metrics(self, predictions: Dict, ground_truth: Dict, 
                                 iou_threshold: float) -> Dict[str, Any]:
        """Compute detection metrics (AP, AR, etc.)"""
        metrics = {}
        
        all_precisions = []
        all_recalls = []
        
        for image_id in ground_truth:
            preds = predictions.get(image_id, [])
            gts = ground_truth[image_id]
            
            if not gts:
                continue
                
            # Sort predictions by confidence
            preds = sorted(preds, key=lambda x: x.get('score', 0), reverse=True)
            
            # Compute matches
            matches = self._match_predictions_to_gt(preds, gts, iou_threshold)
            
            # Compute precision and recall
            tp = sum(matches)
            fp = len(preds) - tp
            fn = len(gts) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            all_precisions.append(precision)
            all_recalls.append(recall)
        
        # Aggregate metrics
        metrics['mean_precision'] = np.mean(all_precisions) if all_precisions else 0
        metrics['mean_recall'] = np.mean(all_recalls) if all_recalls else 0
        metrics['mean_f1'] = (2 * metrics['mean_precision'] * metrics['mean_recall'] / 
                             (metrics['mean_precision'] + metrics['mean_recall'])) \
                             if (metrics['mean_precision'] + metrics['mean_recall']) > 0 else 0
        
        return metrics
    
    def _compute_classification_metrics(self, predictions: Dict, ground_truth: Dict,
                                      iou_threshold: float) -> Dict[str, Any]:
        """Compute classification accuracy for matched detections"""
        metrics = {}
        
        all_true_labels = []
        all_pred_labels = []
        
        for image_id in ground_truth:
            preds = predictions.get(image_id, [])
            gts = ground_truth[image_id]
            
            if not preds or not gts:
                continue
            
            # Match predictions to ground truth
            matches, matched_gt_indices = self._match_predictions_to_gt_with_indices(
                preds, gts, iou_threshold)
            
            for i, (pred, is_match) in enumerate(zip(preds, matches)):
                if is_match:
                    gt_idx = matched_gt_indices[i]
                    all_true_labels.append(gts[gt_idx]['category_id'])
                    all_pred_labels.append(pred['category_id'])
        
        if all_true_labels and all_pred_labels:
            # Classification accuracy
            correct = sum(t == p for t, p in zip(all_true_labels, all_pred_labels))
            metrics['classification_accuracy'] = correct / len(all_true_labels)
            
            # Confusion matrix
            unique_labels = sorted(set(all_true_labels + all_pred_labels))
            cm = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_labels)
            metrics['confusion_matrix'] = cm
            metrics['confusion_matrix_labels'] = unique_labels
            
            # Per-class classification report
            class_names = [self.category_names.get(label, f'Cat_{label}') 
                          for label in unique_labels]
            report = classification_report(all_true_labels, all_pred_labels,
                                         labels=unique_labels, target_names=class_names,
                                         output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        
        return metrics
    
    def _compute_per_class_metrics(self, predictions: Dict, ground_truth: Dict,
                                 iou_threshold: float) -> Dict[str, Any]:
        """Compute per-class detection metrics"""
        metrics = {}
        per_class_metrics = {}
        
        # Group by category
        pred_by_category = defaultdict(list)
        gt_by_category = defaultdict(list)
        
        for image_id in predictions:
            for pred in predictions[image_id]:
                pred_by_category[pred['category_id']].append(pred)
        
        for image_id in ground_truth:
            for gt in ground_truth[image_id]:
                gt_by_category[gt['category_id']].append(gt)
        
        # Compute metrics per category
        for cat_id in self.categories:
            cat_preds = pred_by_category.get(cat_id, [])
            cat_gts = gt_by_category.get(cat_id, [])
            
            if not cat_gts:
                continue
                
            # Sort predictions by confidence
            cat_preds = sorted(cat_preds, key=lambda x: x.get('score', 0), reverse=True)
            
            # Compute AP for this class
            ap = self._compute_class_ap(cat_preds, cat_gts, iou_threshold)
            
            cat_name = self.category_names[cat_id]
            per_class_metrics[cat_name] = {
                'ap': ap,
                'num_gt': len(cat_gts),
                'num_pred': len(cat_preds)
            }
        
        metrics['per_class_metrics'] = per_class_metrics
        
        # Overall metrics
        all_aps = [m['ap'] for m in per_class_metrics.values() if m['ap'] is not None]
        metrics['mean_ap'] = np.mean(all_aps) if all_aps else 0
        
        return metrics
    
    def _match_predictions_to_gt(self, predictions: List[Dict], ground_truth: List[Dict],
                               iou_threshold: float) -> List[bool]:
        """Match predictions to ground truth based on IoU threshold"""
        if not predictions or not ground_truth:
            return [False] * len(predictions)
        
        matches = []
        used_gt = set()
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            pred_bbox = pred['bbox']
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in used_gt:
                    continue
                
                gt_bbox = gt['bbox']
                iou = self._calculate_bbox_iou(pred_bbox, gt_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                matches.append(True)
                used_gt.add(best_gt_idx)
            else:
                matches.append(False)
        
        return matches
    
    def _match_predictions_to_gt_with_indices(self, predictions: List[Dict], 
                                            ground_truth: List[Dict],
                                            iou_threshold: float) -> Tuple[List[bool], List[int]]:
        """Match predictions to ground truth and return matched GT indices"""
        if not predictions or not ground_truth:
            return [False] * len(predictions), [-1] * len(predictions)
        
        matches = []
        matched_indices = []
        used_gt = set()
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            pred_bbox = pred['bbox']
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in used_gt:
                    continue
                
                gt_bbox = gt['bbox']
                iou = self._calculate_bbox_iou(pred_bbox, gt_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                matches.append(True)
                matched_indices.append(best_gt_idx)
                used_gt.add(best_gt_idx)
            else:
                matches.append(False)
                matched_indices.append(-1)
        
        return matches, matched_indices
    
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to XYXY format
        x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
        x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _compute_class_ap(self, predictions: List[Dict], ground_truth: List[Dict],
                         iou_threshold: float) -> Optional[float]:
        """Compute Average Precision for a single class"""
        if not ground_truth:
            return None
        
        if not predictions:
            return 0.0
        
        # Sort by confidence
        predictions = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
        
        # Match predictions to ground truth
        matches = self._match_predictions_to_gt(predictions, ground_truth, iou_threshold)
        
        # Compute precision-recall curve
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        
        for match in matches:
            if match:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp)
            recall = tp / len(ground_truth)
            
            precisions.append(precision)
            recalls.append(recall)
        
        ap = self._compute_ap_11_point(precisions, recalls)
        return ap
    
    def _compute_ap_11_point(self, precisions: List[float], recalls: List[float]) -> float:
        """Compute AP"""
        recall_levels = np.linspace(0, 1, 11)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        interpolated_precisions = []
        
        for r in recall_levels:
            # Find precisions for recalls >= r
            valid_precisions = precisions[recalls >= r]
            if len(valid_precisions) > 0:
                interpolated_precisions.append(np.max(valid_precisions))
            else:
                interpolated_precisions.append(0.0)
        
        return np.mean(interpolated_precisions)
    
    def analyze_failure_cases(self, predictions: Dict, ground_truth: Dict,
                            iou_threshold: float = 0.5) -> Dict[str, Any]:
        """Analyze common failure modes"""
        analysis = {
            'false_positives': [],
            'false_negatives': [],
            'misclassifications': [],
            'low_confidence_correct': [],
            'high_confidence_incorrect': []
        }
        
        for image_id in ground_truth:
            preds = predictions.get(image_id, [])
            gts = ground_truth[image_id]
            
            if not gts:
                continue
            
            # Sort predictions by confidence
            preds = sorted(preds, key=lambda x: x.get('score', 0), reverse=True)
            
            # Match predictions to ground truth
            matches, matched_gt_indices = self._match_predictions_to_gt_with_indices(
                preds, gts, iou_threshold)
            
            # Find false positives (unmatched predictions)
            for i, (pred, is_match) in enumerate(zip(preds, matches)):
                if not is_match:
                    analysis['false_positives'].append({
                        'image_id': image_id,
                        'prediction': pred,
                        'confidence': pred.get('score', 0)
                    })
                else:
                    # Check for misclassifications
                    gt_idx = matched_gt_indices[i]
                    gt = gts[gt_idx]
                    if pred['category_id'] != gt['category_id']:
                        analysis['misclassifications'].append({
                            'image_id': image_id,
                            'predicted_class': self.category_names.get(pred['category_id']),
                            'true_class': self.category_names.get(gt['category_id']),
                            'confidence': pred.get('score', 0)
                        })
                    
                    # Check confidence levels
                    confidence = pred.get('score', 0)
                    if confidence < 0.5:  # Low confidence but correct
                        analysis['low_confidence_correct'].append({
                            'image_id': image_id,
                            'category': self.category_names.get(pred['category_id']),
                            'confidence': confidence
                        })
            
            # Find false negatives (unmatched ground truth)
            matched_gt_set = set(matched_gt_indices)
            for gt_idx, gt in enumerate(gts):
                if gt_idx not in matched_gt_set:
                    analysis['false_negatives'].append({
                        'image_id': image_id,
                        'ground_truth': gt,
                        'category': self.category_names.get(gt['category_id'])
                    })
            
            # High confidence incorrect predictions
            for pred in preds:
                if pred.get('score', 0) > 0.8:  # High confidence
                    # Check if this is a false positive or misclassification
                    is_fp = pred in [fp['prediction'] for fp in analysis['false_positives'][-len(preds):]]
                    is_misclass = pred in [mc['prediction'] if 'prediction' in mc else None 
                                          for mc in analysis['misclassifications'][-len(preds):]]
                    
                    if is_fp or is_misclass:
                        analysis['high_confidence_incorrect'].append({
                            'image_id': image_id,
                            'prediction': pred,
                            'confidence': pred.get('score', 0),
                            'type': 'false_positive' if is_fp else 'misclassification'
                        })
        
        # Summarize failure modes
        analysis['summary'] = {
            'total_false_positives': len(analysis['false_positives']),
            'total_false_negatives': len(analysis['false_negatives']),
            'total_misclassifications': len(analysis['misclassifications']),
            'avg_fp_confidence': np.mean([fp['confidence'] for fp in analysis['false_positives']]) 
                                if analysis['false_positives'] else 0,
            'most_confused_classes': self._find_most_confused_classes(analysis['misclassifications']),
            'most_missed_classes': Counter([fn['category'] for fn in analysis['false_negatives']]).most_common(10)
        }
        
        return analysis
    
    def _find_most_confused_classes(self, misclassifications: List[Dict]) -> List[Tuple[str, str, int]]:
        """Find the most commonly confused class pairs"""
        confusion_pairs = [(mc['true_class'], mc['predicted_class']) 
                          for mc in misclassifications]
        confusion_counts = Counter(confusion_pairs)
        return confusion_counts.most_common(10)
    
    def compute_unicode_accuracy(self, predictions: Dict, ground_truth: Dict,
                               iou_threshold: float = 0.5) -> Dict[str, Any]:
        """Compute accuracy for Unicode prediction (if available)"""
        if not self.unicode_mapping:
            return {'unicode_accuracy': None, 'message': 'No Unicode mapping provided'}
        
        correct_unicode = 0
        total_matched = 0
        
        for image_id in ground_truth:
            preds = predictions.get(image_id, [])
            gts = ground_truth[image_id]
            
            matches, matched_gt_indices = self._match_predictions_to_gt_with_indices(
                preds, gts, iou_threshold)
            
            for i, (pred, is_match) in enumerate(zip(preds, matches)):
                if is_match:
                    total_matched += 1
                    gt_idx = matched_gt_indices[i]
                    gt = gts[gt_idx]
                    
                    # Get predicted and true Unicode
                    pred_gardiner = self.category_names.get(pred['category_id'])
                    true_gardiner = self.category_names.get(gt['category_id'])
                    
                    pred_unicode = self.unicode_mapping.get(pred_gardiner)
                    true_unicode = gt.get('attributes', {}).get('Unicode')
                    
                    if pred_unicode and true_unicode and pred_unicode == true_unicode:
                        correct_unicode += 1
        
        unicode_accuracy = correct_unicode / total_matched if total_matched > 0 else 0
        
        return {
            'unicode_accuracy': unicode_accuracy,
            'correct_unicode': correct_unicode,
            'total_matched': total_matched
        }
    
    def export_results(self, results: Dict[str, Any], output_path: str):
        """Export evaluation results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Evaluation results exported to {output_path}")


class DetectronEvaluatorWrapper:
    """Wrapper for Detectron2's built-in evaluators"""
    
    def __init__(self, dataset_name: str, cfg, output_dir: str):
        """Initialize Detectron2 evaluator wrapper"""
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.output_dir = output_dir
        self.evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir)
    
    def evaluate_model(self, predictor: DefaultPredictor) -> Dict[str, Any]:
        """Run full evaluation using Detectron2's evaluator"""   
        # Build data loader
        data_loader = build_detection_test_loader(self.cfg, self.dataset_name)
        
        # Run inference
        results = inference_on_dataset(predictor.model, data_loader, self.evaluator)
        
        return results
