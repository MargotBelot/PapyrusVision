"""
Dataset utilities for PapyrusNU Hieroglyph Detection
Handles COCO format annotations and data splitting
"""

import json
import numpy as np
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
from pycocotools import mask as coco_mask


class HieroglyphDatasetUtils:
    """Utilities for handling hieroglyph dataset in COCO format"""
    
    def __init__(self, annotation_file: str):
        """Initialize with annotation file path"""
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        self.annotations = self.data['annotations']
        self.images = {img['id']: img for img in self.data['images']}
        self.categories = {cat['id']: cat for cat in self.data['categories']}
        
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {}
        
        # Basic counts
        stats['num_images'] = len(self.images)
        stats['num_annotations'] = len(self.annotations)
        stats['num_categories'] = len(self.categories)
        
        # Category distribution
        cat_counts = Counter([ann['category_id'] for ann in self.annotations])
        stats['category_counts'] = dict(cat_counts)
        stats['most_common_categories'] = cat_counts.most_common(10)
        stats['least_common_categories'] = cat_counts.most_common()[:-11:-1]
        
        # Annotation statistics
        areas = [ann['area'] for ann in self.annotations]
        stats['area_stats'] = {
            'mean': np.mean(areas),
            'std': np.std(areas),
            'min': np.min(areas),
            'max': np.max(areas),
            'median': np.median(areas)
        }
        
        # Bounding box statistics
        bbox_areas = [bbox[2] * bbox[3] for ann in self.annotations for bbox in [ann['bbox']]]
        bbox_widths = [bbox[2] for ann in self.annotations for bbox in [ann['bbox']]]
        bbox_heights = [bbox[3] for ann in self.annotations for bbox in [ann['bbox']]]
        
        stats['bbox_stats'] = {
            'areas': {'mean': np.mean(bbox_areas), 'std': np.std(bbox_areas)},
            'widths': {'mean': np.mean(bbox_widths), 'std': np.std(bbox_widths)},
            'heights': {'mean': np.mean(bbox_heights), 'std': np.std(bbox_heights)}
        }
        
        # Unicode and Gardiner codes
        unicode_codes = []
        gardiner_codes = []
        for ann in self.annotations:
            if 'attributes' in ann and 'Unicode' in ann['attributes']:
                unicode_codes.append(ann['attributes']['Unicode'])
            cat_name = self.categories[ann['category_id']]['name']
            gardiner_codes.append(cat_name)
            
        stats['unique_unicode_codes'] = len(set(unicode_codes))
        stats['unique_gardiner_codes'] = len(set(gardiner_codes))
        
        return stats
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, 
                     test_ratio: float = 0.1, seed: int = 42) -> Tuple[Dict, Dict, Dict]:
        """
        Split dataset maintaining class distribution
        Since we have only one image, we split annotations spatially
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Group annotations by category for stratified split
        ann_by_category = defaultdict(list)
        for ann in self.annotations:
            ann_by_category[ann['category_id']].append(ann)
        
        train_annotations = []
        val_annotations = []
        test_annotations = []
        
        # Split each category proportionally
        for cat_id, anns in ann_by_category.items():
            n_anns = len(anns)
            
            # If category has very few annotations, put at least one in train
            if n_anns <= 3:
                train_annotations.extend(anns[:max(1, int(n_anns * train_ratio))])
                remaining = anns[max(1, int(n_anns * train_ratio)):]
                if remaining:
                    val_annotations.extend(remaining[:len(remaining)//2])
                    test_annotations.extend(remaining[len(remaining)//2:])
            else:
                # Shuffle annotations
                shuffled_anns = anns.copy()
                random.shuffle(shuffled_anns)
                
                # Split
                n_train = int(n_anns * train_ratio)
                n_val = int(n_anns * val_ratio)
                
                train_annotations.extend(shuffled_anns[:n_train])
                val_annotations.extend(shuffled_anns[n_train:n_train + n_val])
                test_annotations.extend(shuffled_anns[n_train + n_val:])
        
        # Create dataset dictionaries
        train_data = self._create_subset_data(train_annotations)
        val_data = self._create_subset_data(val_annotations)
        test_data = self._create_subset_data(test_annotations)
        
        return train_data, val_data, test_data
    
    def _create_subset_data(self, annotations: List[Dict]) -> Dict:
        """Create a COCO format subset from annotations"""
        # Get unique categories from annotations
        used_cat_ids = set(ann['category_id'] for ann in annotations)
        subset_categories = [cat for cat_id, cat in self.categories.items() if cat_id in used_cat_ids]
        
        # Re-assign annotation IDs
        for i, ann in enumerate(annotations, 1):
            ann['id'] = i
        
        return {
            'info': self.data['info'],
            'licenses': self.data['licenses'],
            'images': list(self.images.values()),  # Keep all images (we only have one)
            'annotations': annotations,
            'categories': subset_categories
        }
    
    def save_split_data(self, train_data: Dict, val_data: Dict, test_data: Dict, 
                       output_dir: str = "data/annotations"):
        """Save split data to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        with open(output_path / "train_annotations.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(output_path / "val_annotations.json", 'w') as f:
            json.dump(val_data, f, indent=2)
            
        with open(output_path / "test_annotations.json", 'w') as f:
            json.dump(test_data, f, indent=2)
    
    def get_gardiner_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get mapping of Gardiner codes to Unicode and category info"""
        mapping = {}
        
        for ann in self.annotations:
            cat_id = ann['category_id']
            cat_name = self.categories[cat_id]['name']
            
            if cat_name not in mapping:
                mapping[cat_name] = {
                    'category_id': cat_id,
                    'unicode_codes': set(),
                    'count': 0
                }
            
            mapping[cat_name]['count'] += 1
            
            if 'attributes' in ann and 'Unicode' in ann['attributes']:
                mapping[cat_name]['unicode_codes'].add(ann['attributes']['Unicode'])
        
        # Convert sets to lists for JSON serialization
        for cat_name in mapping:
            mapping[cat_name]['unicode_codes'] = list(mapping[cat_name]['unicode_codes'])
        
        return mapping
    
    def analyze_class_balance(self) -> Dict[str, Any]:
        """Analyze class balance and identify rare classes"""
        cat_counts = Counter([ann['category_id'] for ann in self.annotations])
        
        total_annotations = len(self.annotations)
        class_percentages = {cat_id: (count / total_annotations) * 100 
                           for cat_id, count in cat_counts.items()}
        
        # Categorize classes by frequency
        rare_classes = []      # < 1% of data
        common_classes = []    # 1-5% of data
        frequent_classes = []  # > 5% of data
        
        for cat_id, percentage in class_percentages.items():
            cat_name = self.categories[cat_id]['name']
            if percentage < 1.0:
                rare_classes.append((cat_name, cat_counts[cat_id], percentage))
            elif percentage < 5.0:
                common_classes.append((cat_name, cat_counts[cat_id], percentage))
            else:
                frequent_classes.append((cat_name, cat_counts[cat_id], percentage))
        
        return {
            'total_annotations': total_annotations,
            'num_classes': len(cat_counts),
            'rare_classes': sorted(rare_classes, key=lambda x: x[2]),
            'common_classes': sorted(common_classes, key=lambda x: x[2], reverse=True),
            'frequent_classes': sorted(frequent_classes, key=lambda x: x[2], reverse=True),
            'class_percentages': class_percentages
        }
    
    def validate_annotations(self) -> Dict[str, List]:
        """Validate annotations and find potential issues"""
        issues = {
            'missing_segmentation': [],
            'invalid_bbox': [],
            'zero_area': [],
            'missing_unicode': [],
            'invalid_category': []
        }
        
        for ann in self.annotations:
            ann_id = ann['id']
            
            # Check segmentation
            if 'segmentation' not in ann or not ann['segmentation']:
                issues['missing_segmentation'].append(ann_id)
            
            # Check bounding box
            bbox = ann['bbox']
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                issues['invalid_bbox'].append(ann_id)
            
            # Check area
            if ann['area'] <= 0:
                issues['zero_area'].append(ann_id)
            
            # Check Unicode attribute
            if 'attributes' not in ann or 'Unicode' not in ann['attributes']:
                issues['missing_unicode'].append(ann_id)
            
            # Check category
            if ann['category_id'] not in self.categories:
                issues['invalid_category'].append(ann_id)
        
        return issues


def create_detectron2_dataset_dict(coco_data: Dict, image_dir: str) -> List[Dict]:
    """Convert COCO format to Detectron2 dataset format"""
    dataset_dicts = []

    images = {img['id']: img for img in coco_data['images']}

    # Create a mapping from original category IDs to new 0-based indices
    # based on the order of categories in the input coco_data
    original_categories = {cat['id']: cat for cat in coco_data['categories']}
    category_names_list = [original_categories[cat_id]['name'] for cat_id in sorted(original_categories.keys())]
    original_to_new_cat_id_map = {
        original_cat_id: new_cat_id
        for new_cat_id, original_cat_id in enumerate(sorted(original_categories.keys()))
    }

    for image_info in coco_data['images']:
        record = {}

        filename = Path(image_dir) / image_info['file_name']
        record['file_name'] = str(filename)
        record['image_id'] = image_info['id']
        record['height'] = image_info['height']
        record['width'] = image_info['width']

        # Get annotations for this image
        annotations = [ann for ann in coco_data['annotations']
                      if ann['image_id'] == image_info['id']]

        objs = []
        for ann in annotations:
            # Use the mapping to get the correct 0-based category_id
            original_cat_id = ann['category_id']
            new_cat_id = original_to_new_cat_id_map.get(original_cat_id, -1) # Use -1 or handle error if ID not found

            if new_cat_id != -1: # Only include annotations with a valid mapped category ID
                obj = {
                    'bbox': ann['bbox'],
                    'bbox_mode': 1,
                    'segmentation': ann['segmentation'],
                    'category_id': new_cat_id,  # Use the new 0-based index
                    'iscrowd': ann.get('iscrowd', 0)
                }

                # Add custom attributes
                if 'attributes' in ann:
                    obj['attributes'] = ann['attributes']

                objs.append(obj)
            else:
                print(f"Warning: Annotation with category ID {original_cat_id} not found in category mapping. Skipping.")


        record['annotations'] = objs
        dataset_dicts.append(record)

    return dataset_dicts
