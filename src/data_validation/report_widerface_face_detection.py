"""
Face Detection Dataset Report - WiderFace Dataset
===============================================

This script generates a comprehensive report on the WiderFace face detection dataset,
including dataset statistics, splits, and metadata analysis.
"""

import os
import json
from datetime import datetime
import scipy.io
import numpy as np

def load_widerface_annotations(mat_file):
    """Load WiderFace annotations from .mat file"""
    try:
        mat_data = scipy.io.loadmat(mat_file)
        return mat_data
    except Exception as e:
        print(f"❌ Error loading {mat_file}: {e}")
        return None

def parse_bbox_file(bbox_file):
    """Parse WiderFace bounding box ground truth file"""
    bboxes = {}
    try:
        with open(bbox_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            # File path
            file_path = lines[i].strip()
            i += 1

            # Number of faces
            num_faces = int(lines[i].strip())
            i += 1

            # Bounding boxes
            file_bboxes = []
            for j in range(num_faces):
                bbox_line = lines[i].strip()
                i += 1
                # Parse bbox: x1, y1, w, h, blur, expression, illumination, occlusion, pose, invalid
                parts = bbox_line.split()
                if len(parts) >= 4:
                    x1, y1, w, h = map(int, parts[:4])
                    file_bboxes.append({
                        'x1': x1, 'y1': y1, 'w': w, 'h': h,
                        'x2': x1 + w, 'y2': y1 + h,
                        'attributes': parts[4:] if len(parts) > 4 else []
                    })

            bboxes[file_path] = file_bboxes

    except Exception as e:
        print(f"❌ Error parsing {bbox_file}: {e}")

    return bboxes

def analyze_widerface_directory(base_path, split_name):
    """Analyze a WiderFace split directory"""
    images_dir = os.path.join(base_path, f"WIDER_{split_name}", "images")

    if not os.path.exists(images_dir):
        return None

    # Count images and events
    event_dirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    total_images = 0
    image_files = []

    for event in event_dirs:
        event_path = os.path.join(images_dir, event)
        event_images = [f for f in os.listdir(event_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(event_images)
        image_files.extend([os.path.join(event, f) for f in event_images])

    return {
        'event_count': len(event_dirs),
        'image_count': total_images,
        'events': event_dirs,
        'image_files': image_files
    }

def generate_widerface_report():
    """Generate comprehensive report for WiderFace face detection dataset"""

    print("📊 Generating WiderFace Face Detection Dataset Report...")

    base_path = "datasets/widerface"

    if not os.path.exists(base_path):
        print(f"❌ WiderFace dataset not found at {base_path}")
        return None

    # Initialize report structure
    report = {
        "dataset_name": "WiderFace",
        "dataset_type": "Face Detection",
        "task": "Face Detection with Bounding Boxes",
        "generated_at": datetime.now().isoformat(),
        "dataset_splits": {},
        "annotation_statistics": {},
        "image_statistics": {},
        "file_structure": {},
        "quality_metrics": {}
    }

    # Analyze each split
    splits = ['train', 'val', 'test']
    total_images = 0
    total_faces = 0

    for split in splits:
        print(f"🔍 Analyzing {split} split...")

        # Analyze directory structure
        dir_info = analyze_widerface_directory(base_path, split)

        if dir_info is None:
            print(f"⚠️  {split} split not found, skipping...")
            continue

        split_data = {
            'image_count': dir_info['image_count'],
            'event_count': dir_info['event_count'],
            'events': dir_info['events']
        }

        # Load annotations if available
        if split != 'test':  # Test set doesn't have ground truth
            mat_file = os.path.join(base_path, "wider_face_split", f"wider_face_{split}.mat")
            bbox_file = os.path.join(base_path, "wider_face_split", f"wider_face_{split}_bbx_gt.txt")

            # Load .mat file
            mat_data = load_widerface_annotations(mat_file)
            if mat_data:
                split_data['mat_annotations_loaded'] = True
            else:
                split_data['mat_annotations_loaded'] = False

            # Load bbox file
            bboxes = parse_bbox_file(bbox_file)
            if bboxes:
                split_data['bbox_annotations_loaded'] = True
                split_data['annotated_images'] = len(bboxes)

                # Calculate face statistics
                face_counts = [len(faces) for faces in bboxes.values()]
                split_data['face_statistics'] = {
                    'total_faces': sum(face_counts),
                    'min_faces_per_image': min(face_counts) if face_counts else 0,
                    'max_faces_per_image': max(face_counts) if face_counts else 0,
                    'avg_faces_per_image': sum(face_counts) / len(face_counts) if face_counts else 0,
                    'images_with_faces': len([c for c in face_counts if c > 0]),
                    'images_without_faces': len([c for c in face_counts if c == 0])
                }

                total_faces += split_data['face_statistics']['total_faces']
            else:
                split_data['bbox_annotations_loaded'] = False
        else:
            split_data['has_ground_truth'] = False
            split_data['note'] = "Test set does not include ground truth annotations"

        report["dataset_splits"][split] = split_data
        total_images += dir_info['image_count']

    # Overall statistics
    report["total_images"] = total_images
    report["total_faces"] = total_faces

    # Image statistics (estimate based on file analysis)
    report["image_statistics"] = {
        "estimated_total_images": total_images,
        "splits": {
            split: info['image_count'] for split, info in report["dataset_splits"].items()
        },
        "file_formats": [".jpg"],
        "note": "Detailed image statistics require loading all images"
    }

    # Annotation statistics
    report["annotation_statistics"] = {
        "total_annotated_faces": total_faces,
        "annotation_format": "MATLAB .mat files and text files with bounding boxes",
        "bbox_attributes": ["blur", "expression", "illumination", "occlusion", "pose", "invalid"],
        "splits_with_annotations": [s for s in splits if s != 'test']
    }

    # File structure
    report["file_structure"] = {
        "base_path": base_path,
        "directories": [
            "WIDER_train/images/",
            "WIDER_val/images/",
            "WIDER_test/images/",
            "wider_face_split/",
            "wider_face_eval_tools/",
            "WiderFace-Evaluation-master/"
        ],
        "annotation_files": [
            "wider_face_split/wider_face_train.mat",
            "wider_face_split/wider_face_train_bbx_gt.txt",
            "wider_face_split/wider_face_val.mat",
            "wider_face_split/wider_face_val_bbx_gt.txt",
            "wider_face_split/wider_face_test.mat",
            "wider_face_split/wider_face_test_filelist.txt"
        ]
    }

    # Quality metrics
    report["quality_metrics"] = {
        "data_integrity": {
            "all_directories_accessible": all(os.path.exists(os.path.join(base_path, f"WIDER_{s}")) for s in ['train', 'val', 'test']),
            "annotation_files_exist": all(os.path.exists(os.path.join(base_path, "wider_face_split", f"wider_face_{s}.mat")) for s in ['train', 'val', 'test'])
        },
        "annotation_quality": {
            "has_attribute_annotations": True,
            "has_occlusion_labels": True,
            "has_pose_variations": True
        }
    }

    return report

def save_report(report, filename="widerface_face_detection_report.json"):
    """Save report to JSON file"""

    os.makedirs("reports", exist_ok=True)
    filepath = os.path.join("reports", filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"💾 Report saved to: {filepath}")
    return filepath

def print_report_summary(report):
    """Print a human-readable summary of the report"""

    print("\n" + "="*60)
    print("WIDERFACE FACE DETECTION DATASET REPORT")
    print("="*60)

    print(f"📊 Dataset: {report['dataset_name']}")
    print(f"🎯 Task: {report['task']}")
    print(f"🖼️  Total Images: {report['total_images']:,}")
    print(f"👤 Total Faces: {report['total_faces']:,}")
    print(f"📅 Generated: {report['generated_at'][:19]}")

    print(f"\n📂 Dataset Splits:")
    for split, info in report['dataset_splits'].items():
        print(f"  • {split.upper()}: {info['image_count']} images, {info.get('event_count', 'N/A')} events")
        if 'face_statistics' in info:
            faces = info['face_statistics']
            print(f"    - Faces: {faces['total_faces']} total, {faces['avg_faces_per_image']:.1f} avg per image")

    print(f"\n📏 Annotation Details:")
    print(f"  • Format: {report['annotation_statistics']['annotation_format']}")
    print(f"  • Attributes: {', '.join(report['annotation_statistics']['bbox_attributes'])}")
    print(f"  • Annotated splits: {', '.join(report['annotation_statistics']['splits_with_annotations'])}")

    print(f"\n🖼️  Image Statistics:")
    print(f"  • File formats: {', '.join(report['image_statistics']['file_formats'])}")
    print(f"  • Organized by events: {len(report['dataset_splits']['train']['events'])} event categories")

    print(f"\n✅ Quality Checks:")
    integrity = report['quality_metrics']['data_integrity']
    print(f"  • All directories accessible: {'✅' if integrity['all_directories_accessible'] else '❌'}")
    print(f"  • Annotation files exist: {'✅' if integrity['annotation_files_exist'] else '❌'}")

    print("="*60)

if __name__ == "__main__":
    # Generate report
    report = generate_widerface_report()

    if report:
        # Save detailed JSON report
        save_report(report)

        # Print summary
        print_report_summary(report)

        print("\n✅ WiderFace Face Detection dataset report completed!")
        print(f"📁 Report saved in: reports/{os.path.basename(save_report(report))}")
    else:
        print("❌ Failed to generate report")