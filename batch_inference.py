"""
Script untuk Batch Inference Coffee Grading System
Memproses folder gambar, mendeteksi biji kopi, melakukan klasifikasi,
menyimpan gambar hasil anotasi (opsional dengan Grad-CAM),
dan menghasilkan report dashboard HTML yang interaktif dan premium.
"""
import sys
import os
import argparse
import cv2
import numpy as np
import json
import time
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import ImagePreprocessor
from model import CoffeeGradingModel
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Inference Coffee Grading System")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data",
        help="Path ke folder input gambar (atau subfolder data)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Path ke folder output untuk menyimpan anotasi dan report"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Confidence threshold (default dari Config)"
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Aktifkan Test-Time Augmentation untuk prediksi lebih akurat"
    )
    parser.add_argument(
        "--gradcam", "-g",
        action="store_true",
        help="Sertakan visualisasi Grad-CAM pada biji kopi yang dideteksi"
    )
    parser.add_argument(
        "--no-awb",
        action="store_true",
        help="Nonaktifkan Auto White Balance"
    )
    return parser.parse_args()


def generate_html_report(stats, image_results, output_dir):
    """
    Menghasilkan report dashboard HTML premium dan modern.
    Mendukung jumlah kelas dinamis dari Config.GRADE_LABELS.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.html")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    grade_labels = Config.GRADE_LABELS if Config is not None else ['Grade A', 'Grade B', 'Grade C']
    
    # Calculate totals
    total_graded = sum(stats.get(label, 0) for label in grade_labels)
    
    # Percentages per class
    pcts = {}
    for label in grade_labels:
        pcts[label] = (stats.get(label, 0) / total_graded * 100) if total_graded > 0 else 0
    p_u = (stats['Uncertain'] / stats['total_beans'] * 100) if stats['total_beans'] > 0 else 0
    
    # CSS color palette per class
    css_colors = [
        '#10b981',  # Green   → Normal
        '#ef4444',  # Red     → Biji Hitam
        '#f97316',  # Orange  → Biji Cokelat
        '#06b6d4',  # Cyan    → Berlubang
        '#a855f7',  # Purple  → Pecah
        '#eab308',  # Yellow  → Berjamur
        '#3b82f6',  # Blue    (extra)
        '#ec4899',  # Pink    (extra)
    ]
    css_class_names = ['green', 'red', 'orange', 'cyan', 'purple', 'yellow', 'blue', 'pink']
    
    # Build summary cards HTML
    cards_html = f"""
            <div class="card">
                <div class="card-title">Total Biji Dideteksi</div>
                <div class="card-value">{stats['total_beans']}</div>
                <div class="card-subtitle">Rata-rata {stats['total_beans'] / max(1, stats['total_images']):.1f} biji / gambar</div>
            </div>
    """
    for idx, label in enumerate(grade_labels):
        css_cls = css_class_names[idx % len(css_class_names)]
        css_col = css_colors[idx % len(css_colors)]
        cards_html += f"""
            <div class="card">
                <div class="card-title">{label}</div>
                <div class="card-value" style="color: {css_col};">{stats.get(label, 0)}</div>
                <div class="card-subtitle">{pcts[label]:.1f}% dari total graded</div>
            </div>
        """
    
    # Build bar chart HTML
    bars_html = ""
    for idx, label in enumerate(grade_labels):
        css_col = css_colors[idx % len(css_colors)]
        pct = pcts[label]
        bars_html += f"""
                    <div class="bar-group">
                        <div class="bar-label">
                            <span>{label}</span>
                            <strong>{pct:.1f}%</strong>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: {pct}%; background: {css_col};"></div>
                        </div>
                    </div>
        """
    
    # Premium CSS & HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coffee Grading System - Laporan Analisis Batch</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0c0f12;
            --card-bg: #14191f;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --primary: #d97706;
            --border: #222c37;
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            padding: 40px 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 24px;
            margin-bottom: 40px;
        }}
        h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #fbbf24 0%, #d97706 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .meta-info {{
            text-align: right;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            transition: transform 0.2s, border-color 0.2s;
        }}
        .card:hover {{
            transform: translateY(-2px);
            border-color: var(--primary);
        }}
        .card-title {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            margin-bottom: 12px;
        }}
        .card-value {{
            font-size: 2rem;
            font-weight: 700;
        }}
        .card-subtitle {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        .dashboard-content {{
            display: grid;
            grid-template-columns: 3fr 2fr;
            gap: 30px;
            margin-bottom: 40px;
        }}
        @media(max-width: 900px) {{
            .dashboard-content {{
                grid-template-columns: 1fr;
            }}
        }}
        .section-title {{
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .table-container {{
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        th, td {{
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background-color: rgba(0,0,0,0.15);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        tr:hover {{
            background-color: rgba(255,255,255,0.02);
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge.green {{ background-color: rgba(16,185,129,0.15); color: #10b981; }}
        .badge.yellow {{ background-color: rgba(245,158,11,0.15); color: #f59e0b; }}
        .chart-container {{
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 30px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .bar-group {{
            margin-bottom: 20px;
        }}
        .bar-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            margin-bottom: 8px;
        }}
        .bar-track {{
            background-color: rgba(255,255,255,0.05);
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 1s ease-in-out;
        }}
        .issue-text {{
            color: #f59e0b;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        .link-image {{
            color: var(--primary);
            text-decoration: none;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        .link-image:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Laporan Grading Biji Kopi</h1>
                <p style="color: var(--text-secondary); margin-top: 4px;">Sistem Analisis Batch — {len(grade_labels)} Kelas Defect</p>
            </div>
            <div class="meta-info">
                <p>Waktu Analisis: <strong>{timestamp}</strong></p>
                <p>Total Gambar: <strong>{stats['total_images']}</strong></p>
            </div>
        </header>

        <div class="grid">
            {cards_html}
        </div>

        <div class="dashboard-content">
            <div>
                <h2 class="section-title">Detail Hasil Gambar</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Nama File</th>
                                <th>Biji Dideteksi</th>
                                <th>Distribusi Kelas</th>
                                <th>Status Kualitas</th>
                                <th>Aksi</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
    for res in image_results:
        # Build distribution text dynamically
        dist_parts = []
        for label in grade_labels:
            short_name = label.split()[-1] if len(label.split()) > 1 else label
            dist_parts.append(f"{short_name[:6]}:{res.get(label, 0)}")
        dist_text = " | ".join(dist_parts)
        if res.get('Uncertain', 0) > 0:
            dist_text += f" | U:{res['Uncertain']}"
            
        quality_status = ""
        if res['quality_ok']:
            quality_status = '<span class="badge green">Normal</span>'
        else:
            issues = ", ".join(res['quality_issues'])
            quality_status = f'<span class="badge yellow" title="{issues}">Issue</span><br><span class="issue-text">{issues}</span>'
            
        rel_annotated_path = os.path.basename(res['annotated_path']) if res['annotated_path'] else "#"
        
        html_content += f"""
                            <tr>
                                <td>{res['filename']}</td>
                                <td>{res['beans_count']}</td>
                                <td><strong style="font-size:0.85rem;">{dist_text}</strong></td>
                                <td>{quality_status}</td>
                                <td>
                                    {"-" if not res['annotated_path'] else f'<a class="link-image" href="{rel_annotated_path}" target="_blank">Lihat Hasil &rarr;</a>'}
                                </td>
                            </tr>
        """
        
    html_content += f"""
                        </tbody>
                    </table>
                </div>
            </div>

            <div>
                <h2 class="section-title">Distribusi Kualitas Kopi</h2>
                <div class="chart-container">
                    {bars_html}
                    
                    <div style="margin-top: 20px; border-top: 1px solid var(--border); padding-top: 20px; font-size: 0.85rem; color: var(--text-secondary);">
                        <p>Total Graded: <strong>{total_graded} biji</strong></p>
                        <p>Total Uncertain (Low Confidence): <strong>{stats['Uncertain']} ({p_u:.1f}%)</strong></p>
                        <p>Total File Bermasalah Kualitas: <strong>{stats['quality_warnings']} gambar</strong></p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\nHTML Report berhasil digenerate di: {report_path}")


def main():
    args = parse_args()
    
    # Input folder check
    input_dir = args.input
    # Support checking data subfolder as fallback
    if not os.path.exists(input_dir):
        alt_path = os.path.join(Config.DATA_DIR if Config is not None else "data", input_dir)
        if os.path.exists(alt_path):
            input_dir = alt_path
        else:
            print(f"ERROR: Input directory '{args.input}' tidak ditemukan")
            return
            
    print("=" * 60)
    print("BATCH INFERENCE COFFEE GRADING SYSTEM")
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {args.output}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(args.output, exist_ok=True)
    
    # Setup confidence threshold
    conf_thresh = args.threshold
    if conf_thresh is None:
        conf_thresh = Config.CONFIDENCE_THRESHOLD if Config is not None else 0.6
        
    awb_enabled = not args.no_awb
    
    # Initialize components
    preprocessor = ImagePreprocessor()
    
    # Load models
    model_path = Config.MODEL_SAVE_PATH if Config is not None else "models/coffee_grading_model.h5"
    if not os.path.exists(model_path):
        print(f"WARNING: Model tidak ditemukan di {model_path}. Membuat model baru (belum terlatih).")
        model = CoffeeGradingModel()
    else:
        model = CoffeeGradingModel(model_path)
        
    # Stats trackers — dynamic from Config
    grade_labels = Config.GRADE_LABELS if Config is not None else ['Grade A', 'Grade B', 'Grade C']
    stats = {
        'total_images': 0,
        'total_beans': 0,
        'Uncertain': 0,
        'quality_warnings': 0
    }
    for label in grade_labels:
        stats[label] = 0
    
    image_results = []
    
    # Supported extensions
    supported_exts = Config.SUPPORTED_FORMATS if Config is not None else ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    # Walk directories or list files
    image_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(supported_exts):
                image_files.append(os.path.join(root, f))
                
    if len(image_files) == 0:
        print("ERROR: Tidak ada gambar yang didukung untuk diproses.")
        return
        
    print(f"Menemukan {len(image_files)} gambar untuk di-analisis...")
    print("Memulai pemrosesan...")
    
    start_time = time.time()
    
    for idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        print(f"\n[{idx+1}/{len(image_files)}] Memproses {filename}...")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  FAILED: Gagal membaca gambar")
            continue
            
        stats['total_images'] += 1
        
        # AWB
        if awb_enabled:
            img_processed = preprocessor.apply_auto_white_balance(img)
        else:
            img_processed = img.copy()
            
        # Quality check
        quality_ok, quality_reasons = preprocessor.check_image_quality(img_processed)
        if not quality_ok:
            print(f"  WARNING Kualitas: {', '.join(quality_reasons['issues'])}")
            stats['quality_warnings'] += 1
            
        # Detect beans
        detected_beans = preprocessor.detect_coffee_beans(img_processed)
        # Apply NMS
        detected_beans = preprocessor.apply_nms(detected_beans, iou_threshold=0.3)
        
        beans_count = len(detected_beans)
        stats['total_beans'] += beans_count
        print(f"  Dideteksi {beans_count} biji kopi")
        
        annotated_img = img_processed.copy()
        
        res_counters = {label: 0 for label in grade_labels}
        res_counters['Uncertain'] = 0
        
        # Colors (BGR) — dynamic
        _color_palette_bgr = [
            (0, 200, 0),      # Green
            (0, 0, 220),      # Red
            (0, 120, 220),    # Dark Orange
            (220, 180, 0),    # Cyan
            (180, 0, 180),    # Purple
            (0, 180, 220),    # Yellow
        ]
        grade_colors = {}
        for i, label in enumerate(grade_labels):
            grade_colors[label] = _color_palette_bgr[i % len(_color_palette_bgr)]
        grade_colors['Uncertain'] = (128, 128, 128)
        
        for bean in detected_beans:
            bbox = bean['bbox']
            contour = bean['contour']
            
            roi = preprocessor.extract_bean_roi(img_processed, bbox)
            if roi is not None:
                # Normalize
                normalized_roi = preprocessor.normalize_image(roi)
                
                # Predict
                if args.tta:
                    grade, confidence, probs = model.predict_with_tta(normalized_roi)
                else:
                    grade, confidence, probs = model.predict(normalized_roi)
                    
                # Apply confidence threshold
                if confidence < conf_thresh:
                    final_grade = 'Uncertain'
                else:
                    final_grade = grade
                    
                # Update counters
                res_counters[final_grade] += 1
                stats[final_grade] += 1
                
                # Apply Grad-CAM if requested
                if args.gradcam:
                    try:
                        predicted_class = np.argmax(probs)
                        heatmap, _ = model.get_gradcam_heatmap(normalized_roi, pred_index=predicted_class)
                        gradcam_roi = model.overlay_gradcam(roi, heatmap, alpha=0.4)
                        x, y, w, h = bbox
                        gradcam_roi_resized = cv2.resize(gradcam_roi, (w, h))
                        annotated_img[y:y+h, x:x+w] = gradcam_roi_resized
                    except Exception as e:
                        pass
                
                # Draw contour
                color = grade_colors.get(final_grade, (255, 255, 255))
                cv2.drawContours(annotated_img, [contour], -1, color, 2)
                
                # Draw Bounding Box & Label
                x, y, w, h = bbox
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 1)
                
                label = f"{final_grade} ({confidence:.0%})"
                cv2.putText(annotated_img, label, (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            
        # Save annotated image
        annotated_path = None
        if beans_count > 0:
            annotated_filename = f"annotated_{filename}"
            annotated_path = os.path.join(args.output, annotated_filename)
            cv2.imwrite(annotated_path, annotated_img)
            
        image_result = {
            'filename': filename,
            'beans_count': beans_count,
            'Uncertain': res_counters.get('Uncertain', 0),
            'quality_ok': quality_ok,
            'quality_issues': quality_reasons['issues'],
            'annotated_path': annotated_path
        }
        for label in grade_labels:
            image_result[label] = res_counters.get(label, 0)
        image_results.append(image_result)
        
    duration = time.time() - start_time
    print(f"\nAnalisis selesai dalam {duration:.1f} detik.")
    
    # Save statistics JSON
    stats_json_path = os.path.join(args.output, "batch_statistics.json")
    with open(stats_json_path, "w") as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'duration_seconds': duration,
            'results': [
                {k: v for k, v in r.items() if k != 'annotated_path'}
                for r in image_results
            ]
        }, f, indent=4)
    print(f"Statistik lengkap disave ke JSON: {stats_json_path}")
    
    # Generate HTML Report dashboard
    generate_html_report(stats, image_results, args.output)
    
    print("\nBatch processing berhasil dilakukan sepenuhnya!")


if __name__ == "__main__":
    main()
