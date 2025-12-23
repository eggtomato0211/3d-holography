"""
Plotlyを使った3D可視化のデモスクリプト
使い方: python demo_plotly_visualization.py
"""

from visualize_3d_plotly import visualize_h5_file
import os

# 可視化するh5ファイルのパス
hdf_file = r"c:\Users\Owner\mizusaki\3d-holography\hdf\32x32x128_d=4e-06_pixels=1_2plots_128images\2plots_128images_FalserandomMode_NumberFrom1.h5"

# 出力ディレクトリ（HTMLファイルが保存される）
output_dir = r"C:\Users\Owner\.gemini\antigravity\brain\76a15ec6-a4a9-49b9-b8fd-0d855333ec28"

print("=" * 60)
print("🎨 Plotly 3D可視化デモ")
print("=" * 60)
print()

# 可視化を実行
# visualization_type: 'scatter' (スキャッター), 'volume' (ボリューム), 'both' (両方)
visualize_h5_file(hdf_file, output_dir, visualization_type='scatter')

print()
print("=" * 60)
print("✨ 完了！")
print("=" * 60)
print()
print("📁 出力ディレクトリ:")
print(f"   {output_dir}")
print()
print("🌐 ブラウザで以下のHTMLファイルを開いてください:")
print("   - *_label_3d_scatter.html (Label Dataの3D可視化)")
print("   - *_raw_3d_scatter.html (Raw Dataの3D可視化)")
print("   - *_label_projections.html (Label Dataの投影図)")
print("   - *_raw_projections.html (Raw Dataの投影図)")
print("   - *_bead_distribution.html (ビーズ分布分析)")
print()
print("💡 HTMLファイルはインタラクティブです:")
print("   - マウスドラッグで回転")
print("   - スクロールでズーム")
print("   - ポイントにホバーで詳細表示")
print()
