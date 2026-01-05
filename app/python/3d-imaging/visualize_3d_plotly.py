import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def load_h5_data(file_path):
    """HDF5ファイルからデータを読み込む"""
    with h5py.File(file_path, 'r') as f:
        raw_data = f['raw'][:]
        label_data = f['label'][:]
    return raw_data, label_data

def create_3d_scatter(data, title, colorscale='Hot', threshold_percentile=1, opacity=0.6, point_size=2):
    """3Dスキャッタープロットを作成"""
    # 閾値を設定
    threshold = np.percentile(data[data > 0], threshold_percentile) if np.any(data > 0) else 0
    
    # 閾値以上の点を抽出
    z, y, x = np.where(data > threshold)
    values = data[data > threshold]
    
    # 3Dスキャッタープロット
    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=point_size,
            color=values,
            colorscale=colorscale,
            opacity=opacity,
            colorbar=dict(
                title="強度",
                thickness=20,
                len=0.7
            ),
            line=dict(width=0)  # エッジを削除してクリーンに
        ),
        text=[f'X:{xi}, Y:{yi}, Z:{zi}<br>値:{v:.3f}' for xi, yi, zi, v in zip(x, y, z, values)],
        hovertemplate='%{text}<extra></extra>',
        name=title
    )
    
    return scatter

def create_volume_rendering(data, title, colorscale='Hot', opacity=0.1, surface_count=15):
    """ボリュームレンダリングを作成"""
    # データを正規化
    data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-10)
    
    # ボリュームプロット
    volume = go.Volume(
        x=np.arange(data.shape[2]).repeat(data.shape[0] * data.shape[1]),
        y=np.tile(np.arange(data.shape[1]).repeat(data.shape[0]), data.shape[2]),
        z=np.tile(np.arange(data.shape[0]), data.shape[1] * data.shape[2]),
        value=data_normalized.flatten(),
        isomin=0.1,
        isomax=1.0,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=colorscale,
        colorbar=dict(
            title="強度",
            thickness=20,
            len=0.7
        ),
        name=title
    )
    
    return volume

def create_projection_heatmaps(data, title_prefix):
    """XY, XZ, YZ投影のヒートマップを作成"""
    # 投影を計算
    projection_xy = np.max(data, axis=0)  # Z軸方向に最大値投影
    projection_xz = np.max(data, axis=1)  # Y軸方向に最大値投影
    projection_yz = np.max(data, axis=2)  # X軸方向に最大値投影
    
    # サブプロットを作成
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(f'{title_prefix} - XY投影 (上から)', 
                       f'{title_prefix} - XZ投影 (横から)', 
                       f'{title_prefix} - YZ投影 (正面から)'),
        horizontal_spacing=0.1
    )
    
    # XY投影
    fig.add_trace(
        go.Heatmap(z=projection_xy, colorscale='Hot', showscale=True, 
                   colorbar=dict(x=0.3, len=0.9)),
        row=1, col=1
    )
    
    # XZ投影
    fig.add_trace(
        go.Heatmap(z=projection_xz, colorscale='Hot', showscale=True,
                   colorbar=dict(x=0.65, len=0.9)),
        row=1, col=2
    )
    
    # YZ投影
    fig.add_trace(
        go.Heatmap(z=projection_yz.T, colorscale='Hot', showscale=True,
                   colorbar=dict(x=1.0, len=0.9)),
        row=1, col=3
    )
    
    # レイアウト更新
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="Z", row=1, col=2)
    fig.update_xaxes(title_text="Y", row=1, col=3)
    fig.update_yaxes(title_text="Z", row=1, col=3)
    
    fig.update_layout(
        height=400,
        title_text=f"{title_prefix} - 最大値投影",
        showlegend=False
    )
    
    return fig

def analyze_bead_distribution(label_data):
    """ビーズの分布を分析"""
    non_zero_slices = []
    bead_info = []
    
    for z in range(label_data.shape[0]):
        if np.max(label_data[z]) > 0:
            non_zero_slices.append(z)
            num_beads = np.sum(label_data[z] > 0.01)
            max_val = np.max(label_data[z])
            bead_info.append({
                'z': z,
                'count': num_beads,
                'max_value': max_val
            })
    
    return non_zero_slices, bead_info

def create_bead_distribution_plot(bead_info):
    """ビーズ分布のプロットを作成"""
    if not bead_info:
        return None
    
    z_positions = [info['z'] for info in bead_info]
    bead_counts = [info['count'] for info in bead_info]
    max_values = [info['max_value'] for info in bead_info]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('各層のビーズ数', 'ビーズ数のヒストグラム'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # 各層のビーズ数
    fig.add_trace(
        go.Scatter(
            x=z_positions, 
            y=bead_counts,
            mode='markers+lines',
            marker=dict(
                size=8,
                color=max_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="最大強度", x=0.45)
            ),
            line=dict(color='rgba(100,100,100,0.3)'),
            name='ビーズ数'
        ),
        row=1, col=1
    )
    
    # ヒストグラム
    fig.add_trace(
        go.Histogram(
            x=bead_counts,
            nbinsx=int(max(bead_counts)) if bead_counts else 10,
            marker=dict(color='rgba(255,100,100,0.7)'),
            name='分布'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Z位置 (深さ)", row=1, col=1)
    fig.update_yaxes(title_text="ビーズ数", row=1, col=1)
    fig.update_xaxes(title_text="ビーズ数/層", row=1, col=2)
    fig.update_yaxes(title_text="層の数", row=1, col=2)
    
    avg_beads = np.mean(bead_counts)
    fig.update_layout(
        height=400,
        title_text=f"ビーズ分布分析 (平均: {avg_beads:.1f}個/層)",
        showlegend=False
    )
    
    return fig

def visualize_h5_file(file_path, output_dir=None, visualization_type='scatter'):
    """
    HDF5ファイルを可視化
    
    Parameters:
    -----------
    file_path : str
        HDF5ファイルのパス
    output_dir : str, optional
        出力ディレクトリ（指定しない場合はファイルと同じディレクトリ）
    visualization_type : str
        'scatter' (スキャッター), 'volume' (ボリューム), 'both' (両方)
    """
    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # データ読み込み
    print(f"📂 読み込み中: {file_path}\n")
    raw_data, label_data = load_h5_data(file_path)
    
    # データ情報を表示
    print(f"📊 データ形状: {label_data.shape}")
    print(f"   Label範囲: {np.min(label_data):.4f} ～ {np.max(label_data):.4f}")
    print(f"   Raw範囲: {np.min(raw_data):.4f} ～ {np.max(raw_data):.4f}\n")
    
    # ビーズ分布を分析
    non_zero_slices, bead_info = analyze_bead_distribution(label_data)
    print(f"🔍 合計: {len(non_zero_slices)}層にビーズが配置されています\n")
    
    # ファイル名のベース
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # === Label Data の可視化 ===
    print("🎨 Label Dataを可視化中...")
    
    if visualization_type in ['scatter', 'both']:
        # 3Dスキャッタープロット
        fig_label_scatter = go.Figure(data=[
            create_3d_scatter(label_data, 'Label Data', colorscale='Hot', 
                            threshold_percentile=1, opacity=0.8, point_size=3)
        ])
        
        fig_label_scatter.update_layout(
            title=f'Label Data - 3D可視化<br><sub>{len(non_zero_slices)}層にビーズ配置</sub>',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (深さ)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=1000,
            height=800,
            template='plotly_dark'
        )
        
        output_path = os.path.join(output_dir, f'{base_name}_label_3d_scatter.html')
        fig_label_scatter.write_html(output_path)
        print(f"   ✅ 保存: {output_path}")
    
    if visualization_type in ['volume', 'both']:
        # ボリュームレンダリング
        fig_label_volume = go.Figure(data=[
            create_volume_rendering(label_data, 'Label Data', colorscale='Hot', 
                                  opacity=0.15, surface_count=20)
        ])
        
        fig_label_volume.update_layout(
            title=f'Label Data - ボリュームレンダリング',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (深さ)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=1000,
            height=800,
            template='plotly_dark'
        )
        
        output_path = os.path.join(output_dir, f'{base_name}_label_volume.html')
        fig_label_volume.write_html(output_path)
        print(f"   ✅ 保存: {output_path}")
    
    # 投影図
    fig_label_proj = create_projection_heatmaps(label_data, 'Label Data')
    fig_label_proj.update_layout(template='plotly_dark')
    output_path = os.path.join(output_dir, f'{base_name}_label_projections.html')
    fig_label_proj.write_html(output_path)
    print(f"   ✅ 保存: {output_path}")
    
    # === Raw Data の可視化 ===
    print("\n🎨 Raw Dataを可視化中...")
    
    if visualization_type in ['scatter', 'both']:
        # 3Dスキャッタープロット
        fig_raw_scatter = go.Figure(data=[
            create_3d_scatter(raw_data, 'Raw Data', colorscale='Viridis', 
                            threshold_percentile=30, opacity=0.4, point_size=1)
        ])
        
        fig_raw_scatter.update_layout(
            title='Raw Data - 3D可視化 (シミュレーション結果)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (深さ)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=1000,
            height=800,
            template='plotly_dark'
        )
        
        output_path = os.path.join(output_dir, f'{base_name}_raw_3d_scatter.html')
        fig_raw_scatter.write_html(output_path)
        print(f"   ✅ 保存: {output_path}")
    
    if visualization_type in ['volume', 'both']:
        # ボリュームレンダリング
        fig_raw_volume = go.Figure(data=[
            create_volume_rendering(raw_data, 'Raw Data', colorscale='Viridis', 
                                  opacity=0.1, surface_count=15)
        ])
        
        fig_raw_volume.update_layout(
            title='Raw Data - ボリュームレンダリング',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (深さ)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=1000,
            height=800,
            template='plotly_dark'
        )
        
        output_path = os.path.join(output_dir, f'{base_name}_raw_volume.html')
        fig_raw_volume.write_html(output_path)
        print(f"   ✅ 保存: {output_path}")
    
    # 投影図
    fig_raw_proj = create_projection_heatmaps(raw_data, 'Raw Data')
    fig_raw_proj.update_layout(template='plotly_dark')
    output_path = os.path.join(output_dir, f'{base_name}_raw_projections.html')
    fig_raw_proj.write_html(output_path)
    print(f"   ✅ 保存: {output_path}")
    
    # === ビーズ分布分析 ===
    if bead_info:
        print("\n📈 ビーズ分布を分析中...")
        fig_bead_dist = create_bead_distribution_plot(bead_info)
        if fig_bead_dist:
            fig_bead_dist.update_layout(template='plotly_dark')
            output_path = os.path.join(output_dir, f'{base_name}_bead_distribution.html')
            fig_bead_dist.write_html(output_path)
            print(f"   ✅ 保存: {output_path}")
    
    print("\n✨ 完了！ブラウザでHTMLファイルを開いてインタラクティブに操作できます。")

if __name__ == "__main__":
    # 使用例
    # ファイルパスを指定してください
    hdf_file = r"D:\nosaka\data\3d-holography_output\Train\random_32x32x128_d=4e-06_pixels=1_1plot"
    
    # 出力ディレクトリ
    output_dir = r"C:\Users\Owner\.gemini\antigravity\brain\76a15ec6-a4a9-49b9-b8fd-0d855333ec28"
    
    # 可視化を実行
    # visualization_type: 'scatter', 'volume', 'both'
    visualize_h5_file(hdf_file, output_dir, visualization_type='scatter')
