from moviepy.editor import VideoFileClip

# MP4ファイルのパス
mp4_file_path = 'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\movies\\raw_movie.mp4'

# GIFとして保存するパス
gif_file_path = 'C:\\Users\\Owner\\mizusaki\\3d-holography\\app\\python\\3d-imaging\\gif\\movie.gif'

# MP4動画を読み込む
clip = VideoFileClip(mp4_file_path)

# 動画をGIFに変換して保存
clip.write_gif(gif_file_path, fps=30)

print(f"GIFを{gif_file_path}に保存しました。")
