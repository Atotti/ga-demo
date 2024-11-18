import cv2
import os

def images_to_video(output_dir, video_filename='genetic_evolution.avi', fps=10):
    # 出力画像ファイルを取得して並べ替える
    images = sorted([img for img in os.listdir(output_dir) if img.endswith(".jpg")])

    # 最初の画像でサイズを取得
    frame = cv2.imread(os.path.join(output_dir, images[0]))
    height, width, layers = frame.shape

    # ビデオライターのセットアップ
    video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(output_dir, image))
        video.write(frame)

    video.release()
    print(f"動画が保存されました: {video_filename}")
