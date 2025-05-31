import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from PIL import Image
import os
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore")


def plot_pred_target_gif(pred_path, target_path, save_path, cmap='viridis'):
    # 加载预测和标签数据
    pred = np.load(pred_path)  # shape: (T, C, H, W)
    target = np.load(target_path)  # shape: (T, C, H, W)

    # 如果是batch size = 1，直接去掉 batch 维度
    pred = pred.squeeze(axis=0)  # shape: (T, C, H, W)
    target = target.squeeze(axis=0)  # shape: (T, C, H, W)

    T, C, H, W = pred.shape
    frames = []

    for t in range(T):
        pred_frame = pred[t, 0]  # 取第一个 channel
        target_frame = target[t, 0]  # 取第一个 channel

        mse = np.mean((pred_frame - target_frame) ** 2)
        rmse = np.sqrt(mse)

        # 可视化当前帧
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pred_frame, cmap=cmap)
        axes[0].set_title(f'Prediction\nT={t}')
        axes[0].axis('off')

        axes[1].imshow(target_frame, cmap=cmap)
        axes[1].set_title(f'Target\nMSE={mse:.6f} | RMSE={rmse:.6f}')
        axes[1].axis('off')

        fig.tight_layout()

        # 使用 buffer_rgba() 获取图像数据
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # 提取 RGB 通道
        frames.append(Image.fromarray(image))
        plt.close(fig)

    # 保存 GIF
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=400,
        loop=0
    )
    print(f"✅ Saved: {save_path}")


def visualization_weatherbench(data_fp, save_folder='gifs', cmap='viridis'):
    os.makedirs(save_folder, exist_ok=True)
    preds = glob.glob(data_fp + '/pred*.npy')
    targets = glob.glob(data_fp + '/target*.npy')


    for pred_path, target_path in tqdm(zip(preds, targets)):
        name = os.path.basename(pred_path).replace('pred_batch_', '').replace('.npy', '')
        save_path = f'{save_folder}/visual_{name}.gif'
        plot_pred_target_gif(pred_path, target_path, save_path)






def create_gif_grid(pred_dir, target_dir, output_dir='gifs', fps=2):
    os.makedirs(output_dir, exist_ok=True)
    pred_files = sorted(os.listdir(pred_dir))
    target_files = sorted(os.listdir(target_dir))

    # 确保预测和标签文件一一对应
    common_files = sorted(set(pred_files) & set(target_files))
    if not common_files:
        print("未找到匹配的预测和标签文件。")
        return

    for filename in common_files:
        pred_path = os.path.join(pred_dir, filename)
        target_path = os.path.join(target_dir, filename)

        pred = np.load(pred_path)  # 形状: (T, C, H, W)
        target = np.load(target_path)

        if pred.shape != target.shape:
            print(f"文件 {filename} 的预测和标签形状不匹配，跳过。")
            continue
        # pred = pred[0]
        # target = target[0]
        row_start, row_end = int(0.3 * 128), int(0.5 * 128)
        col_start, col_end = int(0.3 * 256), int(0.5 * 256)
        pred = pred[...,row_start:row_end, col_start:col_end]
        target = pred[...,row_start:row_end, col_start:col_end]


        T, C, H, W = pred.shape

        # 计算 MSE 和 RMSE
        mse = 0.0 #mean_squared_error(target.flatten(), pred.flatten())
        rmse = 0.0 #math.sqrt(mse)

        frames = []
        for t in range(T):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            fig.suptitle(f"{filename} - timestamp {t+1}\nMSE: {mse:.6f}, RMSE: {rmse:.6f}", fontsize=12)

            # 显示预测图像
            axes[0].imshow(pred[t, 0], cmap='coolwarm')
            axes[0].set_title("predictions")
            axes[0].axis('off')

            # 显示标签图像
            axes[1].imshow(target[t, 0], cmap='coolwarm')
            axes[1].set_title("targets")
            axes[1].axis('off')

            # 将当前帧保存为图像
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frame = Image.open(buf)
            frames.append(frame)
            plt.close(fig)

        # 保存为 GIF
        gif_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1000//fps, loop=0)
        print(f"已保存 GIF: {gif_path}")
        break



def create_gif_graph(pred_dir, target_dir, station_coord_path, output_dir='gifs', fps=2):
    os.makedirs(output_dir, exist_ok=True)
    station_coords = np.load(station_coord_path)  # shape: (N, 2), [lon, lat]
    pred_files = sorted(os.listdir(pred_dir))
    target_files = sorted(os.listdir(target_dir))

    common_files = sorted(set(pred_files) & set(target_files))
    if not common_files:
        print("没有找到匹配的预测和标签文件。")
        return

    for filename in common_files:
        pred = np.load(os.path.join(pred_dir, filename))  # shape: (T, N, C)
        target = np.load(os.path.join(target_dir, filename))

        if pred.shape != target.shape:
            print(f"文件 {filename} 的形状不一致，跳过")
            continue
        
        T, N, C = pred.shape
        mse = mean_squared_error(target.flatten(), pred.flatten())
        rmse = math.sqrt(mse)

        frames = []
        for t in range(T):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"{filename} - timestamps {t+1}\nMSE: {mse:.6f}, RMSE: {rmse:.6f}", fontsize=12)

            lon, lat = station_coords[:, 0], station_coords[:, 1]

            # 预测图
            scatter1 = axes[0].scatter(lon, lat, c=pred[t, :, 0], cmap='coolwarm', s=80, vmin=np.min(target), vmax=np.max(target))
            axes[0].set_title("predictions")
            fig.colorbar(scatter1, ax=axes[0])
            axes[0].set_xlabel("longitude")
            axes[0].set_ylabel("latitude")

            # 真实标签图
            scatter2 = axes[1].scatter(lon, lat, c=target[t, :, 0], cmap='coolwarm', s=80, vmin=np.min(target), vmax=np.max(target))
            axes[1].set_title("targets")
            fig.colorbar(scatter2, ax=axes[1])
            axes[1].set_xlabel("longitude")
            axes[1].set_ylabel("latitude")

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(Image.open(buf))
            plt.close(fig)

        gif_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1000//fps, loop=0)
        print(f"生成 GIF: {gif_path}")

