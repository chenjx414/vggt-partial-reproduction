import os, sys, subprocess

CO3D_DIR = r"D:/vggt"            # parkingmeter 文件夹的上一级
ANNO_DIR = r"D:/vggt/anno"       # 标注输出目录（会自动创建）
MODEL    = r"D:/vggt/code/ckpts/VGGT-1B/model.pt"   # ← 改成你 model.pt 的实际路径

os.makedirs(ANNO_DIR, exist_ok=True)

# 第一步：生成标注 parkingmeter_test.jgz
subprocess.run([sys.executable, r"D:/vggt/code/preprocess_co3d.py",
                "--category", "parkingmeter",
                "--co3d_v2_dir", CO3D_DIR,
                "--output_dir", ANNO_DIR], check=True)

# 第二步：评估（--debug = 只跑 parkingmeter）
subprocess.run([sys.executable, r"D:/vggt/code/test_co3d.py",
                "--debug", "--fast_eval",
                "--model_path", MODEL,
                "--co3d_dir", CO3D_DIR,
                "--co3d_anno_dir", ANNO_DIR,
                "--seed", "0"], check=True)