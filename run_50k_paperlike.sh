#!/usr/bin/env bash
set -euo pipefail

# Activate conda
source /home/shencc/miniforge3/etc/profile.d/conda.sh
conda activate street-gaussian

# ===== User-configurable paths =====
# Waymo validation set directory containing .tfrecord files
RAW_DIR=/data/shencc/waymo_val_raw
# Output processed data
PROC_DIR=/data/shencc/waymo_processed/validation
# Waymo splits in this repo
SPLIT_DIR=/home/shencc/projects/street_gaussians/script/waymo/waymo_splits
# Tracking file (paper uses tracker results) - set this to your tracker json
TRACK_FILE=/data/shencc/waymo_tracker/val/result.json

# Optional: limit to a subset by customizing split file / segment file
SPLIT_FILE="$SPLIT_DIR/val_dynamic.txt"
SEGMENT_FILE="$SPLIT_DIR/segment_list_val.txt"

# ===== Dependencies =====
# Core deps (idempotent). You may already have these.
pip install opencv-python==4.7.0.72 imageio imageio-ffmpeg tqdm matplotlib pyyaml roma plyfile bidict scikit-learn open3d
pip install "protobuf==3.20.3"
pip install "numpy==1.26.4"

# COLMAP must be installed and on PATH for paper-like initialization
if ! command -v colmap >/dev/null 2>&1; then
  echo "ERROR: colmap not found in PATH. Install COLMAP or disable use_colmap."
  exit 1
fi

# ===== Convert Waymo raw -> processed =====
mkdir -p "$RAW_DIR" "$PROC_DIR"
cd /home/shencc/projects/street_gaussians

python script/waymo/waymo_converter.py \
  --root_dir "$RAW_DIR" \
  --save_dir "$PROC_DIR" \
  --split_file "$SPLIT_FILE" \
  --segment_file "$SEGMENT_FILE" \
  --track_file "$TRACK_FILE"

# ===== Generate LiDAR depth (for all processed scenes) =====
for SCENE_DIR in "$PROC_DIR"/*; do
  if [ -d "$SCENE_DIR" ]; then
    python script/waymo/generate_lidar_depth.py --datadir "$SCENE_DIR"
  fi
done

# ===== Config (paper-like defaults) =====
mkdir -p /home/shencc/projects/street_gaussians/configs/custom
cat > /home/shencc/projects/street_gaussians/configs/custom/waymo_val_50k_paperlike.yaml << EOS
task: waymo_full_exp
source_path: /data/shencc/waymo_processed/validation/000
exp_name: waymo_val_50k_paperlike

data:
  split_test: -1
  split_train: 1
  type: Waymo
  white_background: false
  selected_frames: null
  cameras: [0, 1, 2]
  extent: 10
  use_colmap: true
  filter_colmap: true

model:
  gaussian:
    sh_degree: 1
    fourier_dim: 5
    fourier_scale: 1.
    flip_prob: 0.5
  nsg:
    include_bkgd: true
    include_obj: true
    include_sky: true
    opt_track: true

train:
  iterations: 50000
  test_iterations: [7000, 30000, 50000]
  save_iterations: [50000]
  checkpoint_iterations: [50000]

optim:
  prune_box_interval: 100
  densification_interval: 100
  densify_from_iter: 500
  densify_grad_threshold_bkgd: 0.0006
  densify_grad_abs_bkgd: True
  densify_grad_threshold_obj: 0.0002
  densify_grad_abs_obj: False
  densify_grad_threshold: 0.0002
  densify_until_iter: 25000
  feature_lr: 0.0025
  max_screen_size: 20
  min_opacity: 0.005
  opacity_lr: 0.05
  opacity_reset_interval: 3000
  percent_big_ws: 0.1
  percent_dense: 0.01
  position_lr_delay_mult: 0.01
  position_lr_final: 1.6e-06
  position_lr_init: 0.00016
  position_lr_max_steps: 50000
  rotation_lr: 0.001
  scaling_lr: 0.005
  semantic_lr: 0.01

  lambda_dssim: 0.2
  lambda_sky: 0.05
  lambda_sky_scale: [1, 1, 0]
  lambda_mask: 0.1
  lambda_reg: 0.1
  lambda_depth_lidar: 0.1

  track_position_lr_delay_mult: 0.01
  track_position_lr_init: 0.005
  track_position_lr_final: 5.0e-5
  track_position_max_steps: 30000

  track_rotation_lr_delay_mult: 0.01
  track_rotation_lr_init: 0.001
  track_rotation_lr_final: 1.0e-5
  track_rotation_max_steps: 30000

render:
  fps: 24
  concat_cameras: [1, 0, 2]
EOS

# ===== GPU monitoring =====
LOG_DIR=/home/shencc/projects/street_gaussians/output/waymo_full_exp/waymo_val_50k_paperlike
mkdir -p "$LOG_DIR"
GPU_LOG="$LOG_DIR/gpu_usage.csv"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv -l 1 > "$GPU_LOG" &
GPU_PID=$!

cleanup() {
  if ps -p "$GPU_PID" >/dev/null 2>&1; then
    kill "$GPU_PID" || true
  fi
}
trap cleanup EXIT

# ===== Train =====
python train.py --config configs/custom/waymo_val_50k_paperlike.yaml

# ===== Render =====
python render.py --config configs/custom/waymo_val_50k_paperlike.yaml mode evaluate
python render.py --config configs/custom/waymo_val_50k_paperlike.yaml mode trajectory

# ===== Metrics =====
python metrics.py --config configs/custom/waymo_val_50k_paperlike.yaml eval.eval_train True eval.eval_test False

# ===== Viewer export =====
python make_ply.py --config configs/custom/waymo_val_50k_paperlike.yaml viewer.frame_id 0 mode evaluate

