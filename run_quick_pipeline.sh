#!/usr/bin/env bash
set -euo pipefail

# Activate conda
source /home/shencc/miniforge3/etc/profile.d/conda.sh
conda activate street-gaussian

# Optional: install deps (idempotent)
pip install opencv-python==4.7.0.72 imageio imageio-ffmpeg tqdm matplotlib pyyaml roma plyfile bidict scikit-learn open3d
pip install "protobuf==3.20.3"
pip install "numpy==1.26.4"

# Paths
RAW_DIR=/data/shencc/waymo_raw
PROC_DIR=/data/shencc/waymo_processed/training
SPLIT_DIR=/data/shencc/waymo_splits
TFRECORD_SRC=/data/shencc/individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
TFRECORD_DST=$RAW_DIR/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord

# Prepare directories and splits
mkdir -p "$RAW_DIR" "$PROC_DIR" "$SPLIT_DIR"
if [ ! -e "$TFRECORD_DST" ]; then
  ln -s "$TFRECORD_SRC" "$TFRECORD_DST" || cp "$TFRECORD_SRC" "$TFRECORD_DST"
fi

cat > "$SPLIT_DIR/single.txt" << EOS
# scene_id, seg_name, start_timestep, end_timestep, scene_type
0,seg100170,0,-1,dynamic
EOS

cat > "$SPLIT_DIR/segment_list_single.txt" << EOS
segment-10017090168044687777_6380_000_6400_000_with_camera_labels
EOS

# Convert Waymo raw -> processed
cd /home/shencc/projects/street_gaussians
python script/waymo/waymo_converter.py \
  --root_dir "$RAW_DIR" \
  --save_dir "$PROC_DIR" \
  --split_file "$SPLIT_DIR/single.txt" \
  --segment_file "$SPLIT_DIR/segment_list_single.txt" \
  --track_file /dev/null

# Generate LiDAR depth
python script/waymo/generate_lidar_depth.py \
  --datadir "$PROC_DIR/000"

# Create quick config
mkdir -p /home/shencc/projects/street_gaussians/configs/custom
cat > /home/shencc/projects/street_gaussians/configs/custom/waymo_train_000_quick.yaml << EOS
task: waymo_full_exp
source_path: /data/shencc/waymo_processed/training/000
exp_name: waymo_train_000_quick

data:
  split_test: -1
  split_train: 1
  type: Waymo
  white_background: false
  selected_frames: null
  cameras: [0, 1, 2]
  extent: 10
  use_colmap: false
  filter_colmap: false

model:
  gaussian:
    sh_degree: 1
    fourier_dim: 5
    fourier_scale: 1.
    flip_prob: 0.5
  nsg:
    include_bkgd: true
    include_obj: true
    include_sky: false
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
  lambda_sky: 0.0
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

# GPU monitoring
LOG_DIR=/home/shencc/projects/street_gaussians/output/waymo_full_exp/waymo_train_000_quick
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

# Train
python train.py --config configs/custom/waymo_train_000_quick.yaml

# Render (evaluate)
python render.py --config configs/custom/waymo_train_000_quick.yaml mode evaluate

# Metrics (train)
python metrics.py --config configs/custom/waymo_train_000_quick.yaml eval.eval_train True eval.eval_test False

# Export PLY for viewer
python make_ply.py --config configs/custom/waymo_train_000_quick.yaml viewer.frame_id 0 mode evaluate

# Trajectory video
python render.py --config configs/custom/waymo_train_000_quick.yaml mode trajectory

