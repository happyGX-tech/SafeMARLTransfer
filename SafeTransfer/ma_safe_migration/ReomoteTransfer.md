# SafeTransfer 远程迁移指导（Transfer.md）

本指南用于将当前项目从本地迁移到远程 Linux 服务器，并确保 CTCE + FSRL 训练与采集流程可复现。

适用范围：
- 项目目录：SafeTransfer/ma_safe_migration
- 依赖本地源码：SRL/safety-gymnasium-main
- 运行环境：Python 3.8 + Conda

---

## 1. 迁移目标

迁移完成后，应满足以下能力：

1. 可成功创建并运行 Safety-Gymnasium 多智能体环境。
2. 可执行 FSRL 行为策略训练（CTCE 模式）。
3. 可执行离线数据采集（FSRL checkpoint，CTCE Collector 模式）。
4. 可将采集结果导出为 OSRL/DSRL 可读 NPZ 数据。
4. 可稳定保存日志、checkpoint、离线数据。

---

## 2. 迁移前检查（本地）

### 2.1 固化代码与环境信息

在本地记录以下信息，便于失败时回滚：

1. 当前分支与提交号
2. 本地未提交改动列表
3. Conda 环境导出
4. Pip 依赖锁定

建议命令（Windows PowerShell）：

```powershell
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status

conda env export -n SafeMARL > env_safemarl.yml
pip freeze > requirements_lock.txt
```

### 2.2 明确需要迁移的目录

必须迁移：

1. SafeTransfer/ma_safe_migration
2. SRL/safety-gymnasium-main

可选迁移（如果希望复用历史成果）：

1. SafeTransfer/ma_safe_migration/third_party/FSRL/logs_local
2. SafeTransfer/ma_safe_migration/data

不建议迁移：

1. __pycache__
2. 临时截图/临时渲染文件
3. 本地 IDE 缓存

---

## 3. 目录组织建议（服务器）

建议在服务器保持与本地一致的相对路径：

```text
/workspace/SafeMARL/
  SafeTransfer/ma_safe_migration
  SRL/safety-gymnasium-main
```

原因：
- 项目使用了 editable 安装（pip install -e ../SRL/safety-gymnasium-main）
- 保持相对路径可减少脚本修改与路径错误

---

## 4. 传输方案

### 方案 A（推荐）：Git 仓库 + 服务器重建

优点：
- 干净、可追踪、便于后续更新

流程：

1. 本地提交/推送代码
2. 服务器 clone
3. 服务器安装依赖
4. 回归测试

### 方案 B：打包上传（适合大量未提交改动）

本地打包示例（Windows PowerShell）：

```powershell
Set-Location D:\RL-lab\safe RL\SafeMARL
Compress-Archive -Path .\SafeTransfer\ma_safe_migration, .\SRL\safety-gymnasium-main -DestinationPath .\safemarl_transfer.zip -Force
```

上传示例（本地执行）：

```bash
scp safemarl_transfer.zip user@server:/workspace/
```

服务器解压示例：

```bash
cd /workspace
unzip safemarl_transfer.zip
```

---

## 5. 服务器初始化（Ubuntu 示例）

### 5.1 系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential cmake pkg-config \
  libgl1-mesa-glx libgl1-mesa-dev libglew-dev libosmesa6-dev \
  libglfw3 libglfw3-dev patchelf ffmpeg xvfb
```

### 5.2 Conda 环境

```bash
conda create -n SafeMARL python=3.8 -y
conda activate SafeMARL
python -m pip install --upgrade pip setuptools wheel
```

### 5.3 PyTorch

CPU 版（更稳妥）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

如果是 NVIDIA GPU，按驱动和 CUDA 版本改为对应源。

### 5.4 项目依赖安装

```bash
cd /workspace/SafeMARL/SafeTransfer/ma_safe_migration

pip install -r requirements.txt
pip install -e ../SRL/safety-gymnasium-main
pip install -e third_party/FSRL
pip install -e third_party/DSRL
pip install mujoco glfw xmltodict gymnasium-robotics
```

### 5.5 无头服务器渲染变量（推荐）

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

建议写入 ~/.bashrc 持久化。

---

## 6. 迁移后验证（必须执行）

在 /workspace/SafeMARL/SafeTransfer/ma_safe_migration 目录下执行。

### 6.1 环境可用性验证

```bash
python verify_safety_gym_backend.py --num-agents 3 --steps 20 --render-mode rgb_array --randomize-layout
```

通过标准：
- 不报 ImportError / ModuleNotFoundError
- 正常输出 rewards/costs 并完成 Verification completed

### 6.2 数据采集 smoke test（FSRL checkpoint）

```bash
python third_party/FSRL/examples/customized/collect_dataset.py \
  --backend safety_gym \
  --num_agents 3 \
  --collect_only True \
  --num_episodes 2 \
  --policy_checkpoint ./third_party/FSRL/logs_local/ctce_smoke_train/checkpoint/model.pt \
  --output_dir ./data/ctce_smoke_remote \
  --dataset_name offline_dataset_ctce.hdf5 \
  --seed 7
```

通过标准：
- 生成 `offline_dataset_ctce.hdf5`

### 6.3 CTCE FSRL 训练与采样验证

先训练：

```bash
python third_party/FSRL/examples/customized/collect_dataset.py \
  --backend safety_gym \
  --num_agents 3 \
  --collect_only False \
  --epoch 2 \
  --step_per_epoch 2000 \
  --training_num 2 \
  --testing_num 1 \
  --device cpu \
  --name ctce_smoke_train \
  --logdir ./third_party/FSRL/logs_local \
  --logger_backend tensorboard \
  --vector_env auto \
  --memory_guard auto
```

再采样并导出 OSRL 数据：

```bash
python third_party/FSRL/examples/customized/collect_dataset.py \
  --backend safety_gym \
  --num_agents 3 \
  --collect_only True \
  --num_episodes 2 \
  --policy_checkpoint ./third_party/FSRL/logs_local/fast-safe-rl/ctce-n3-cost-5-100/ctce_smoke_train/checkpoint/model.pt \
  --output_dir ./data/ctce_fsrl_smoke_remote \
  --dataset_name offline_dataset_ctce.hdf5 \
  --seed 11

python third_party/OSRL/examples/tools/export_ctce_hdf5_to_osrl.py \
  --input-hdf5 ./data/ctce_fsrl_smoke_remote/<run_tag>/offline_dataset_ctce.hdf5 \
  --output-npz ./data/ctce_fsrl_smoke_remote/<run_tag>/offline_dataset_osrl.npz \
  --num-agents 3 \
  --policy-tag fsrl_ctce \
  --checkpoint-id ctce_smoke_train \
  --seed 11
```

通过标准：
- 采样不报维度不匹配错误
- 导出 `offline_dataset_osrl.npz` 与 `offline_dataset_osrl.meta.json`

---

## 7. 长任务运行建议（服务器）

1. 使用 tmux/screen 持久会话运行训练与采样。
2. 训练日志重定向到文件：

```bash
python third_party/FSRL/examples/customized/collect_dataset.py --collect_only False ... 2>&1 | tee train_ctce.log
```

3. 采样日志重定向到文件：

```bash
python third_party/FSRL/examples/customized/collect_dataset.py ... 2>&1 | tee collect_ctce.log
```

4. 建议定时备份：
- third_party/FSRL/logs_local
- data

---

## 8. 常见问题与排查

### 8.1 找不到 safety_gymnasium

原因：
- 未执行 editable 安装或路径错误

检查：

```bash
pip show safety-gymnasium
```

处理：

```bash
cd /workspace/SafeMARL/SafeTransfer/ma_safe_migration
pip install -e ../SRL/safety-gymnasium-main
```

### 8.2 MuJoCo/渲染报错

处理顺序：

1. 确认系统库安装完整（glfw/mesa/osmesa）
2. 设置无头渲染环境变量：MUJOCO_GL=egl
3. 用 rgb_array 验证，先不使用 human 渲染

### 8.3 FSRL checkpoint 维度不匹配

典型原因：
- 训练时和采样时 num_agents 不一致
- ctce/per_agent 模式与 checkpoint 不匹配

建议：

1. 保持训练与采样 num_agents 一致
2. CTCE 训练后优先使用 `third_party/FSRL/examples/customized/collect_dataset.py` 采样

---

## 9. 回滚方案

如果迁移后失败，按以下顺序回退：

1. 切换到已验证提交（git checkout <commit>）
2. 重新创建 Conda 环境
3. 按第 5 节完整重装依赖
4. 从第 6 节重新做验证

---

## 10. 最终验收清单

1. verify_safety_gym_backend.py 可通过
2. CTCE 采样可生成 `offline_dataset_ctce.hdf5`
3. CTCE FSRL 训练可产出 checkpoint
4. OSRL 导出脚本可生成 `offline_dataset_osrl.npz` 与 metadata
5. 日志与数据目录结构清晰、可备份

完成以上五项，即可进入远程大规模训练/采集阶段。

---

## 附录 A：项目此前未用 Git 时，如何从零开始并推送到 GitHub

本附录适用于“整个项目此前没有 Git 管理”的情况。

### A.1 先决定 Git 仓库根目录

建议把仓库根放在 SafeMARL 目录，以便同时管理以下两个关键目录：

1. SafeTransfer/ma_safe_migration
2. SRL/safety-gymnasium-main

示例根目录：

```text
D:\RL-lab\safe RL\SafeMARL
```

### A.2 初始化 Git 仓库（本地）

Windows PowerShell：

```powershell
Set-Location "D:\RL-lab\safe RL\SafeMARL"
git init
git branch -M main
```

### A.3 创建 .gitignore（强烈建议）

在仓库根目录创建 .gitignore，建议至少包含：

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Envs and local configs
.venv/
venv/
.env

# IDE
.vscode/
.idea/

# Training artifacts
**/logs_local/
**/wandb/
**/runs/

# Offline datasets and heavy binaries
**/data/
*.hdf5
*.pkl
*.pt
*.pth
*.ckpt

# OS
.DS_Store
Thumbs.db
```

说明：

1. 默认不提交数据集和大模型文件，仓库会更轻量。
2. 如果你需要版本化 checkpoint/数据，请使用 Git LFS（见 A.9）。

### A.4 首次检查与提交

```powershell
git status
git add .
git status
git commit -m "chore: initialize repository for SafeMARL migration"
```

如果提示缺少身份信息：

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### A.5 在 GitHub 创建远程仓库

在 GitHub 网页端创建一个空仓库（不要勾选 README/.gitignore 初始化），例如：

```text
https://github.com/<your_user>/<your_repo>.git
```

### A.6 关联远程并推送

```powershell
git remote add origin https://github.com/<your_user>/<your_repo>.git
git push -u origin main
```

后续更新只需要：

```powershell
git add .
git commit -m "feat: ..."
git push
```

### A.7 在服务器拉取项目

```bash
cd /workspace
git clone https://github.com/<your_user>/<your_repo>.git
cd <your_repo>
```

拉取后按本指南第 5 节安装环境，第 6 节执行验证。

### A.8 常见 Git 问题（新手高频）

1. push 被拒绝（non-fast-forward）

```powershell
git pull --rebase origin main
git push
```

2. 不小心提交了大文件

```powershell
git rm --cached <big_file>
git commit -m "chore: remove large file from tracking"
git push
```

3. 想看最近提交历史

```powershell
git log --oneline -n 20
```

### A.9 可选：使用 Git LFS 管理模型与数据

如果你确实要把 checkpoint 或部分数据放到 GitHub：

```powershell
git lfs install
git lfs track "*.pt" "*.pth" "*.hdf5" "*.pkl"
git add .gitattributes
git add <files>
git commit -m "chore: track model/data files with Git LFS"
git push
```

注意：

1. GitHub 免费额度下 LFS 有容量和流量限制。
2. 对大规模离线数据，仍建议走对象存储/网盘，不建议长期放 Git。
