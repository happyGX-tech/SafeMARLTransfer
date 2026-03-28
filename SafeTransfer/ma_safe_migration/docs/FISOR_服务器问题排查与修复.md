# FISOR 服务器问题排查与修复

本文记录本次从本地迁移到服务器后遇到的典型问题、根因和可直接执行的修复命令。

## 1. FISOR 部分目录在 GitHub 看起来“没上传”

### 现象

服务器目录里看不到本地存在的 `data`、`results`、`wandb` 等目录内容。

### 根因

这些目录被 `.gitignore` 规则忽略，属于运行产物目录，不会被 Git 跟踪。

### 说明

这通常是预期行为，不影响代码运行。

## 2. `ModuleNotFoundError: No module named jaxrl5.data`

### 根因

主仓库根目录 `.gitignore` 中的规则 `**/data/` 误伤了源码目录，导致以下源码未被提交：

- `SafeTransfer/ma_safe_migration/third_party/FISOR/jaxrl5/data/dataset.py`
- `SafeTransfer/ma_safe_migration/third_party/FISOR/jaxrl5/data/dsrl_datasets.py`

### 已采取修复

在根目录 `.gitignore` 增加白名单，允许该目录下 `.py` 文件被跟踪：

```gitignore
!SafeTransfer/ma_safe_migration/third_party/FISOR/jaxrl5/data/
!SafeTransfer/ma_safe_migration/third_party/FISOR/jaxrl5/data/*.py
```

### 提交与推送

```bash
cd "d:\RL-lab\safe RL\SafeMARL"
git add .gitignore \
  SafeTransfer/ma_safe_migration/third_party/FISOR/jaxrl5/data/dataset.py \
  SafeTransfer/ma_safe_migration/third_party/FISOR/jaxrl5/data/dsrl_datasets.py
git commit -m "Fix FISOR upload: track jaxrl5 data source modules"
git push origin main
```

服务器更新：

```bash
cd /home/work3/SafeMARL/SafeMARLTransfer
git pull
```

## 3. `No module named ml_collections` / `No module named wandb`

### 根因

服务器运行时使用的 Python 环境未完整安装 FISOR 依赖。

### 修复命令（服务器）

```bash
conda activate FISOR
cd /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install ml-collections wandb
```

### 国内镜像安装（推荐）

```bash
conda activate FISOR
cd /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/FISOR

# 临时使用清华镜像（只影响当前命令）
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip setuptools wheel
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --prefer-binary --no-cache-dir
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ml-collections wandb
```

```bash
# 或者永久设置 pip 镜像（当前用户）
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

### 快速验证

```bash
python - <<'PY'
import ml_collections, wandb
print('ok')
PY
```

## 4. Gymnasium 重复注册 warning 太多

### 现象

出现类似 warning：
`WARN: Overriding environment ... already in registry`

### 处理

在训练入口加入定向 warning 过滤（仅屏蔽该类提示，不改逻辑）：

- 文件：`SafeTransfer/ma_safe_migration/third_party/FISOR/launcher/examples/train_offline.py`

关键代码：

```python
import warnings

warnings.filterwarnings(
    'ignore',
  message=r'.*Overriding environment .* already in registry.*',
    category=UserWarning,
)
```

## 5. JAX 未使用 GPU（回退 CPU）

### 现象

日志提示：

- `An NVIDIA GPU may be present ... but a CUDA-enabled jaxlib is not installed. Falling back to cpu.`

### 根因

当前环境装的是 CPU 版 `jaxlib`。

### 修复（服务器）

```bash
conda activate FISOR
python -m pip uninstall -y jax jaxlib
python -m pip install -U "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### JAX + 国内镜像建议

`jax[cuda12_pip]` 的 CUDA 轮子主要来自 Google 的 JAX 源，不能完全由国内 PyPI 镜像替代。建议先用国内镜像安装其余依赖，再单独安装 JAX：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U numpy scipy absl-py flax optax tqdm wandb ml-collections
python -m pip install -U "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

若上条不成功：

```bash
python -m pip install -U "jax[cuda12]"
```

### 验证 GPU

```bash
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
import jax
print('jax version:', jax.__version__)
print('backend:', jax.default_backend())
print('devices:', jax.devices())
PY
```

期望看到 `backend: gpu`。

## 6. 使用 `cuda:1` 训练（其余参数不变）

```bash
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python launcher/examples/train_offline.py \
  --config configs/train_config.py:fisor \
  --custom_env_name sg_ant_goal_n4 \
  --num_agents 4 \
  --dataset_path /home/work3/SafeMARL/SafeMARLTransfer/SafeTransfer/ma_safe_migration/third_party/DSRL/processed/ctce_n4-120-428.hdf5 \
  --ratio 0.3 \
  --project ctce_debug \
  --experiment_name ctce_n4_fisor
```

## 7. Linux 与 PowerShell 环境变量写法区别

- PowerShell：`$env:XLA_PYTHON_CLIENT_PREALLOCATE="false"`
- Bash：`export XLA_PYTHON_CLIENT_PREALLOCATE=false`

如果在 Linux Bash 里写 PowerShell 语法，会报 `command not found`。

---

如需，我可以在后续再补一节“最小可复现自检脚本（一次检查 Python 环境、JAX GPU、关键依赖、数据路径）”。
