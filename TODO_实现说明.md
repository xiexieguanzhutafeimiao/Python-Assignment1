# image_encoder.py 的 TO-DO 实现说明

本文解释 `image_encoder.py` 中 TODO-01 ~ TODO-06 每一段代码**为什么这样写**：形状如何变化、权重为什么这样乘、容易踩的坑在哪里。代码只改了 `image_encoder.py`，其余文件（含 `helpers.py`、权重、参考数据）保持原样。

## 0. 先记住全模型的 5 条约定

这几条决定了几乎所有转置方向，是做对本题的前提（README「Important model conventions」）：

1. **线性层权重存成 `[out, in]`**，所以做 `x @ W.T`（qkv、attention 输出投影、MLP 两层、patch 投影都是）。
2. **唯一的例外是 `visual.proj`**，它已经是 `[in, out] = [768,512]`，直接 `x @ self.projection`，**不能转置**。
3. **patch 是 32×32**，224/32 = 7，得到 7×7 = 49 个 patch；patch 投影**没有 bias**。
4. **每个 block 有两次残差相加**，没有 LayerScale。
5. 全程 `float32`；LayerNorm 的 eps=1e-5、GELU 用 `helpers.py` 提供的精确 erf 版本、softmax 用提供的 `attention_weights`。这些不要自己重写，否则数值会漂移。

整体数据流：

```
[10,3,224,224]
  → patchify        [10,49,3072]
  → patch 投影       [10,49,768]
  → CLS + 位置       [10,50,768]
  → pre_norm + 12 blocks（每块残差两次）  [10,50,768]
  → 取 CLS → post_norm → visual.proj → L2 归一化  [10,512]
```

---

## TODO-01 `patchify`：把图像切成 49 行 patch

**契约**：`[B,3,224,224] → [B,49,3072]`，patch 先左后右、先上后下，patch 内按 C,H,W 展平。

**为什么这样写**：

```python
B, C, H, W = images.shape
patch = 32
rows, cols = H // patch, W // patch      # 7, 7
grid = images.reshape(B, C, rows, patch, cols, patch)
grid = grid.transpose(0, 2, 4, 1, 3, 5)
return grid.reshape(B, rows * cols, C * patch * patch)
```

- `reshape(B, C, 7, 32, 7, 32)` 把 H、W 各拆成「第几行/列 patch」和「patch 内偏移」两组轴，轴含义是 `(B, C, i, ph, j, pw)`。此时原始下标关系是 `h = i*32 + ph, w = j*32 + pw`。
- `transpose(0,2,4,1,3,5)` 重排成 `(B, i, j, C, ph, pw)`：把 `i,j` 放到 patch 行之前，`C,ph,pw` 放到最后。因为 `j` 比 `i` 靠后，展平时 `j` 变化最快，正好是「一行内从左到右，再换下一行」。
- 最后 `reshape(..., 49, 3072)` 把 `C*ph*pw = 3*32*32 = 3072` 展成一维，顺序正是 C→H→W。

**为什么不用循环切 7×7**：纯 reshape/transpose 不复制数据、顺序可证明，而且天然支持任意 B。

**坑**：如果只做 `images.reshape(B,3,49,1024)` 会在每行内按 W 优先切，得到的是列优先的 patch，顺序全错；`patches` 阶段参考值是完全 0 误差的，一旦顺序错 max_abs 会是很大的数。

---

## TODO-02 `split_heads`：拆成 12 个头

**契约**：`[B,N,D] → [B,H,N,D/H]`，本模型 `D=768, H=12, head_dim=64`。

**为什么这样写**：

```python
B, N, D = x.shape
grouped = x.reshape(B, N, self.heads, self.head_dim)
return grouped.transpose(0, 2, 1, 3)
```

- 先把每个 token 的 768 个特征切成 12 段连续的 64 维：`[B,N,H,64]`。这是 OpenCLIP/PyTorch 多头注意力的布局（特征维按头连续切分，不是隔一个取一个）。
- 再把头轴换到 token 轴前面得到 `[B,H,N,64]`，这样 `q @ k^T` 可以在每个 `(B,H)` 分组里独立做矩阵乘法，NumPy 会自动广播批处理维度。

**坑**：`transpose(0,2,1,3)` 不能写成 `reshape(B,H,N,64)`——后者会把特征顺序重排错（相当于做了一次错误的转置）。token 顺序 N 在转置前后保持不变。

---

## TODO-03 多头注意力：打分、加权、合并、投影

### 03a 打分

```python
scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(np.float32(self.head_dim))
```

- `q,k` 都是 `[B,H,N,64]`。注意力分数是「每个 query 对每个 key」的内积，所以要交换 **k 的最后两轴** 变成 `[B,H,64,N]`，得到 `[B,H,N,N]`。
- README 特别提醒：k 是 4D，**不能写 `k.T`**。`k.T` 会把四个轴全部反转成 `[64,N,H,B]`，批和头分组被彻底打乱；正确写法是显式 `transpose(0,1,3,2)`，只换最后两轴。
- 除以 `sqrt(head_dim)` 是缩放点积注意力的标准做法，防止内积过大把 softmax 推到饱和区。用 `np.float32(64.0)` 开方可以保持 float32。课堂示例 `transformer_steps.py` 里用的也是 `(q @ k.T)/sqrt(4)`，只是它是 2D 的教学版。

### 03b 加权求和并合并多头

```python
context = coefficients @ v                       # [B,H,N,64]
B, H, N, d = context.shape
merged = context.transpose(0, 2, 1, 3).reshape(B, N, self.width)
```

- `coefficients` 是 `[B,H,N,N]`（每行和为 1），`v` 是 `[B,H,N,64]`，矩阵乘得到每个 token 的加权 value，即 `[B,H,N,64]`。
- `split_heads` 做的是 `[B,N,D] → [B,N,H,64] → [B,H,N,64]`，这里做逆操作：`[B,H,N,64] → [B,N,H,64] → [B,N,768]`。必须先把 N 换回 H 前面再 reshape，否则头会被错误交错。

### 03c 输出投影

```python
return merged @ self.out_weight.T + self.out_bias
```

`out_proj.weight` 存成 `[out,in]=[768,768]`，所以转置后乘；bias `[768]` 会自动广播到每个 token。输出仍是 `[B,N,768]`。

**因果开关**：`if causal: scores = causal_mask(scores)` 是题目给好的，留给后续文本塔用；图像 tower 用 `causal=False`，不需要改。

---

## TODO-04 TransformerBlock：两次残差

```python
update = self.attention(self.norm1(x), causal=causal)   # 已给
residual = x + update                                    # 04a

u = self.norm2(residual)                                 # 已给
hidden = u @ self.fc1_weight.T + self.fc1_bias            # 04b
activated = self.gelu(hidden)
mlp_update = activated @ self.fc2_weight.T + self.fc2_bias
return residual + mlp_update
```

**为什么这样写**：

- 这是 Pre-LN Transformer：注意力吃的是 `norm1(x)`，但残差加到**原始 x** 上，不是加到归一化后的值上。写成 `residual = self.norm1(x) + update` 是典型错误，形状一样但数值错，`block0` 阶段会立刻报错。
- MLP 是「升维 4 倍 → GELU → 降维」：`fc1_weight [3072,768]`、`fc2_weight [768,3072]`，都按 `[out,in]` 存储，所以都要 `.T`。中间 `hidden [B,N,3072]`，输出 `mlp_update [B,N,768]`。
- 第二次残差加到 `residual` 上，得到块的最终输出 `[B,N,768]`。块内维度自始至终不变。

**坑**：GELU 必须用 `helpers.GELU`（精确 erf 版本），不要用近似版 `0.5*x*(1+tanh(...))`，否则 `block0` 开始误差会超限。

---

## TODO-05 `embed`：patch 投影 + CLS + 位置编码

```python
patch_tokens = patches @ self.patch_weight.T          # [B,49,768]
cls = np.tile(self.cls, (images.shape[0], 1, 1))       # 已给，[B,1,768]
tokens = np.concatenate([cls, patch_tokens], axis=1)   # [B,50,768]
return tokens + self.position                          # [50,768] 广播到 B
```

**为什么这样写**：

- `conv1.weight` 原始形状是 `[768,3,32,32]`，构造时已被 reshape 成 `[768,3072]=[out,in]`，所以 `patches @ weight.T` 把 3072 维 patch 投影到 768 维。**没有 bias**（卷积层在这里没有对应 bias 张量，题目也明确说无 bias）。
- CLS 是学出来的一个 768 维行，拼在 token 轴最前面（axis=1），位置编码 `[50,768]` 是按位置加的：第 0 行对应 CLS，第 1~49 行对应 49 个 patch，且顺序与 patchify 保持一致。
- `self.position` 是 2D，`tokens` 是 3D，NumPy 广播自动把它复制 B 份，不需要手动 tile。

**坑**：忘记加位置编码或把 CLS 追加到最后，都会在 `tokens` 阶段失败。参考的 `input_tokens` 是**未过 pre_norm** 的，check.py 里 `tokens` 阶段直接调 `encoder.embed(images)`，别在这里偷偷加 norm。

---

## TODO-06 `__call__`：取 CLS → 后归一化 → 投影 → 单位化

```python
tokens = self.encode_tokens(images)                    # [B,50,768]，已给
cls_token = tokens[:, 0, :]                            # [B,768]
cls_normed = self.post_norm(cls_token)
output = cls_normed @ self.projection                  # [B,512]
return unit_vector(output)                             # 已给
```

**为什么这样写**：

- ViT 用第一个 token（CLS）的表示作为整图表示，所以取 `tokens[:, 0, :]`。
- `post_norm` 是最后一层 LayerNorm，作用对象只是 CLS 那一行；`LayerNorm` 对 `[B,768]` 也能工作（mean/var 都在最后一轴上算，权重 `[768]` 广播），不需要额外保留 token 维。
- `visual.proj` 已经是 `[in,out]=[768,512]`，这是全题唯一**不转置**的线性权重。若错误地写 `.T`，`[B,768] @ [512,768]` 会直接形状报错。
- 最后用给定的 `unit_vector` 做 L2 归一化，每行长度变成 1。`run.py` 保存的 `output_tokens` 就是这个 `[B,512]` 的 float32 数组。

**坑**：不要把 `encode_tokens` 的全部 50 行送去投影；`last_tokens` 是内部序列，最终输出只要 CLS 一行。参考数据只是调试目标，绝不能当输入或直接返回。

---

## 数值校验结果（本机实测）

命令：`python check.py --stage all`

| 阶段 | 形状 | max_abs | mean_abs | 容差 | 结果 |
| --- | --- | --- | --- | --- | --- |
| assets | — | — | — | — | PASS（152 个张量 + 10 对图/参考） |
| preprocess | (10,3,224,224) | 0.000e+00 | 0.000e+00 | 2e-6 | PASS |
| patches | (2,49,3072) | 0.000e+00 | 0.000e+00 | 2e-4 | PASS |
| tokens | (2,50,768) | 4.005e-05 | 1.289e-06 | 2e-4 | PASS |
| block0 | (2,50,768) | 7.063e-06 | 3.250e-07 | 2e-4 | PASS |
| last_tokens | (2,50,768) | 4.768e-05 | 2.696e-06 | 2e-4 | PASS |
| output | (10,512) | 5.439e-07 | 8.311e-08 | 2e-5 | PASS |
| batch | (3,512) | 4.619e-07 | 7.632e-08 | 2e-5 | PASS（乱序 + 重复图片） |

端到端：`python run.py --output output_tokens.npz` → 保存 `(10,512)` float32，每行 L2 范数均为 1.0（在 2e-4 容差内）。

> `patches` 的 0 误差说明 patch 顺序和展平方式完全正确；后面几阶段出现 1e-5 量级差异，是 NumPy 与独立 Torch FP32 实现在归约/矩阵乘累加顺序上的正常差异，远低于容差。

---

## 形状变化速查 & 两个典型 bug

**一个形状变化**：`[B,3,224,224] → [B,49,3072] → [B,49,768] → [B,50,768] →（12 块保持 `[B,50,768]`）→ [B,768] → [B,512]`。注意力内部还有 `[B,N,768] → [B,H,N,64] → [B,H,N,N] → [B,H,N,64] → [B,N,768]`。

**一个易犯 bug 及修法**：把残差加到 norm 之后的值上，即写 `residual = self.norm1(x) + update`。形状完全正确、不会报错，但 `block0` 的 max_abs 会远超 2e-4。修法是加回原始输入 `residual = x + update`，与 `transformer_steps.py` 中 `residual = x + update` 一致。另一个同级别的坑是对 4D 的 k 写 `k.T`；形状直接对不上，修法是 `k.transpose(0,1,3,2)`。

---

## 怎么运行

```bash
cd C:\Users\as\Downloads\project1-numpy-student
python -m pip install -r requirements.txt   # 如已装 numpy / Pillow 可跳过
python check.py --stage all
python run.py --output output_tokens.npz
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

注意：`weights/image_encoder.npz` 与 `assets/reference_outputs.npz` 由课程附件提供，按 README 要求未纳入仓库；本地运行前需把附件放回原路径。

在最终项目中复用时，把本文件和题目提供的 `helpers.py` 一起放到 `clip.py`、`text_encoder.py` 旁边；`ImageEncoder(weights)`、`LayerNorm`、`TransformerBlock(weights,prefix,heads)` 的接口保持不变。

---

## 贡献与所用辅助（contributions and assistance used）

> 以下为草稿，提交前请按实际情况核对、修改。

**完成的工作**

- 阅读 `README.md`、`PPT-CODE-MAP.txt` 与课堂示例 `transformer_steps.py`，确定 6 组 TODO 的实现方案。
- 在 `image_encoder.py` 中补全 TODO-01 至 TODO-06：patchify、多头拆分、注意力打分/加权/合并/投影、TransformerBlock 的两次残差与 MLP、CLS 与位置编码嵌入、CLS 输出投影与归一化。只改动该文件，未修改题目提供的权重、参考数据和其他脚本。
- 逐阶段运行 `python check.py --stage all` 与端到端 `run.py`，记录误差（见上表），并据此定位形状与数值问题。
- 撰写本说明文档，解释每处实现的原因、形状变化、一个 bug 的成因与修法。

**所用辅助**

- **WorkBuddy AI（腾讯混元 Hy）**：协助解读 TODO 契约、生成 NumPy 实现草稿与本文档、整理校验结果；所有代码与文字已经过人工核对。
- **课程提供的材料**：`README.md`、`transformer_steps.py`（课堂算术示例）、`helpers.py`（LayerNorm / 精确 GELU / softmax / unit_vector）、`check.py`（数值校验）、`run.py`、`config.json`；权重与参考输出仅用于校验，未作为编码器输入，也未重新提交。
- **运行环境**：Python 3.10+，依赖 NumPy 与 Pillow（见 `requirements.txt`），CPU 即可，无需 GPU、PyTorch 或 OpenCLIP 包。

**未使用**：网络上的第三方 OpenCLIP 完整实现、参考输出的缓存结果、任何参数训练或微调。
