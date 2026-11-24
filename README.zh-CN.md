# 项目一：复现最终项目的图像编码器

给定图像和学习到的模型参数 **theta**，用 NumPy 实现 OpenCLIP ViT-B-32 的**前向计算**。你的输出必须在数值上与提供的参考结果一致。这个完成后的编码器将作为最终项目中 `szu_python_openclip` 的图像塔（image tower）。本项目不涉及模型训练。

## 输入与输出

- `ImageEncoder` 的输入：完成给定预处理后的 float32 数组 **[B,3,224,224]**。
- 经过 12 个 block 后的内部图像 token 数组：**[B,50,768]**（CLS 加 49 个 patch token）。
- 最终**输出 token**：float32 **[B,512]**，每张图像一行归一化后的数值。
  这正是最终项目中 `ImageEncoder.__call__` 的输出。

本作业中"输出 token"（output token）一词指最终的那一行 512 维数值。
`last_tokens` 则单独指完整的 50 行内部序列。


## 附件中的文件

解压附件并打开该目录。本 README 中的命令都在此目录下执行；Python 会相对于脚本解析模型/图像文件。
不需要课程目录的绝对路径。

- `practice/project1_exercises.pdf`：12 道 NumPy 练习题，不含答案。
- `lecture-demos/`：NumPy 课堂示例，以及注明来源的本地猫图片。
- `transformer_steps.py`：随附的微型 block 演算程序。
- `PPT-CODE-MAP.txt`：当前 76 页幻灯片的对照映射。
- `SOURCE-MANIFEST.json`：来源说明、SHA-256 以及有意做出的打包改动。

本作业需要 NumPy 和 Pillow。imageio 课堂示例还需要 imageio；附件的 requirements 中已包含这三者。

## 快速开始

使用 Python 3.9 或更高版本。不需要 GPU、PyTorch、OpenCLIP 包或网络。

```bash
python -m pip install -r requirements.txt
python check.py --stage assets
python check.py --stage preprocess
python check.py --stage patches
```

前两项检查在未改动的初始代码中即可通过。后续检查会报告第一个尚未填写的 TODO，而不会假装初始代码已经完成。填写完六个练习后：

```bash
python check.py --stage all
python run.py --output output_tokens.npz
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

可在任意目录下用完整路径运行脚本。`run.py` 按输入顺序保存图像 ID 与输出 token。

## 按照课堂上的计算过程

```bash
python transformer_steps.py
```

这个随附程序演示了第 46-66 页讲解的运算过程：
行均值、中心化、方差、Norm 的 scale/bias、q/k/v、分数、系数、
加权后的 value、注意力投影、两个残差以及 MLP。课堂上使用完整的图像数组形状；该程序则打印一个微型计算。
示例使用**三行 token、四个特征、一个 head 和六个 MLP 中间值**。其合成参数保证所有数值都可见。它不是训练得到的 checkpoint。显示的数值保留两位小数；计算过程保持完整的 float32 精度。

请把运算迁移到 `image_encoder.py` 中真实的形状和 TODO 上。
Norm 2 的结果命名为 `u`，与评测用的编码器一致。
微型示例使用二维表，所以 `k.T` 会交换它的两个轴。真实的 k 是四维的：
只交换最后两个轴，并保留图像/head 分组。head 的组织方式需你自行学习和实现。该程序没有涉及这一部分。
真实模型仍使用所提供的精确图像参数与参考输出。六个受评练习组仍全部未完成。

第 67-70 页把最终输出拆分为 CLS 选取、CLS 后置 Norm、
输出投影和单位长度缩放。第 71-73 页汇总了编码器约定、六个实现任务、课堂命令和数值检查。
第 74-76 页说明最终项目中的复用方式和文本扩展。
当前课件共 76 页。

## 需要实现的内容

**只修改 image_encoder.py**。一个练习可能包含多个标记的步骤。

| 练习 | 方法 | 要求的结果 |
| --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | [B,3,224,224] 变为 [B,49,3072]，保持 patch 顺序 |
| TODO-02 | `MultiHeadAttention.split_heads` | [B,N,D] 变为 [B,H,N,D/H] |
| TODO-03 | `MultiHeadAttention.__call__` | 分数、加权 value、head 合并与输出投影 |
| TODO-04 | `TransformerBlock.__call__` | 注意力/MLP 计算以及两个残差相加 |
| TODO-05 | `ImageEncoder.embed` | patch 投影、CLS 拼接与学习到的位置编码 |
| TODO-06 | `ImageEncoder.__call__` | 选取 CLS、后置 Norm、投影与最终输出 |

使用 NumPy 的数组运算、索引、reshape/transpose、矩阵乘法、
归约和广播。自行学习多头组织方式。
激活函数和注意力系数的转换已提供。

## 已提供的内容

- `preprocess_image.py`：精确的 resize、中心裁剪和像素归一化。
- `helpers.py`：LayerNorm、精确 GELU、注意力系数和向量缩放。
- `runtime.py`、`run.py`、`check.py`：加载、命令行和参考检查。
- `weights/image_encoder.npz`：**仅包含最终模型的 152 个视觉张量**。
  原始张量名称、形状和 float32 数值均保留。不含文本权重。
- `config.json` 和 `weights/manifest.json`：精确的架构与指纹。
- `assets/images/`、`images.json`：十张本地图像，具有稳定的 ID 和来源说明。
- `assets/reference_outputs.npz`：按 ID 顺序独立生成的 FP32 各阶段/输出数组。

配置文件描述的就是这个固定 checkpoint。改动配置中的某个数字并不会把它变成另一个模型。请遵循所提供的权重和接口。

## 数值检查

`check.py` 会与独立的 Torch FP32 前向结果以及最终项目的教师参考结果进行比较。学生只能使用 NumPy。所谓"一致"指形状相同、
float32 数值有限，且 `abs(actual-reference) <= atol + rtol*abs(reference)`。
输出 token 的容差为 **atol=2e-5，rtol=2e-4**。内部 token 的容差为
**atol=2e-4，rtol=2e-4**。预处理使用 atol=2e-6。检查会打印最大绝对误差和平均绝对误差。`batch` 检查会打乱顺序并加入重复图像。

按实现顺序推荐使用的命令：

```bash
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

在不改动所提供文件的情况下查看目标数据：

```python
import numpy as np
with np.load("assets/reference_outputs.npz", allow_pickle=False) as ref:
    print(ref["ids"])
    print(ref["last_tokens"].shape)    # (10,50,768)
    print(ref["output_tokens"].shape)  # (10,512)
```

`PPT-CODE-MAP.txt` 把课件片段对应到这些具体的初始代码行。

阅读中间数组，以定位**第一个**出现偏差的阶段。它们只是调试用的参照，绝不能作为编码器的输入。你的实现还必须能处理其他图像和 batch 大小。不要直接返回缓存好的参考行。

## 重要的模型约定

1. patch 为 32x32，因此 224/32 得到 7x7 个 patch。每个 patch 按 C,H,W 展平。
2. patch 投影**没有 bias**。它使用 `patch_weight.T`。
3. 注意力/MLP 的线性权重按 **[out,in]** 存储，因此使用 `weight.T`。
4. `visual.proj` 已经是 **[in,out] = [768,512]**。不要对它转置。
5. 使用所提供的精确 GELU 和 Norm epsilon。不进行任何参数更新。
6. 图像 block 有两个残差相加。它没有 LayerScale 参数。

## 提交

提交你完成的 `image_encoder.py`、一份简短说明如何运行的 README，
以及你的报告。报告中需包含各阶段的数值误差，并解释一处形状变化
以及你修复的一个 bug。还要包含你的贡献和所借助的帮助。不要重新提交庞大的参数文件或参考数据。评分遵循课程的"代码 30 / 报告 70"规则以及已公布的面试/截止日期政策。

## 在最终项目中复用

把完成的 `image_encoder.py` **以及所提供的 `helpers.py`** 复制到另行分发的最终项目附件中 `clip.py` 和 `text_encoder.py` 旁边。
路径相对于该解压后的附件。`ImageEncoder(weights)`、
`LayerNorm` 和 `TransformerBlock(weights,prefix,heads)` 保持所需的 API 不变。
所提供的可选 causal 开关保持对后续文本塔的兼容性。
最终项目仍会提供完整的图像/文本 checkpoint 和 tokenizer。
不要用这些仅含图像的权重替代完整的多模态 checkpoint。
