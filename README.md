# autodiff — 最小计算图自动微分引擎

基于 numpy 的反向传播引擎：`Tensor` 记录运算构成计算图，`backward()` 沿图
反向累计梯度，`optim` 提供 SGD/Adam 完成参数更新。可以端到端训练小型
模型（见 `examples/train_tiny.py`），不是符号求导打印器。

## 快速开始

```python
import autodiff as ad
from autodiff import nn, optim, tensor
from autodiff.nn import softmax_ce_loss

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for _ in range(100):
    opt.zero_grad()
    loss = softmax_ce_loss(model(tensor(X)), labels)
    loss.backward()
    opt.step()
```

运行示例与测试：

```bash
python examples/train_tiny.py
python -m pytest -q
```

## 计算图生命周期

图的存活完全依赖普通 Python 引用：每个可微运算的输出 `Tensor` 持有其
`_grad_fn` 节点，节点再引用父张量。**没有全局 tape / 全局图注册表**；
用户只要持有输出张量，整条图就可达，释放输出后图自然被 GC 回收。

- `backward(retain_graph=True)`（默认）：图保留，可重复反传，梯度按
  PyTorch 语义**累计**到 `.grad`；`zero_grad()` 清空。
- `backward(retain_graph=False)`：反传后释放节点引用，再次反传同一图
  会抛出 `RuntimeError`。
- `no_grad()` 上下文：期间构造的运算不建图（推理、优化器更新用）。

## 广播与形状规则

广播遵循 NumPy 规则：形状右对齐，各维度相等或其中一方为 1（缺失视为 1），
否则抛 `ValueError`；`autodiff.broadcast_shapes` 可独立校验。

反向时，梯度对**新增的前导维**和**被拉伸的 size-1 维**求和，再 reshape
回输入形状，保证梯度形状与参数形状一致（引擎内部会校验，不一致即报错）。

各算子形状约定：

| 算子 | 形状规则 |
| --- | --- |
| `add/sub/mul/div/pow` | 任意可广播形状；`pow` 支持标量或张量指数 |
| `matmul` | `(m,k)@(k,n)`、`(m,k)@(k,)`、`(k,)@(k,n)`、`(k,)@(k,)` |
| `sum/mean` | `axis=None|int|tuple`，`keepdims` 控制是否保留维 |
| `reshape` | 元素数不变；`transpose`/`T` 支持任意轴置换 |
| `softmax` | 沿最后一维，数值稳定（先减最大值） |
| `exp/log/relu/sigmoid/tanh` | 逐元素，形状不变 |

## 原地修改与旧图失效

每个张量带版本计数 `_version`：

- `add_/sub_/mul_` 或 `t.data = ...` 会递增版本；之后对**引用该张量的旧图**
  调用 `backward()` 会明确抛出 `RuntimeError("modified in place")`，
  而不是静默给出错误梯度。
- `t.data` 底层 numpy 数组是只读的，直接 `t.data[...] = x` 立即抛
  `ValueError`，防止绕过版本机制污染图。
- 优化器在 `no_grad` 下通过受控的 `add_` 更新参数，因此上一步的图在
  更新后自动失效——训练循环必须每步重新前向（与 PyTorch 一致）。

## 中间状态重算（gradient checkpointing）

长序列前向不必保存全部中间值：`checkpoint(fn, *args)` 只保留段的输入/输出，
反传时重放 `fn` 重建局部图；`checkpoint_sequential(layers, x, segments)` 把层序列
切成若干重算区间。短序列直接保留全部状态即可，两者损失与梯度完全一致
（`tests/test_checkpoint.py` 对照验证）。

- **无副作用重放**：随机算子（`randn`/`rand`/`nn.Dropout`，用 `manual_seed` 播种）
  走全局 RNG；段边界保存 RNG 状态，重算前恢复，因此重放取样与本次前向**完全相同**，
  且不会第二次推进生成器——重算没有外部可见副作用。
- **梯度累计**：共享参数、多分支汇合的梯度与全保留图一致；闭包捕获的参数在
  重算时照常累计 `.grad`。
- **失效检测**：checkpoint 记录前向时所有存活张量的版本，前向后任何被引用的
  参数被原地修改（含优化器更新、`load_state_dict`），`backward()` 仍抛
  `RuntimeError`，不会混用上一轮计算图。
- `fn` 只应依赖传入的张量参数与闭包参数并返回单个 `Tensor`；只有参数会被
  做版本检查。参数全部不需要梯度而闭包参数需要时，传 `requires_grad=True`。

### 重算中的外部事件（不能靠“调用者保证纯函数”）

随机算子通过保存/恢复 RNG 状态保证重放等价；任意**外部效果**（网络请求、计数、
写文件、扣费……）没有这层保证，直接写在 `fn` 里会在前向一次、反向重算再发生一次。
因此外部效果必须显式登记：

```python
from autodiff import EventLog, record_external_event, use_event_log, checkpoint

log = EventLog("train-events.jsonl")

def segment(t):
    # 前向：真正调用 fetch_minibatch()；反向重算/重复 backward：直接返回已记录结果，
    # 回调绝不会被再次调用。
    batch = record_external_event("step-12-batch", fetch_minibatch)
    return model(t + as_tensor(batch))

with use_event_log(log):
    loss = checkpoint(segment, x)
loss.backward()          # 不产生第二次外部效果；日志追加内部 replay 记录
```

- **用户可见事件 vs 内部重算**：事件日志（`autodiff.events.EventLog`）里外部效果
  记为 `event`（`submitted` → `committed` 两条），重算只追加 `replay` 条目，可用
  `log.events()` / `log.replays()` 分别查看。
- **无法支持时提前拒绝**：重算遇到未登记的事件、没有事件日志、或事件仍停留在
  `submitted` 时，抛 `ReplayExternalEventError` / `PendingExternalEventError`，
  发生在任何回调执行**之前**，不会重复产生效果。
- **幂等标识**：`event_id` 必须在前向、重算、重启重放之间稳定（如
  `"step-12-batch"`）；重复提交同一已确认 id 直接返回原结果。

### 保存重开与失败回退

`EventLog(path)` 使用 append-only、每行 fsync 的 JSONL 持久化：

- 重新打开日志后重跑某一步，已 `committed` 的事件**不会再次提交**，返回原结果；
  配合 `Module.state_dict()`/`load_state_dict()` 保存参数即可断点续训（见
  `examples/resumable_training.py`）。
- 效果产生**前**先落盘 `submitted`。若回调失败或进程崩溃，重新打开日志会抛
  `PendingExternalEventError`：该效果“可能已经发生”，系统不会自动重试或回滚。
  人工核对外部系统后调用 `log.resolve_pending(event_id, result=...)` 显式确认——
  日志只追加、从不删除，因此失败回退无法抹掉真实发生过的效果。

## 保存与恢复

`Module.state_dict()` / `load_state_dict()` 以 numpy 数组快照/恢复参数；恢复通过
`.data` 赋值递增版本号，加载前构建的旧图自动失效。

## 正确性验证

`tests/` 使用**中心差分数值梯度**独立核对所有核心算子与组合图
（`tests/grad_check.py`），并覆盖：

- 共享中间值与分支汇合的梯度累计；
- 同一变量在表达式中重复使用；
- 同一图连续两次 `backward` 的累计语义；
- 原地修改使旧图失效并明确报错；
- 广播梯度、各 `axis/keepdims` 归约；
- 逻辑回归（SGD/Adam）与 MLP 多分类的端到端训练收敛。

## 目录结构

```
autodiff/
  core.py   # Tensor / Function / backward / no_grad / 版本失效
  ops.py    # 可微算子与广播反向
  nn.py     # Module / Linear / Sequential / 损失函数
  optim.py  # SGD(momentum) / Adam
  checkpoint.py  # 中间状态重算（gradient checkpointing）
  events.py      # 外部事件登记 + 持久化训练事件日志（EventLog）
tests/      # 数值梯度核对 + 行为测试
examples/   # 端到端训练示例
```
