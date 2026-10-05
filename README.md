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

- **内部随机事件**：`randn`/`rand`/`nn.Dropout`（用 `manual_seed` 播种）在前向时把抽样
  记录为区间内部事件；反向重算只读取记录值，因此掩码完全相同，且全局 RNG 只推进一次。
- **用户可见外部事件**：请求外部服务、写队列、扣费、读取外部可变数据等操作必须通过
  `external_event(event_id, apply, replay)` 执行。`apply` 只在用户可见前向执行一次并返回
  JSON 可保存的结果；`replay(payload)` 必须是纯函数，在反向重算或进程重开时重建张量。
  无法提供纯 `replay` 的操作会在产生效果前抛出 `UnsupportedExternalEventError`。
- **梯度累计**：共享参数、多分支汇合的梯度与全保留图一致；闭包捕获的参数在
  重算时照常累计 `.grad`。
- **失效检测**：checkpoint 记录前向时所有存活张量的版本，前向后任何被引用的
  参数被原地修改（含优化器更新、`load_state_dict`），`backward()` 仍抛
  `RuntimeError`，不会混用上一轮计算图。
- `fn` 只应依赖传入的张量参数与闭包参数并返回单个 `Tensor`；只有参数会被
  做版本检查。参数全部不需要梯度而闭包参数需要时，传 `requires_grad=True`。

## 保存与恢复

`Module.state_dict()` / `load_state_dict()` 以 numpy 数组快照/恢复参数；恢复通过
`.data` 赋值递增版本号，加载前构建的旧图自动失效。

训练中的外部事件使用 `TrainingEventLog(path)` 持久化为追加式 JSONL：

```python
log = ad.TrainingEventLog("checkpoints/events.jsonl")
with log.use():
    batch = ad.external_event(
        f"batch/{step}",
        lambda: api.fetch_next_batch(),  # 用户可见，只执行一次
        lambda payload: ad.tensor(payload),  # 重算/重开时的纯重建
    )
    loss = ad.checkpoint(model_loss, batch)
    loss.backward()
```

重新打开同一个日志后，已确认的事件 ID 不会再次执行 `apply`，而是从记录恢复。日志只追加、
不重写；训练失败或优化器回滚不能删除已经实际发生的事件记录。相同 ID 若使用不同 payload
重新提交，会抛出 `EventExistsError`。

### 单项训练步骤的外部效果状态机

一组相互依赖的外部效果必须先在同一个步骤中声明，并按声明顺序执行：

```python
try:
    with log.begin_step(step, ["reserve", "submit", "publish"]) as tx:
        reservation = tx.run_effect("reserve", api.reserve)
        submission = tx.run_effect("submit", lambda: api.submit(reservation))
        loss.backward()
        tx.complete(model, optimizer=opt, loss=loss)
except Exception as exc:
    tx.fail(exc)
```

每个声明效果都会落入三种可审计状态之一：

- `occurred`：已经确认发生，并保存可用于重建的 JSON payload；
- `unconfirmed`：请求可能发出但结果未知，后续效果保持 `not_executed`；
- `not_executed`：因依赖项未确认或步骤中断而没有执行。

进程重开后可补充核实结论，再从明确位置继续。重复核实同一个“已发生”结论只会追加核实记录，
不会再次执行效果；只有核实为“未发生”才允许重新申请。`complete()` 只在同一轮训练的全部
外部效果均已确认、所有必需梯度已经计算后才可调用。完成记录绑定效果 payload 摘要、优化前
模型与梯度、优化后模型、loss 和 RNG 状态；因此“部分效果已发生但计算未完成”只会保存为
失败/待续步骤，不会伪装成普通完成步骤。原失败原因与每次用户核实记录都作为追加 JSONL 事件
保留，可通过 `log.step_failure(step)` 和 `log.audit_events(step)` 追查。

### 外部效果的补偿安排

已发生的外部效果不随检查点回退而消失。放弃某条训练分支时，可以为每个
`occurred` 效果登记补偿安排，原效果记录永远保持 `occurred`，不会被改回未执行：

```python
log.decide_compensation(step, "submit", "compensate", quantity=2.0)  # 申请抵消（限定范围）
log.decide_compensation(step, "reserve", "retain")                   # 保留登记
log.decide_compensation(step, "publish", "pending")                  # 仍等待核实
```

一次总批准可以拆给多个互相独立的外部收款对象；拆分数量之和必须等于总批准数量：

```python
log.decide_compensation(
    step, "submit", "compensate",
    quantity=10.0, allocations={"vendor-a": 4.0, "vendor-b": 6.0},
)
log.run_compensation(step, "submit", apply, quantity=2.0,
                     recipient_id="vendor-a", attempt_id="a-1")
log.compensation_allowance(step, "submit")
```

每个对象的已确认、待核对、可用额度与总批准分别核对。某对象已发出但尚未确认的数量继续占用该对象额度，
其他对象仍可并行执行。未使用额度不会自动共享；只有显式调用 `reallocate_compensation` 才会转移，
省略目标对象表示撤回。`compensation_reallocation_options` 可在不修改台账的前提下比较维持原分配、
撤回和部分转移。迟到回执必须携带原对象、尝试 ID 与回执 ID；旧对象待核对额度未解除前不能转走，
同一回执也不能在新对象重复扣记。

- `run_compensation(step, name, apply, replay, quantity=...)` 执行抵消效果，本身也会落入
  `occurred / unconfirmed / not_occurred`；已确认的补偿再次申请只会重放记录，重复申请
  不会双重抵消。请求数量超过剩余批准余额（批准 − 已确认 − 待核对）时在执行任何外部
  动作前抛出 `CompensationError`，绝不通过只裁剪内部记录来伪装符合批准。
- 回执丢失的不确定尝试会把数量记入 `quantity_pending`（保留待核对）：余额被预留，
  重试被拒绝（`UnconfirmedEffectError`），批准额度也不能降到已确认加待核对之下；
  既不会重复消耗同一批准，也不会直接当作未发生。
- `verify_compensation(step, name, occurred, ...)` 处理迟到的补偿回执；确认数量只能来自
  待核对余额，超出尝试数量或批准余额都会被拒绝。部分成功只结算已确认部分，剩余待核对
  数量保留；核实为未发生才释放预留。重复确认只追加审计记录，已补偿数量不增加。每次
  尝试与核实（含数量、理由、备注）都作为追加审计事件保留，原效果在部分补偿后被重新
  核实，也不影响补偿进度。
- 没有 `occurred` 证据的效果不能登记或执行补偿（`UnconfirmedEffectError`）。
- `log.compensation(step, name)` 返回目标效果、决定、允许范围与已补偿数量；
  `log.continuation(step)` 按声明顺序报告哪些相互依赖的效果可以继续、哪些仍与外部
  事实冲突（部分补偿、未决定、未确认都算冲突并阻断后续效果）。

### 从检查点建立训练分支

```python
branch = log.create_branch("retry", from_step=step, path="checkpoints/branch.jsonl")
branch.restore_branch(model, optimizer=opt)  # 只恢复参数/清空梯度/重置优化器状态
```

只能从 `complete` 的检查点建分支。新分支继承真实发生的外部事实（`occurred` 的原效果
与已确认补偿），但不继承另一分支尚未获确认的补偿决定。恢复只写参数数据、清空梯度并
重置优化器动量/矩估计，不执行任何参数更新，因此处理补偿不会重复应用一次更新。

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
tests/      # 数值梯度核对 + 行为测试
examples/   # 端到端训练示例
```
