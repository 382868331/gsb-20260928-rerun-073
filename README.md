# 计算图自动微分引擎

一个基于 NumPy 的、define-by-run（动态图）风格的小型反向自动微分引擎。
它不是符号打印器：前向时真正执行数值计算并记录算子边，反向时沿真实的
计算图传播梯度，并且能端到端训练神经网络（见 `examples/train_demo.py`，
测试集 MSE 从约 0.82 降到 0.0014）。

## 目录结构

| 文件 | 内容 |
| --- | --- |
| `autodiff/tensor.py` | `Tensor`、`Function`、`Context`、反向拓扑遍历 |
| `autodiff/ops.py` | 可微原语：四则运算、幂、`exp/log/sqrt/sin`、`relu/sigmoid/tanh`、`matmul`、`sum/mean`、`reshape/transpose`、切片 |
| `autodiff/nn.py` | `Parameter`、`Module`、`Linear`、`Sequential`、MSE 与 BCE-with-logits |
| `autodiff/optim.py` | `SGD`（动量、weight decay）与 `Adam` |
| `autodiff/grad_check.py` | 独立的中心有限差分数值梯度核对 |
| `examples/` | 训练演示、梯度检查演示 |
| `tests/` | 25 个 pytest 用例 |

## 快速上手

```python
from autodiff import Tensor, ops, nn, optim

x = Tensor([[1.0, -2.0, 0.5]], requires_grad=True)
W = Tensor([[0.2, -0.1], [0.3, 0.0], [-0.4, 0.5]], requires_grad=True)
loss = ops.relu(x @ W).sum()
loss.backward()
print(x.grad)       # 与 loss 同形状的 ndarray，梯度自动累加
```

训练一个小网络：

```python
model = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1))
opt = optim.Adam(model.parameters(), lr=0.02)
for epoch in range(200):
    opt.zero_grad()
    loss = nn.mse_loss(model(x_train), y_train)
    loss.backward()
    opt.step()        # 只能在两次前向/反向之间更新叶子
```

运行演示与测试：

```bash
python -m examples.gradcheck_demo
python -m examples.train_demo
python -m pytest tests/ -q
```

## 广播与形状规则

* 逐元素二元算子（`+ - * /`）完全遵循 NumPy 广播：从最后一维向前对齐，维度
  相等、其中一个为 1、或缺维（视为 1）时可广播，否则报错。反向时，被从 1
  拉伸的维度以及前向补出的维度会对梯度求和（`keepdims` 式还原），使输入梯度
  形状严格等于输入本身形状，例如 `(3,4) * (4,) -> (3,4)` 时右侧梯度为 `(4,)`。
* `matmul` 支持 `1-D/2-D` 组合：向量点积、矩阵-向量、向量-矩阵、矩阵乘法，
  梯度形状与各操作数一致。
* `sum/mean` 支持 `axis`（int 或 tuple）、`keepdims`，语义同 NumPy；
  `axis=None` 归约全部轴得到 0 维标量。
* `reshape/transpose/切片` 的输出梯度会散回原形状（切片未选中的位置为 0）。
* `backward()` 对标量可省略上游梯度；非标量必须显式传入同形状的 `grad_output`。
* 只有浮点张量可以 `requires_grad=True`；整数/布尔张量作为常量参与。

## 共享中间值与分支汇合

反向按拓扑序遍历，每个中间节点持有一个梯度缓冲；当一个张量被多次使用
（菱形图、重复变量、同一中间值接到多个分支）时，所有出边回传的梯度会**相加**，
与链式法则中偏导求和一致。测试见
`test_shared_intermediate_accumulates`、`test_diamond_branch_merge`、
`test_reused_variable_in_chained_expression`。

## 连续两次反传

* `backward()` 默认在结束后释放图边（叶子值和叶子身份保留）。对同一张量再次
  `backward()` 会抛 `AutogradError`，而不是算出无意义结果。
* 需要对同一图反传两次（例如梯度惩罚、自定义累积）时，每次都传
  `retain_graph=True`；梯度在叶子 `.grad` 上累加，两次反传结果为单次的两倍。
* 正常训练循环里每次迭代都重新前向建图，旧图随对象生命周期自然回收。

## 图生命周期：不依赖全局变量

图就是张量上的 `creator -> inputs` 引用关系，没有任何模块级全局 tape：
* 多张图可以同时存在、互不干扰（`test_independent_graphs_coexist`）；
* 图只被 Python 引用计数持有，函数返回/对象释放即回收；
* 每个算子节点快照其输入叶子的 `_version`。

## 原地修改与失效检测

对历史已经记录的缓冲区做原地改写会让旧图描述的计算与当前数据不一致，因此：
* 直接下标赋值、`+= *=` 等就地运算在数据缓冲区层面被直接拒绝
  （`AutogradError`，见 `test_direct_inplace_mutation_rejected`）；
* 唯一的 sanctioned 更新通道是优化器：它在两次反传之间替换叶子缓冲区并把
  `_version` 加一。若仍拿着旧图反传，版本号不匹配会立即报
  “version mismatch”（`test_stale_graph_after_leaf_replace_rejected`、
  `test_optimizer_step_invalidates_old_graph`）；更新后重新前向建图即正常。

## 独立数值核对

`autodiff/grad_check.py` 用中心差分 `(f(x+e)-f(x-e))/(2e)` 独立估计梯度，
与反向传播完全不共享代码；测试覆盖广播乘除/幂、matmul、激活函数、归约、
reshape/transpose/切片以及 BCE 损失。
