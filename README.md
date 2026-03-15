# Open Quantum Steady State

这个项目把论文《Variational Neural-Network Ansatz for Steady States in Open Quantum Systems》及其 `NeuralQuantum.jl` 里最核心的链路改写成了 Python：

- `RBMSplit` 和正定的 `NeuralDensityMatrix (NDM)` 密度矩阵 ansatz
- 双态空间上的 `Metropolis` MCMC 采样
- `Stochastic Reconfiguration (SR)` 自然梯度更新
- Liouvillian 代价函数 `L^\dagger L`
- 可观测量 `Sx/Sy/Sz/H` 的精确评估接口

## 安装

```bash
python -m pip install -e .
```

## 运行论文示例对应的 1D dissipative Ising

```bash
python examples/dissipative_ising1d.py
```

如果你更喜欢模块方式，也可以运行：

```bash
python -m examples.dissipative_ising1d
```

## 说明

当前实现重点是把论文训练流程和 Julia 示例完整迁移到 Python，并保持代码结构清晰、便于后续继续扩展：

- 稀疏矩阵形式的 Liouvillian / `L^\dagger L`
- 解析的 log-derivative
- 精确 density matrix 与 observables

如果后续你要继续补 Julia 仓库里更广泛的泛化能力，比如更多采样规则、GPU、批处理缓存、任意局域算符工厂，这个结构可以继续往上长。
