#import "@preview/bubble-sysu:0.1.0": *

#show: report.with(
  title: "实验九：CUDA 矩阵乘法",
  subtitle: "并行程序设计与算法实验报告",
  student: (name: "元朗曦", id: "23336294"),
  school: "计算机学院",
  major: "计算机科学与技术",
  class: "计八",
)

= 实验目的

本实验使用 CUDA 实现通用矩阵乘法，目标是掌握计算密集型并行程序的线程组织、访存优化和任务划分方法，并通过实际测量分析以下因素对性能的影响：

- 使用二维 Grid 和二维 Block 将输出矩阵元素映射到 GPU 线程；
- 比较直接访问全局内存和共享内存分块两种访存方式；
- 比较每线程计算一个输出元素和每线程计算两个输出元素的任务划分；
- 分析矩阵规模和线程块大小对设备利用率、吞吐率及端到端时间的影响；
- 使用 CUDA Event 计时，并通过 CPU 抽样计算验证结果正确性。

= 实验环境

- 操作系统：CachyOS Linux，内核 7.0.11，x86_64；
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，计算能力 8.9，显存 7834 MiB；
- NVIDIA 驱动：610.43.02；
- CUDA 工具链：CUDA 13.3，nvcc 13.3.33；
- 主机编译器：GNU C++ 16.1.1；
- 构建工具：CMake 4.3.3；
- 语言标准：C++17 / CUDA C++17。

工程仅包含一个 CUDA 源文件 `src/matmul.cu`，由根目录下的 `CMakeLists.txt` 生成 `build/matmul`。`run.py` 负责 Release 构建、运行 45 组测试，并将结果原子覆盖写入 `report/assets/metrics.csv`。

= 问题描述与并行设计

== 问题定义

输入三个整数 $m, n, k$，取值范围均为 $[128, 2048]$。程序随机生成行主序单精度矩阵

$ A in RR^(m times n), quad B in RR^(n times k), $

并计算

$ C = A B, quad C_(i,j) = sum_(p=0)^(n-1) A_(i,p) B_(p,j). $

矩阵乘法共执行约 $2 m n k$ 次浮点运算。程序输出矩阵信息、kernel 平均执行时间、端到端时间和吞吐率。性能使用 GFLOP/s 表示：

$ P = (2 m n k) / (t_"kernel" times 10^6), $

其中 $t_"kernel"$ 的单位为毫秒。

== Naive：每线程一个输出元素

Naive kernel 使用二维线程坐标直接确定 $C$ 中的一个元素。线程遍历矩阵 A 的一行和矩阵 B 的一列，完成长度为 $n$ 的点积。

#figure(
  ```cpp
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= k) return;

  float sum = 0.0F;
  for (int p = 0; p < n; ++p) {
      sum += a[row * n + p] * b[p * k + col];
  }
  c[row * k + col] = sum;
  ```,
  caption: [Naive kernel 的线程映射与点积计算],
)

同一 warp 中相邻线程计算同一行的相邻输出列，因此每一轮对 B 的访问是连续的，可以合并；这些线程读取相同的 A 元素，硬件缓存和广播机制能够减少部分重复访存。但每个输出元素仍要从全局内存读取 $2n$ 个操作数，数据复用主要依赖缓存。

== Tiled：共享内存分块

Tiled kernel 将 A 和 B 划分为 `tile × tile` 子矩阵。每个 block 的线程协作将两个 tile 从全局内存载入共享内存，再完成 tile 内乘加。若 $n$ 不是 tile 的整数倍，越界位置补零，因此实现也适用于一般的 $m, n, k$。

#figure(
  ```cpp
  tileA[localRow * tileSize + localCol] =
      (row < m && aCol < n) ? a[row * n + aCol] : 0.0F;
  tileB[localRow * tileSize + localCol] =
      (bRow < n && col < k) ? b[bRow * k + col] : 0.0F;
  __syncthreads();

  for (int p = 0; p < tileSize; ++p) {
      sum += tileA[localRow * tileSize + p]
           * tileB[p * tileSize + localCol];
  }
  __syncthreads();
  ```,
  caption: [共享内存 tiled kernel 的核心循环],
)

每个载入共享内存的元素可被 block 内多个线程复用，理论上能降低全局内存流量；代价是动态共享内存、每个 tile 两次同步以及额外的装载与边界判断。

== Coarsened：每线程两个输出元素

Coarsened kernel 在 tiled 方法上进行线程粗化。一个线程计算同一输出行中相距一个 tile 的两个元素 `col0` 和 `col1`，同时保留两个累加器。每轮只载入一份 A tile，但载入并使用两份相邻的 B tile。

#figure(
  ```cpp
  const int col0 = blockIdx.x * (2 * tileSize) + localCol;
  const int col1 = col0 + tileSize;

  for (int p = 0; p < tileSize; ++p) {
      const float av = tileA[localRow * tileSize + p];
      sum0 += av * tileB[p * (2 * tileSize) + localCol];
      sum1 += av * tileB[p * (2 * tileSize) + tileSize + localCol];
  }
  ```,
  caption: [线程粗化后复用 A 数据计算两个输出元素],
)

相对于每线程一个元素，该划分将 x 方向 block 数约减半，并使一个 A 值服务于两次乘加，提高每线程计算量和算术强度。但两个累加器及更多地址计算会增加寄存器压力，小矩阵中也可能无法抵消额外控制开销。

= 实验方法

== 测试参数

性能测试使用方阵 $m = n = k$，覆盖以下笛卡尔积：

- 矩阵规模：`128`、`256`、`512`、`1024`、`2048`；
- 方形线程块：`8×8`、`16×16`、`32×32`；
- kernel：`naive`、`tiled`、`coarsened`。

共得到 $5 times 3 times 3 = 45$ 组记录。每组先执行 2 次 warm-up，再用 CUDA Event 测量 10 次 kernel 的总时间并取平均。`total_ms` 从设备内存分配开始计时，包含 H2D 传输、warm-up、正式执行、D2H 传输和同步，因此用于分析端到端开销；不同实现的核心计算能力主要比较 `kernel_ms` 和 GFLOP/s。

== 正确性验证

为避免对最大规模矩阵再执行一次完整的串行 $O(m n k)$ 乘法，程序在输出矩阵的行、列方向各均匀选取 8 个位置，共验证 64 个元素。CPU 使用 double 累加得到参考值，并计算

$ e_"rel" = abs(c_"GPU" - c_"CPU") / max(1, abs(c_"CPU")). $

最大相对误差不超过 $10^(-3)$ 时判定为通过。45 组记录的 `verify` 均为 `PASS`；最大绝对误差为 $4.119 times 10^(-5)$，最大相对误差为 $1.078 times 10^(-5)$，说明三个 kernel 在全部测试配置下均得到正确结果。

= 实验结果与性能分析

== 总体结果

表 1 直接读取 `assets/metrics.csv`，展示 `run.py` 生成的全部 45 组实验记录。

#let metrics = csv("assets/metrics.csv")

#figure(
  text(size: 5pt)[
    #table(
      columns: 13,
      table.header(..metrics.at(0).map(it => strong(it))),
      ..metrics.slice(1).flatten(),
    )
  ],
  caption: [全部矩阵乘法性能测试结果],
)

表中字段依次为矩阵三个维度、kernel、线程块大小、预热和重复次数、平均 kernel 时间、端到端时间、吞吐率、最大抽样绝对与相对误差以及验证状态。全部记录的 `verify` 均为 `PASS`。

== 矩阵规模

固定为整体表现最稳定的 `16×16` block，矩阵从 1024 阶增大到 2048 阶时，运算量扩大 8 倍，而 coarsened 时间由 2.4153 ms 增至 15.1832 ms，约为 6.29 倍；吞吐率由 889.1 增至 1131.5 GFLOP/s。说明较大矩阵提供了更多并行工作并提高了设备利用率，固定开销被进一步摊薄。

除 128 阶小矩阵外，最佳结果均来自 `coarsened + 16×16`。最大规模 2048 的吞吐率达到 1131.5 GFLOP/s。小矩阵只有很少的 block，GPU 并行资源没有充分利用，且 kernel 启动与调度等固定成本占比更高，因此吞吐率较低。

128 阶时 naive 最快，而从 256 阶开始 coarsened 持续领先。线程粗化的复用收益需要足够多的计算才能覆盖额外的寄存器和控制成本，因此其优势随问题规模增大更加明显。

== 线程块大小

以 1024 阶矩阵为例，naive 在 `8×8`、`16×16`、`32×32` 下的吞吐率分别为 667.6、825.9、757.2 GFLOP/s；tiled 分别为 613.3、729.3、627.8 GFLOP/s；coarsened 分别为 814.4、889.1、859.8 GFLOP/s。

`8×8` 只有 64 个线程，单个 block 的并行度和 tile 内复用范围较小；`32×32` 包含 1024 个线程，达到单 block 常见线程数上限，并增加每块寄存器和共享内存资源占用，可能限制同时驻留的 block 数。`16×16` 的 256 个线程在并行度、tile 复用和 occupancy 之间取得更稳定的平衡。

跨五种规模取 GFLOP/s 算术平均，naive 的 `8/16/32` 分别为 574.2、735.8、716.5；tiled 为 549.8、647.8、593.6；coarsened 为 721.5、800.2、728.0。三种实现的平均值均以 `16×16` 最高，说明该配置不是只在单一规模上偶然占优。

== 访存方式

共享内存 tiled 方法的理论目标是显式复用 A、B tile，减少全局内存重复读取。但实测中它并未普遍优于 naive：在 `16×16` 下，五种规模的 tiled 吞吐率均低于 naive。例如 2048 阶时二者分别为 845.6 和 956.8 GFLOP/s。

这一结果与实际访问模式有关。Naive kernel 中，相邻线程每轮连续读取 B，因此访存可以合并；同一 warp 对 A 的地址相同，也可利用缓存或广播。当前 tiled 实现则需要动态共享内存、每轮两次 `__syncthreads()`、额外边界判断，并使用运行时 `tileSize` 循环，使同步和控制开销可能超过显式复用的收益。此外，RTX 4060 的 L1/L2 缓存已经能够缓解部分重复访问。

共享内存仍在特定配置下有效。2048 阶、`8×8` block 时，tiled 时间为 24.3672 ms，naive 为 28.2656 ms，取得约 1.16 倍加速。这说明共享内存本身不是性能保证，tile 大小、同步成本和硬件缓存行为必须共同考虑。

== 任务与数据划分

线程粗化是本实验最有效的优化。2048 阶、`16×16` 时，coarsened 时间为 15.1832 ms，相比 naive 的 17.9552 ms 加速 1.18 倍，相比 tiled 的 20.3160 ms 加速 1.34 倍。

该方法让每个线程计算同一行的两个输出元素。一份 A tile 同时参与两组输出计算，x 方向 block 数约减半，每个线程的算术工作量增加，因而减少部分调度和 A 数据装载开销。虽然两个累加器会增加寄存器使用，但在 256 至 2048 阶矩阵上，其数据复用收益占主导。128 阶时 `coarsened + 16×16` 比 naive 慢约 12.6%，再次说明任务粗化需要足够大的工作集才能发挥作用。

== Kernel 时间与端到端时间

小矩阵的 `total_ms` 约为数十毫秒，而单次 kernel 仅约 0.01 ms。原因是端到端计时包含显存分配、H2D/D2H 传输、2 次 warm-up、10 次正式执行和同步，固定成本远大于小矩阵计算本身。随着规模增大，kernel 占比显著上升；例如 2048 阶 coarsened `16×16` 的单次平均 kernel 时间为 15.1832 ms，端到端时间为 262.9897 ms，其中 12 次 kernel 已占约 182.2 ms。

因此，评价算法核心并行效率应使用 CUDA Event 得到的 `kernel_ms`；评价实际应用则还需考虑数据传输和内存管理。若矩阵连续参与多次运算，应复用设备缓冲区并尽量让数据驻留 GPU，以摊薄端到端开销。

= 结论

本实验完成了三种 CUDA 通用矩阵乘法实现，并在 RTX 4060 Laptop GPU 上测试了 5 种矩阵规模、3 种线程块和 3 种任务/访存组织，共 45 组配置，所有结果均通过正确性验证。

实验得到以下结论：

- 矩阵规模增大后，GPU 并行资源利用率提高，吞吐率整体上升；
- `16×16` block 在三种实现中均表现最稳定，过小的 block 并行度不足，`32×32` 则可能受到资源占用限制；
- 共享内存分块需要付出同步和数据搬运成本，只有当复用收益超过这些成本时才会优于依赖硬件缓存的 naive 方法；
- 每线程计算两个输出元素能够复用 A 数据并减少线程任务数量，在 256 至 2048 阶矩阵上均取得最佳性能；
- 本次最佳配置为 2048 阶矩阵、`coarsened` kernel、`16×16` block，平均 kernel 时间 15.1832 ms，吞吐率 1131.5 GFLOP/s。

由此可见，CUDA 优化不能只依赖单一技巧。线程块大小、共享内存、缓存行为、同步开销、寄存器压力和每线程工作量需要结合具体 GPU 与问题规模实测权衡。

= 复现实验

在项目根目录执行以下命令：

```bash
# 重新构建并运行 45 组测试，覆盖生成 metrics.csv
./run.py

# 构建 CUDA 程序并编译本报告
./run.py --report
```
