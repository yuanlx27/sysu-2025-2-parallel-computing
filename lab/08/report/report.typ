#import "@preview/bubble-sysu:0.1.0": *

#show: report.with(
  title: "实验八：CUDA 矩阵转置",
  subtitle: "并行程序设计与算法实验报告",
  student: (name: "元朗曦", id: "23336294"),
  school: "计算机学院",
  major: "计算机科学与技术",
  class: "计八",
)

= 实验目的

本实验通过两个 CUDA 程序掌握 GPU 并行程序的基本组织方式和存储器访问优化方法：

- 理解 Grid、Block 和二维 Thread 的编号方式，观察设备端并行输出的顺序；
- 使用 CUDA 实现方阵转置，并验证计算结果的正确性；
- 比较矩阵规模、线程块大小、访存方式以及任务与数据划分方式对性能的影响；
- 使用 CUDA Event 测量 kernel 执行时间，并以有效内存带宽评价访存效率。

= 实验环境

- 操作系统：Linux 7.0.11，x86_64；
- CUDA 工具链：CUDA 13.3，nvcc 13.3.33；
- 主机编译器：GNU C++ 16.1.1；
- 构建工具：CMake；
- 语言标准：C++17 / CUDA C++17；

工程使用 `CMakeLists.txt` 编译 `cuda_hello_world` 和 `cuda_matrix_transpose` 两个可执行文件。根目录下的 `run.py` 自动完成 Release 构建、运行实验，并将结果写入 `report/assets`。构建配置与运行脚本均使用跨平台接口，可在安装了 CMake、Python 和 CUDA 工具链的 Linux 或 Windows 环境中运行。

= 任务一：CUDA Hello World

== 实验内容

输入三个整数 $n, m, k$，分别表示线程块数量和二维线程块的两个维度。程序创建 $n$ 个 block，每个 block 包含 $m times k$ 个 thread。主机端先输出一行提示，随后每个 GPU 线程打印自身的 `blockIdx.x`、`threadIdx.x` 和 `threadIdx.y`。

本次实验使用参数 $(n, m, k) = (2, 4, 4)$，因此理论上会产生 $2 times 4 times 4 = 32$ 条设备端输出。核心 kernel 如下。

#figure(
  ```cpp
  __global__ void helloKernel() {
      printf("Hello World from Thread (%d, %d) in Block %d!\n",
             threadIdx.x, threadIdx.y, blockIdx.x);
  }
  ```,
  caption: [CUDA Hello World 核心 kernel],
)

== 实验结果

以下内容直接读取 `assets/hello_world_output.txt`。

#figure(
  raw(read("assets/hello_world_output.txt"), lang: "text", block: true),
  caption: [CUDA Hello World 运行输出],
)

输出共包含一条主机端信息和 32 条设备端信息，线程总数与理论值一致。本次运行中 Block 1 的输出先于 Block 0，说明不同 block 的执行与输出顺序不由 block 编号决定。单个 block 内本次观察到 `threadIdx.x` 先变化、`threadIdx.y` 后变化，但这只是当前运行的表现，CUDA 不保证不同线程的设备端 `printf` 具有固定的全局顺序。其顺序会受到 block/warp 调度和设备端输出缓冲的共同影响。

= 任务二：CUDA 矩阵转置

== 实验内容

=== 算法与数据划分

对于 $n times n$ 的行主序矩阵 $A$，转置结果满足

$ A^T_(i, j) = A_(j, i). $

程序随机生成单精度浮点矩阵，并实现以下两种 kernel。

- *Naive 方法*：二维线程直接对应输入矩阵的一个元素。输入坐标为 `(row, col)`，输出位置为 `(col, row)`。连续线程读取连续输入元素，但写入转置矩阵时地址步长为 $n$，写操作难以合并。

  #figure(
    ```cpp
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
    ```,
    caption: [Naive 矩阵转置的索引映射],
  )

- *Shared-memory tiled 方法*：每个 block 负责一个 `tile × tile` 子矩阵。线程先以合并访问方式将数据读入共享内存，再交换 block 坐标，以合并访问方式写回输出矩阵。共享内存的行跨度设置为 `tile + 1`，通过 padding 避免转置访问时出现严重的 bank conflict。线程块维度为 `(tile, block_rows)`，每个线程沿行方向循环处理多个元素，从而将 tile 大小与每个 block 的线程数量解耦。

  #figure(
    ```cpp
    extern __shared__ float tile[];
    const int stride = blockDim.x + 1;

    tile[localRow * stride + threadIdx.x] =
        input[inputRow * n + inputCol];
    __syncthreads();
    output[outputRow * n + outputCol] =
        tile[threadIdx.x * stride + localCol];
    ```,
    caption: [共享内存 tiled 转置的关键访存过程],
  )

=== 计时与正确性验证

正式计时前先执行一次 warm-up。随后使用 CUDA Event 记录 20 次 kernel 执行的总时间，并计算单次平均耗时。有效内存带宽按一次读取和一次写入计算：

$ B_"eff" = (2 times n^2 times "sizeof(float)") / t. $

每次执行后，CPU 逐元素检查输出矩阵是否满足 $A^T_(i, j) = A_(j, i)$。CSV 中所有实验记录的 `correct` 字段均为 `yes`，说明两种实现和各组线程块配置均得到正确结果。由于 $n$ 最小为 512，报告不展开打印完整矩阵，而以逐元素校验覆盖全部 $n ^ 2$ 个输出元素。

== 实验结果与性能分析

表 1 直接读取 `assets/matrix_transpose_metrics.csv`，展示 benchmark 生成的全部实验记录。

#let metrics = csv("assets/matrix_transpose_metrics.csv")

#text(size: 8pt)[
  #figure(
    table(
      columns: 7,
      table.header(..metrics.at(0).map(it => strong(it))),
      ..metrics.slice(1).flatten(),
    ),
  )
]

表中字段包括矩阵规模、实现方法、tile 大小、`block_rows`、平均 kernel 时间、有效内存带宽和正确性。重复记录来自 benchmark 参数组合的重合，表格按 CSV 原始内容完整保留。

=== 性能分析

==== 矩阵规模

数据量与 $n^2$ 成正比。在最佳 shared 配置 `32×8` 下，$n$ 从 1024 增大到 2048 时，数据量扩大 4 倍，耗时由 0.0154 ms 增至 0.0582 ms，约为 3.78 倍；有效带宽则由 546.1333 GB/s 增至 576.8696 GB/s。大矩阵能够提供更多并行任务并摊薄固定开销，因此测得的有效带宽更接近设备可持续访存能力。

==== 线程块大小

Naive 方法中，`8×8` 和 `16×16` 的性能接近，而 `32×32` 明显变慢。在 $n = 2048$ 时，三者耗时分别为 0.0895 ms、0.0940 ms 和 0.3958 ms。`32×32` 包含 1024 个线程，达到单个 block 的常见线程数上限，容易受到寄存器、调度和 occupancy 约束，因此更大的 block 并不必然更快。

==== 访存方式

最佳 naive 配置在三个规模下的耗时分别为 0.0073 ms、0.0238 ms 和 0.0895 ms；shared-memory 的 `32×8` 配置分别为 0.0054 ms、0.0154 ms 和 0.0582 ms，加速比分别为 1.35、1.55 和 1.54。共享内存版本通过 tile 转置将原本跨行的离散写入转化为合并写入，并使用 padding 降低 bank conflict，因此在较优配置下取得更高带宽。

同时，shared 方法并非对所有 block 配置都占优。例如 $n = 2048$ 时，`shared 8×4` 的耗时为 0.1461 ms，慢于 `naive 8×8` 的 0.0895 ms。这说明共享内存和同步本身存在开销，只有 tile 大小和线程组织合理时，访存优化的收益才能覆盖额外成本。

==== 任务与数据划分

固定 `tile=32` 时，将 `block_rows` 从 32 减少到 16、8，$n = 2048$ 时的耗时从 0.1765 ms 降至 0.0846 ms、0.0582 ms。此时每个 block 负责的数据 tile 不变，但线程数从 1024 减少到 512、256，每个线程通过循环处理更多行。`32×8` 在线程级并行度、每线程工作量和 occupancy 之间取得了更好的平衡，是本次测试中性能最优的划分方式。

= 结论

本实验完成了 CUDA 线程层次结构输出和并行矩阵转置。实验一验证了二维线程编号与 block 编号的使用方法，并观察到设备线程输出没有固定的全局顺序。实验二验证了矩阵转置是典型的内存带宽受限问题：直接增加线程块尺寸不能保证性能提升，而合理使用共享内存、合并全局内存访问、padding 以及每线程多元素处理能够显著改善性能。在本次数据中，`tile=32`、`block_rows=8` 的共享内存实现表现最好，相对最佳 naive 实现获得约 1.35 至 1.55 倍加速，且所有配置均通过完整矩阵正确性检查。

= 复现实验

在项目根目录执行：

```bash
python3 run.py
```

脚本会调用 CMake 进行 Release 构建，随后生成：

- `report/assets/hello_world_output.txt`
- `report/assets/matrix_transpose_metrics.csv`

编译本报告可执行：

```bash
typst compile --root . report/report.typ report/report.pdf
```
