#import "@preview/bubble-sysu:0.1.0": *

#show: report.with(
  title: "实验十：CUDA 卷积",
  subtitle: "并行程序设计与算法实验报告",
  student: (name: "元朗曦", id: "23336294"),
  school: "计算机学院",
  major: "计算机科学与技术",
  class: "计八",
)

= 实验目的

本实验使用 CUDA 实现二维图像卷积，目标是掌握卷积运算的并行映射、共享内存优化、`im2col` 变换以及高性能库调用方法，并通过实际测量分析以下因素对性能的影响：

- 使用二维 Grid 和二维 Block 将输出图像位置映射到 GPU 线程；
- 比较直接滑窗、共享内存分块和 `im2col + GEMM` 三种实现；
- 分析图像规模、步幅、填充和线程块大小对性能的影响；
- 使用 cuDNN 完成相同卷积，并与自行实现的方法比较；
- 使用 CUDA Event 测量执行时间，通过 CPU 串行结果验证正确性。

= 实验环境

- 操作系统：CachyOS Linux，内核 7.0.11，x86_64；
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，计算能力 8.9，显存 8188 MiB；
- NVIDIA 驱动：610.43.02；
- CUDA 工具链：CUDA 13.3，nvcc 13.3.33；
- cuDNN：9.23.1；
- 主机编译器：GNU C++ 16.1.1；
- 构建工具：CMake 4.3.3；
- 语言标准：C++17 / CUDA C++17。

工程由根目录下的 `CMakeLists.txt` 生成单一可执行文件 `build/lab10_conv`。CMake 只负责查找 CUDA、可选的 cuDNN 开发文件并完成编译。`run.py` 负责配置 Release 构建、检测 GPU 计算能力、运行完整实验矩阵，并将结果写入 `report/assets/metrics.csv`。

= 问题描述

== 卷积定义

输入为单张 NCHW 布局的单精度图像，输入通道数 $C_"in" = 3$，高度和宽度分别为 $H$、$W$。卷积核大小固定为 $K = 3$，卷积核个数即输出通道数 $C_"out" = 3$，不使用 bias。

本实验采用神经网络中的互相关定义，不翻转卷积核。对输出通道 $f$ 和输出位置 $(y, x)$，结果为

$ Y_(f,y,x) = sum_(c=0)^(2) sum_(i=0)^(2) sum_(j=0)^(2)
  X_(c, y S+i-P, x S+j-P) W_(f,c,i,j), $

其中 $S$ 为 stride，$P$ 为 padding。输入坐标落在图像外时按零处理。输出尺寸为

$ H_"out" = floor((H + 2P - K) / S) + 1, quad
  W_"out" = floor((W + 2P - K) / S) + 1. $

当 $S=1, P=0$ 时，输出尺寸即题目给出的 $(H-2) times (W-2)$。

每个输出元素需要 $C_"in" K^2 = 27$ 次乘加，因此计算量按一次乘法和一次加法分别计数为

$ F = 2 H_"out" W_"out" C_"out" C_"in" K^2. $

吞吐率使用 GFLOP/s 表示：

$ Q = F / (t_"kernel" times 10^6), $

其中 $t_"kernel"$ 的单位为毫秒。

= CUDA 实现

== 直接滑窗卷积

直接版本令每个线程计算一个输出位置，`blockIdx.z` 对应输出卷积核。线程在寄存器中维护一个累加器，依次遍历 3 个输入通道和每个通道的 $3 times 3$ 邻域。相邻线程计算相邻输出列，在同一轮中读取相邻输入元素，因此全局内存访问具有较好的合并条件。

#figure(
  ```cpp
  const int outputX = blockIdx.x * blockDim.x + threadIdx.x;
  const int outputY = blockIdx.y * blockDim.y + threadIdx.y;
  const int filter = blockIdx.z;

  float sum = 0.0F;
  for (int channel = 0; channel < 3; ++channel) {
      for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
              sum += input[inputIndex] * weights[weightIndex];
          }
      }
  }
  output[outputIndex] = sum;
  ```,
  caption: [直接滑窗卷积的线程映射和核心计算],
)

该实现没有显式的数据复用结构，但 $3 times 3$ 窗口在相邻线程之间高度重叠，GPU 的 L1/L2 缓存能够复用一部分输入数据。卷积核总共只有 $3 times 3 times 3 times 3 = 81$ 个浮点数，也容易驻留在缓存中。

== 共享内存滑窗卷积

共享内存版本仍由每个线程计算一个输出位置，但每个 block 先协作加载覆盖其输出区域所需的输入 tile。对 `block_x × block_y` 的输出块，共享内存 tile 尺寸为

$ T_x = (B_x - 1) S + K, quad T_y = (B_y - 1) S + K. $

加载越界位置时写入零，从而统一处理 padding。每个输入通道分别经历“加载、同步、计算、同步”过程。

#figure(
  ```cpp
  for (int channel = 0; channel < 3; ++channel) {
      for (int index = linearThread; index < tileElements;
           index += threadCount) {
          tile[index] = inBounds ? input[inputIndex] : 0.0F;
      }
      __syncthreads();

      for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
              sum += tile[(localY + ky) * tileWidth + localX + kx]
                   * weights[weightIndex];
          }
      }
      __syncthreads();
  }
  ```,
  caption: [共享内存版本的 tile 加载与复用],
)

该方法显式复用同一 block 内重叠窗口的输入数据，但需要动态共享内存和每通道两次同步。由于不同输出卷积核位于不同的 `blockIdx.z`，三个卷积核仍会分别加载相同的输入 tile。

== im2col 与并行 GEMM

`im2col` 将每个卷积窗口展开为长度 27 的列向量。所有输出位置拼接后得到矩阵

$ X_"col" in RR^(27 times (H_"out" W_"out")), $

卷积核重排为

$ W_"mat" in RR^(3 times 27). $

卷积可写成矩阵乘法

$ Y_"mat" = W_"mat" X_"col",
  quad Y_"mat" in RR^(3 times (H_"out" W_"out")). $

程序首先启动一维 CUDA kernel 生成 `X_col`，然后使用实验九同类的 `16×16` 共享内存 tiled GEMM 完成矩阵乘法。计时覆盖 `im2col` 和 GEMM 两个 kernel。

#figure(
  ```cpp
  const int outputIndex = index % outputCount;
  const int kernelIndex = index / outputCount;
  const int outputY = outputIndex / outputWidth;
  const int outputX = outputIndex % outputWidth;

  columns[index] = inBounds
      ? input[(channel * height + inputY) * width + inputX]
      : 0.0F;

  launchTiledGemm(weights, columns, output,
                  3, 27, outputHeight * outputWidth);
  ```,
  caption: [im2col 展开及矩阵乘法调用],
)

该方法把不规则滑窗访问转化为规则矩阵访问，但会显式产生较大的中间矩阵。例如 `2048×2048`、`stride=1`、`padding=0` 时，`X_col` 约包含 $27 times 2046^2$ 个浮点数，占用约 452 MB，是原输入图像大小的约 9 倍。

== cuDNN 卷积

cuDNN 版本创建 NCHW 输入、输出和卷积核描述符，将卷积模式设为 `CUDNN_CROSS_CORRELATION`，然后通过 `cudnnGetConvolutionForwardAlgorithm_v7` 选择第一个可用算法并申请所需 workspace。算法选择、描述符创建和 workspace 分配均在正式计时前完成，CUDA Event 只测量 `cudnnConvolutionForward`。

#figure(
  ```cpp
  cudnnSetConvolution2dDescriptor(
      convDesc, padding, padding, stride, stride,
      1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  cudnnConvolutionForward(
      handle, &alpha, inputDesc, input,
      filterDesc, weights, convDesc, algorithm,
      workspace, workspaceBytes, &beta, outputDesc, output);
  ```,
  caption: [cuDNN 卷积配置与执行],
)

若构建环境缺少 cuDNN 开发文件，CMake 仍可编译其余实现，程序会在 CSV 中记录 `cudnn_unavailable`。本次实验环境成功启用了 cuDNN，全部 cuDNN 记录状态均为 `ok`。

= 实验方法

== 测试参数

性能测试覆盖以下参数组合：

- 输入图像：`256×256`、`512×512`、`1024×1024`、`2048×2048`；
- stride：`1`、`2`、`3`；
- padding：`0`、`1`；
- 直接和共享内存版本的 block：`8×8`、`16×16`、`32×8`、`32×16`；
- 实现：direct、direct-shared、im2col、cuDNN。

每个图像尺寸、stride 和 padding 组合包含 4 组 direct、4 组 direct-shared、1 组 im2col 和 1 组 cuDNN，共得到

$ 4 times 3 times 2 times (4 + 4 + 1 + 1) = 240 $

条记录。每组先执行 3 次 warm-up，再使用 CUDA Event 独立测量 10 次执行，并记录平均、最小和最大时间。输入、卷积核和设备缓冲区在计时前创建，因此结果反映卷积核心计算，不包含主机与设备间传输和显存分配。

== 正确性验证

正式基准前，程序使用 `17×19` 的非方形输入，对 stride 为 `1/2/3`、padding 为 `0/1` 的 6 种组合逐一执行 CPU 串行卷积，并与 direct、direct-shared、im2col 和 cuDNN 输出逐元素比较。误差判定条件为

$ abs(y_"GPU" - y_"CPU") <= 10^(-4) + 10^(-4) abs(y_"CPU"). $

验证覆盖了边界填充、输出尺寸不能整除线程块以及不同步幅。240 条记录均为 `verified=true`、`status=ok`，表示四种实现通过上述预检且基准运行正常。

= 实验结果与性能分析

== 总体结果

完整原始数据保存在 `assets/metrics.csv`。由于原表包含 240 行和 20 列，正文使用表 1 汇总 `padding=0` 时每种图像尺寸和 stride 下 direct、direct-shared 的最佳 block 时间，以及 im2col 和 cuDNN 时间。单位均为毫秒。

#figure(
  text(size: 7pt)[
    #table(
      columns: 8,
      table.header(
        [尺寸], [S], [Direct block], [Direct],
        [Shared block], [Shared], [im2col], [cuDNN],
      ),
      [256], [1], [16×16], [0.0206], [8×8], [0.0281], [0.1441], [0.1382],
      [256], [2], [32×16], [0.0094], [16×16], [0.0156], [0.0423], [0.0450],
      [256], [3], [8×8], [0.0054], [8×8], [0.0117], [0.0206], [0.0238],
      [512], [1], [32×16], [0.0618], [8×8], [0.0906], [0.5693], [0.1555],
      [512], [2], [32×16], [0.0182], [16×16], [0.0378], [0.1454], [0.0634],
      [512], [3], [16×16], [0.0126], [8×8], [0.0330], [0.0682], [0.0372],
      [1024], [1], [32×8], [0.2377], [8×8], [0.3491], [2.3229], [1.0334],
      [1024], [2], [32×16], [0.0501], [16×16], [0.1105], [0.5011], [0.1457],
      [1024], [3], [32×8], [0.0268], [8×8], [0.0831], [0.2057], [0.0766],
      [2048], [1], [32×8], [1.0071], [8×8], [1.3646], [8.1321], [2.0690],
      [2048], [2], [8×8], [0.6666], [8×8], [0.6708], [2.1108], [0.8624],
      [2048], [3], [32×8], [0.6363], [32×8], [0.6364], [1.0172], [0.3910],
    )
  ],
  caption: [padding=0 时各方法的代表性平均执行时间],
)

24 组“尺寸、stride、padding”组合中，direct 取得 21 组最快，direct-shared 取得 1 组，cuDNN 在 `2048×2048, stride=3` 的两组中最快。该结果与本实验的特殊卷积形状密切相关：batch、输入通道和输出通道都很小，单个输出只需要 27 次乘加，通用库和矩阵化方法的固定开销不容易摊薄。

== 图像规模与步幅

固定 `stride=1, padding=0` 并选择各尺寸下最佳 direct block，图像从 256 增大到 512、1024、2048 时，时间依次为 0.0206、0.0618、0.2377 和 1.0071 ms。1024 到 2048 的输出元素和计算量约扩大 4 倍，时间扩大 4.24 倍，吞吐率从 711.8 降至 673.4 GFLOP/s，整体接近按数据量线性扩展。

在 1024 尺寸下，最佳 direct 的 stride 1、2、3 时间分别为 0.2377、0.0501、0.0268 ms。输出面积近似缩小到原来的 $1/4$ 和 $1/9$，时间分别缩短到约 $1/4.75$ 和 $1/8.88$，与计算量变化基本一致。

但 2048 尺寸下，三种 stride 的 direct 时间为 1.0071、0.6666 和 0.6363 ms，stride 增大后的时间下降远小于输出元素减少比例。较大的输入工作集显著增加全局内存流量；同时 stride 增大使相邻窗口重叠减少，缓存复用率下降，kernel 更容易转为访存受限。因此减少算术运算并没有带来同比例的时间下降。`2048×2048, stride=3` 时，cuDNN 以 0.3910 ms 超过 direct，说明其在该低复用访问场景下采用了更高效的数据访问或专用算法。

padding 只影响边界和最多一行、一列输出。对最佳 direct 配置进行配对比较，padding 从 0 改为 1 的平均时间变化约为 -0.05%，总体没有稳定影响。小规模 kernel 时间只有数微秒，个别比例差异主要来自固定调度开销和测量波动。

== 线程块大小

表 2 给出每种 block 在全部尺寸、stride 和 padding 下的平均 GFLOP/s。

#figure(
  table(
    columns: 3,
    table.header([实现], [Block], [平均 GFLOP/s]),
    [Direct], [8×8], [408.1],
    [Direct], [16×16], [487.5],
    [Direct], [32×8], [492.3],
    [Direct], [32×16], [486.0],
    [Shared], [8×8], [287.4],
    [Shared], [16×16], [279.8],
    [Shared], [32×8], [279.9],
    [Shared], [32×16], [258.1],
  ),
  caption: [不同线程块配置的平均吞吐率],
)

Direct 的 `8×8` 只有 64 个线程，平均吞吐率最低；`16×16`、`32×8` 和 `32×16` 分别包含 256、256 和 512 个线程，整体差距较小，其中 `32×8` 的平均值最高。以 `2048×2048, stride=1, padding=0` 为例，`32×8` 为 1.0071 ms，相比 `8×8` 的 1.1685 ms 加速约 1.16 倍。

Shared 版本反而以 `8×8` 的平均吞吐率最高，`32×16` 最低。更大的输出 block 会扩大共享内存 tile，增加每个 block 的线程数、共享内存占用和同步等待。`2048×2048, stride=1` 时，shared 的 `8×8` 为 1.3646 ms，`32×16` 为 1.6901 ms，前者快约 1.24 倍。说明线程块越大并不必然越快，资源占用、活跃 block 数和每线程工作量需要共同权衡。

== 共享内存访存优化

对每组尺寸、stride、padding 分别选择最佳 block 后，shared 相对 direct 的时间比几何平均为 1.68，即本实现的共享内存版本整体更慢。`2048×2048, stride=1, padding=0` 时，最佳 direct 为 1.0071 ms，最佳 shared 为 1.3646 ms，shared 慢约 35.5%。

主要原因如下：

- Direct 的相邻线程读取相邻地址，能够形成合并访问；重叠窗口和很小的权重矩阵也可由硬件缓存复用；
- Shared 每个输入通道需要两次 `__syncthreads()`，三个通道共引入 6 次 block 级同步；
- `blockIdx.z` 将三个输出卷积核分配给不同 block，相同输入 tile 被重复加载三次，未实现跨卷积核复用；
- stride 增大时共享 tile 尺寸按 $S$ 扩大，而窗口重叠减少，加载到共享内存的数据复用次数下降。

在 `2048×2048, stride=3` 时，最佳 shared 与 direct 分别为 0.6364 和 0.6363 ms，差距几乎消失。此时两者均明显受到大输入全局访存的限制，共享内存额外开销相对总访存时间不再突出，但也没有形成明显优势。

== im2col 性能

`im2col + GEMM` 相对最佳 direct 的时间比几何平均为 5.89。`2048×2048, stride=1, padding=0` 时，im2col 用时 8.1321 ms，是 direct 的约 8.08 倍。

本实验的矩阵乘法形状为

$ (3 times 27) (27 times H_"out"W_"out"). $

其中输出行数只有 3。`16×16` GEMM block 中大量线程对应越界输出行，无法形成高效的大规模二维矩阵计算；同时 `im2col` 需要先写出最多约 452 MB 的中间矩阵，再由 GEMM 重新读取。对通道数和卷积核数都很小的卷积，规则矩阵乘法带来的收益不足以抵消数据展开、额外显存流量、第二次 kernel 启动和低效 GEMM 形状。

`im2col` 更适合 batch、输入通道或输出通道较大，使 GEMM 的 $M$、$N$、$K$ 三个维度都足以发挥并行矩阵乘法吞吐率的场景。本实验固定为 3 通道和 3 个卷积核，因此直接滑窗更符合问题形状。

== 与 cuDNN 比较

cuDNN 相对最佳 direct 的时间比几何平均为 2.83，但并非所有配置都较慢。代表性结果如下：

- `256×256, stride=1`：direct 0.0206 ms，cuDNN 0.1382 ms，direct 约快 6.72 倍；
- `1024×1024, stride=1`：direct 0.2377 ms，cuDNN 1.0334 ms，direct 约快 4.35 倍；
- `2048×2048, stride=1`：direct 1.0071 ms，cuDNN 2.0690 ms，direct 约快 2.05 倍；
- `2048×2048, stride=3`：cuDNN 0.3910 ms，direct 0.6363 ms，cuDNN 约快 1.63 倍。

cuDNN 面向通用深度学习卷积，需要支持不同 batch、数据类型、通道数、卷积核和布局。当前 `N=1, C_in=3, C_out=3, K=3` 的计算规模较窄，简单 direct kernel 的控制路径短，且 81 个权重很容易被缓存，因此多数配置下能够超过通用库。随着图像增大和输入复用降低，cuDNN 的算法与访存优化开始体现优势。

本次计时没有包含 cuDNN 描述符创建、算法查询和 workspace 分配，因此比较的是稳定执行阶段。若将这些一次性初始化成本计入，小规模单次卷积中 cuDNN 的端到端优势会进一步降低；实际神经网络会重复使用描述符和 workspace，因此将初始化排除更符合真实推理流程。

== 可能的改进方法

自行实现的卷积仍有以下优化空间：

- 将 81 个卷积核权重放入 CUDA constant memory，利用 warp 广播减少权重加载；
- 让一个线程同时计算三个输出卷积核，使用三个累加器复用同一输入窗口，并取消 `grid.z` 导致的输入重复读取；
- 共享内存版本一次加载三个输入通道，再完成全部输出卷积核计算，将每通道同步改为更少的整体同步；
- 针对固定的 `3×3×3` 循环完全展开，减少循环控制和地址计算；
- 为 stride 1、2、3 分别生成模板特化 kernel，消除运行时分支和整数运算；
- im2col 路径可改用 cuBLAS，并将展开与 GEMM 融合为 implicit GEMM，避免显式写入大型中间矩阵；
- 使用 half、TF32 或 Tensor Core 路径提高吞吐率，但需要重新评估数值误差。

= 结论

本实验完成了 direct、direct-shared、`im2col + tiled GEMM` 和 cuDNN 四种 CUDA 卷积实现，测试了 4 种图像规模、3 种 stride、2 种 padding 和 4 种线程块配置，共得到 240 条性能记录，四种实现均通过 CPU 预检。

实验得到以下结论：

- 对固定的 3 输入通道、3 输出卷积核和 `3×3` 核，直接滑窗最适合当前问题形状，在 24 组形状中取得 21 组最快；
- Direct 的 `16×16`、`32×8` 和 `32×16` 整体接近，`32×8` 平均吞吐率最高，为 492.3 GFLOP/s；
- 当前 shared 实现的同步和重复 tile 加载成本超过显式复用收益，整体比最佳 direct 慢约 1.68 倍；
- 显式 im2col 产生较大中间矩阵，而 `3×27` 的 GEMM 形状无法充分利用 tiled 矩阵乘法，整体比 direct 慢约 5.89 倍；
- cuDNN 在小通道、小卷积核数量下存在通用算法开销，但在 `2048×2048, stride=3` 的低复用场景中取得 1.63 倍于 direct 的速度；
- 图像规模、线程块、窗口重叠、缓存容量、共享内存同步和中间数据流量必须结合具体卷积形状共同分析，单一优化方法不能保证性能提升。

本次最佳吞吐率出现在 `1024×1024, stride=2, padding=1` 的 direct `32×16` 配置，平均时间为 0.0501 ms，达到 847.0 GFLOP/s。

= 复现实验

在项目根目录执行：

```bash
# 使用 CMake 进行 Release 构建，运行完整实验并覆盖 metrics.csv
./run.py

# 直接调用 Typst 编译 report/report.typ
./run.py --report
```
