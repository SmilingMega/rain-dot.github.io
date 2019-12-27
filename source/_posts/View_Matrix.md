---
title: Understanding the View Matrix
top: false
cover: false
toc: true
mathjax: true
date: 2019-12-13 11:15:21
password:
summary:
tags: opengl viewMatrix
categories: opengl
---

- 参考文章：[Understanding the View Matrix](<https://www.3dgep.com/understanding-the-view-matrix/>)

$\qquad$在本文中，我将尝试解释如何正确构造视图矩阵以及如何使用视图矩阵将模型的顶点转换到裁剪空间。我还将尝试解释如何从视图矩阵计算摄像机在世界空间中的位置（也称为眼睛位置）。

<!-- TOC -->

- [Introduction](#introduction)
- [Convention](#convention)
  - [Memory Layout of Column-Major Matrices](#memory-layout-of-column-major-matrices)
- [Transformations](#transformations)
- [The Camera Transformation](#the-camera-transformation)
- [The View Matrix](#the-view-matrix)
- [Look At Camera](#look-at-camera)
- [FPS Camera](#fps-camera)
- [Arcball Orbit Camera](#arcball-orbit-camera)
- [Converting between Camera Transformation Matrix and View Matrix](#converting-between-camera-transformation-matrix-and-view-matrix)
- [Download the Demo](#download-the-demo)
- [Conclusion](#conclusion)

<!-- /TOC -->

# Introduction

$\qquad$了解<font color=red>**视图矩阵**</font>在3D空间中的工作原理是3D游戏编程中最被低估的概念之一。造成这种情况的原因是这种难以捉摸的矩阵的抽象性质。<font color=red>世界变换矩阵是确定3D空间中对象的位置和方向的矩阵。视图矩阵用于将模型的顶点从世界空间转换到视图空间。</font>不要认为这两件事情是一回事！

$\qquad$你可以这样想：

$\qquad$想象一下，你正拿着一台摄像机，拍一张汽车的照片。您可以通过移动相机来获得不同的汽车视图，当您通过相机的取景器查看图像时，场景似乎正在移动。<font color=red>在计算机程序中，相机根本不会移动，实际上，世界正朝着相反的方向移动，以实现相机在实际中的移动方式。</font>

为了正确理解这一点，我们必须从两个不同的角度思考：

- **<font color=red>The Camera Transformation Matrix</font>**：将相机置于世界空间中的正确位置和方向的变换（如果您想在场景中表示相机，则应用于相机的3D模型的变换）。
- **<font color=red>The View Matrix</font>**：该矩阵将顶点从世界空间转换为视图空间。<font color=blue>**该矩阵是相机变换矩阵（`camera’s transformation matrix`）的逆矩阵**</font>。

![][View-Matrix]

在上图中，摄像机的世界变换显示在左侧窗格中，摄像机的视图显示在右侧。

# Convention

$\qquad$在本文中，将矩阵视为列向量。 也就是说，在 `4×4` 齐次变换矩阵中，第一列表示 `“right”`向量（ $X$ ），第二列表示 `“up”` 向量（ $Y$ ），第三列表示 `“forward”` 向量（ $Z$ ）， 第四列表示由变换矩阵表示的空间的平移向量（原点或位置）（ $W$ ）。

$$
\begin{bmatrix}
    right_x & up_x & forward_x & position_x \\
    right_y & up_y & forward_y & position_y \\
    right_z & up_z & forward_z & position_z \\
    0       & 0    & 0         & 1          \\
\end{bmatrix}
$$

$\qquad$使用此约定，通过变换矩阵实现向量的变换。也就是说，为了通过变换矩阵 $M$ 变换向量 $v$，我们需要将列向量 $v$ 左乘矩阵 $M$。

$$
\begin{aligned}
    v' &= Mv \\
    \begin{bmatrix}
        x' \\
        y' \\
        z' \\
        w'
    \end{bmatrix} &=
    \begin{bmatrix}
        m_{0,0}, m_{0,1}, m_{0,2}, m_{0,3} \\
        m_{1,0}, m_{1,1}, m_{1,2}, m_{1,3} \\
        m_{2,0}, m_{2,1}, m_{2,2}, m_{2,3} \\
        m_{3,0}, m_{3,1}, m_{3,2}, m_{3,3} \\
    \end{bmatrix}
    \begin{bmatrix}
        x \\
        y \\
        z \\
        w
    \end{bmatrix}
\end{aligned}
$$

$\qquad$为了串联一组仿射变换（`affine transformations`）（例如平移（$T$），缩放（$S$）和旋转（$R$）），我们必须应用从右到左的变换：

$$
v' = (T(R(Sv)))
$$

- 先缩放
- 再旋转
- 然后平移

$$
M = TRS \\
v' = Mv
$$

$\qquad$要通过场景图中的父节点变换来变换场景图中的子节点，需要将孩子的局部（相对于父节点）变换矩阵乘以左侧的父世界变换矩阵：

$$
Child_{world} = Parent_{world} * Child_{local}
$$

$\qquad$当然，如果场景图中的节点没有父节点（场景图的根节点），那么节点的世界变换与其局部变换相同（相对于其父节点，在这种情况下仅为单位矩阵）:

$$
Child_{world} = Child_{local}
$$

## Memory Layout of Column-Major Matrices

$\qquad$使用列矩阵，矩阵在计算机存储器中的内存布局列连续的：

$$
M = [m_{0,0}, m_{1,0}, m_{2,0}, \cdots, m_{3,3}]
$$

$\qquad$这有一个恼人的问题，如果我们想要初始化矩阵的值，我们必须转置，以便正确加载矩阵。例如，以下布局是在 C 程序中加载列主矩阵的正确方法：

```c
// Loading a matrix in column-major order.
float right[4]    = { 1, 0, 0, 0 };
float up[4]       = { 0, 1, 0, 0 };
float forward[4]  = { 0, 0, 1, 0 };
float position[4] = { 0, 0, 0, 1 };

float matrix[4][4] = {
    {   right[0],    right[1],    right[2],    right[3] }, // First column
    {      up[0],       up[1],       up[2],       up[3] }, // Second column
    { forward[0],  forward[1],  forward[2],  forward[3] }, // Third column
    {position[0], position[1], position[2], position[3] }  // Forth column
};
```

$\qquad$乍一看，你会想“等一下，矩阵是行主序！”。是的，这确实是事实。行主序矩阵在内存中以相同的顺序存储它的元素，各个向量被认为是行而不是列。

$\qquad$那么最大的区别是什么呢？在执行矩阵乘法的函数中可以看到差异。我们来看一个例子吧。

$\qquad$假设我们有以下C ++定义：

```cpp
struct vec4 {
    float values[4];

    vec4() {
        values[0] = values[1] = values[2] = values[3] = 0;
    }

    vec4( float x, float y, float z, float w ) {
        values[0] = x;
        values[1] = y;
        values[2] = z;
        values[3] = w;
    }

    // Provide array-like index operators for the components of the vector.
    const float& operator[] ( int index ) const {
        return values[index];
    }
    float& operator[] ( int index ) {
        return values[index];
    }
};

struct mat4 {
    vec4 columns[4];

    mat4() {
        columns[0] = vec4( 1, 0, 0, 0 );
        columns[1] = vec4( 0, 1, 0, 0 );
        columns[2] = vec4( 0, 0, 1, 0 );
        columns[3] = vec4( 0, 0, 0, 1 );
    }

    mat4( vec4 x, vec4 y, vec4 z, vec4 w ) {
        columns[0] = x;
        columns[1] = y;
        columns[2] = z;
        columns[3] = w;
    }

    // Provide array-like index operators for the columns of the matrix.
    const vec4& operator[]( int index ) const {
        return columns[index];
    }
    vec4& operator[]( int index ) {
        return columns[index];
    }
};
```

- `vec4` 结构提供了一个索引运算符，允许使用索引来访问向量的各个分量。这使代码更容易阅读。值得注意的是，`vec4` 结构可以解释为行向量或列向量。在这种情况下无法区分差异。
- `mat4` 结构提供了一个索引运算符，允许使用索引来访问矩阵的各个列（而不是行！）。

使用这种技术，为了访问矩阵 $M$ 的第 $i$ 列和第 $j$ 列，我们需要访问矩阵的元素，如下所示：

```cpp
// main.cpp
int i = row;
int j = column;
mat4 M;

// Access the i-th row and the j-th column of matrix M
float m_ij = M[j][i];
```

$\qquad$这非常烦人，我们必须交换 `i` 和 `j` 索引才能访问正确的矩阵元素。这可能是编程时使用行主序矩阵而不是列主序矩阵的一个很好的理由，但线性代数教科书和学术研究论文中最常见的惯例是使用列主序矩阵。因此，使用列主序矩阵主要是出于历史原因。

$\qquad$现在假设我们定义了以下函数：

```cpp
// Matrix-vector multiplication

// Pre-multiply a vector by a multiplying a matrix on the left.
vec4 operator*( const mat4& m, const vec4& v );
// Post-multiply a vector by multiplying a matrix on the right.
vec4 operator*( const vec4& v, const mat4& m );
// Matrix multiplication
mat4 operator*( const mat4& m1, const mat4& m2 );
```

- 第一个函数：$m(4 \times 4) * v(4 \times 1)$
- 第二种函数：$\,\,v(1 \times 4) * m(4 \times 4)$
- 第三种函数：$m(4 \times 4) * m(4 \times 4)$

那么预乘函数看起来像这样：

```cpp
// Pre-multiply a vector by a matrix on the left.
vec4 operator*( const mat4& m, const vec4& v )
{
    return vec4(
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
        m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3]
    );
}
```

- 请注意，我们仍然将矩阵 `m` 的行与列向量 `v` 相乘，但 `m` 的索引进行了交换。

```cpp
// Pre-multiply a vector by a matrix on the right.
vec4 operator*( const vec4& v, const mat4& m )
{
    return vec4(
        v[0] * m[0][0] + v[1] * m[0][1] + v[2] * m[0][2] + v[3] * m[0][3],
        v[0] * m[1][0] + v[1] * m[1][1] + v[2] * m[1][2] + v[3] * m[1][3],
        v[0] * m[2][0] + v[1] * m[2][1] + v[2] * m[2][2] + v[3] * m[2][3],
        v[0] * m[3][0] + v[1] * m[3][1] + v[2] * m[3][2] + v[3] * m[3][3]
    );
}
```

- 在这种情况下，我们将行向量 `v` 乘以矩阵 `m` 的列。请注意，我们仍然需要交换索引来访问矩阵 `m` 的正确列和行。

并且执行矩阵矩阵乘法的最终函数：

```cpp
// Matrix multiplication
mat4 operator*( const mat4& m1, const mat4& m2 )
{
    vec4 X(
        m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1] + m1[2][0] * m2[0][2] + m1[3][0] * m2[0][3],
        m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1] + m1[2][1] * m2[0][2] + m1[3][1] * m2[0][3],
        m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1] + m1[2][2] * m2[0][2] + m1[3][2] * m2[0][3],
        m1[0][3] * m2[0][0] + m1[1][3] * m2[0][1] + m1[2][3] * m2[0][2] + m1[3][3] * m2[0][3]
    );
    vec4 Y(
        m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1] + m1[2][0] * m2[1][2] + m1[3][0] * m2[1][3],
        m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] + m1[2][1] * m2[1][2] + m1[3][1] * m2[1][3],
        m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1] + m1[2][2] * m2[1][2] + m1[3][2] * m2[1][3],
        m1[0][3] * m2[1][0] + m1[1][3] * m2[1][1] + m1[2][3] * m2[1][2] + m1[3][3] * m2[1][3]
    );
    vec4 Z(
        m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1] + m1[2][0] * m2[2][2] + m1[3][0] * m2[2][3],
        m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1] + m1[2][1] * m2[2][2] + m1[3][1] * m2[2][3],
        m1[0][2] * m2[2][0] + m1[1][2] * m2[2][1] + m1[2][2] * m2[2][2] + m1[3][2] * m2[2][3],
        m1[0][3] * m2[2][0] + m1[1][3] * m2[2][1] + m1[2][3] * m2[2][2] + m1[3][3] * m2[2][3]
    );
    vec4 W(
        m1[0][0] * m2[3][0] + m1[1][0] * m2[3][1] + m1[2][0] * m2[3][2] + m1[3][0] * m2[3][3],
        m1[0][1] * m2[3][0] + m1[1][1] * m2[3][1] + m1[2][1] * m2[3][2] + m1[3][1] * m2[3][3],
        m1[0][2] * m2[3][0] + m1[1][2] * m2[3][1] + m1[2][2] * m2[3][2] + m1[3][2] * m2[3][3],
        m1[0][3] * m2[3][0] + m1[1][3] * m2[3][1] + m1[2][3] * m2[3][2] + m1[3][3] * m2[3][3]
    );

    return mat4( X, Y, Z, W );
}
```

- 此函数将 `m1` 的行乘以 `m2` 的列。请注意，我们必须交换 `m1` 和 `m2` 中的索引。

如果我们重用函数，可以稍微简化这个函数：

```cpp
// Matrix multiplication
mat4 operator*( const mat4& m1, const mat4& m2 )
{
    vec4 X = m1 * m2[0];
    vec4 Y = m1 * m2[1];
    vec4 Z = m1 * m2[2];
    vec4 W = m1 * m2[3];

    return mat4( X, Y, Z, W );
}
```

**重点是，无论您使用什么约定，您都坚持使用并保持一致，并始终确保您在API中清楚地记录您正在使用的约定。**

# Transformations

$\qquad$在3D空间中渲染场景时，通常会有3个变换应用于场景中的3D几何体：

- **`World Transform`**：世界变换（或有时称为对象变换或**模型矩阵**）将从对象空间变换模型顶点（和法线）（这是使用 `3D Studio Max` 或`Maya` 等 `3D` 内容创建工具创建模型的空间）进入世界空间。世界空间将模型定位在世界中正确位置的位置，方向（有时是比例）。
- **`View Transform`**：**世界空间顶点位置（和法线）需要转换为相对于摄像机视图的空间**。这被称为“视图空间”（有时称为“相机空间”），是本文将要研究的转换。
- **`Projection Transform`**：**已经变换到视图空间的顶点需要通过投影变换矩阵变换到称为“裁剪空间”的空间中**。这是图形程序员需要担心的最终空间。本文不讨论投影变换矩阵。

$\qquad$如果我们将相机视为场景中的一个物体（就像放置在场景中的任何其他物体一样），那么相机需要一个变换矩阵在世界空间中定位。但是由于我们想要从相机的视角渲染场景，因此我们需要找到一个变换矩阵，将相机转换到“视图空间”。换句话说，我们需要一个转换矩阵，该矩阵将照相机对象放置在指向 $Z$ 轴（正 $Z$ 轴或负 $Z$ 轴）的世界原点上，这取决于我们是左手还是右手 坐标系。换句话说，我们需要找到矩阵 $V$：

$$
I = VM
$$

其中 $M$ 是相机变换矩阵，$V$ 是我们正在寻找的视图矩阵，其将相机变换矩阵变换为单位矩阵 $I$。

嗯，显然矩阵 $V$ 只是 $M$ 的逆矩阵。那是，

$$
V = M^{-1}
$$

$V$ 矩阵用于将场景中的任何对象从世界空间转换为视图空间（或相机空间）。

# The Camera Transformation

$\qquad$相机变换是用于将相机在世界空间中定位的矩阵。这与在场景中定位模型的模型矩阵相同。不应将此转换视为视图矩阵，它不能直接用于将顶点从世界空间转换为视图空间。

$\qquad$计算相机的变换矩阵与计算场景中其他对象的变换矩阵没有什么不同。

$\qquad$如果 $R$ 表示摄像机的方向，并且 $T$ 表示摄像机在世界空间中的平移，则可以通过将两个矩阵相乘来计算摄像机的变换矩阵 $M$。

$$
M = TR
$$

请记住，由于我们正在处理列主序矩阵，因此从右到左读取。也就是说，首先旋转，然后平移。

# The View Matrix

$\qquad$另一方面，视图矩阵用于将顶点从世界空间转换为视图空间（摄像机空间）。通常将此矩阵与对象的世界矩阵和投影矩阵连接在一起，以便可以在顶点程序中将顶点从对象空间直接转换到裁剪空间。

$\qquad$如果 $M$ 表示对象的模型矩阵，$V$ 表示视图矩阵，$P$ 是投影矩阵，则连接模型，视图，投影可以简单地通过将三个矩阵相乘：

$$
MVP = P * V * M
$$

$\qquad$通过与组合矩阵 $MVP$ 相乘，可以将顶点 $v$ 转换到裁剪空间：

$$
v′=MVP∗v
$$

$\qquad$这就是它的用法，那么视图矩阵是如何计算的？有几种计算视图矩阵的方法，首选方法通常取决于您打算如何使用它。

$\qquad$推导视图矩阵的一种常用方法是在给定相机在世界空间中的位置的情况下，计算一个 `Look-at` 矩阵（通常称为“眼睛”位置），一个 `up`向量（通常是 $[0 \quad 1 \quad 0]^T$），以及一个在世间空间中的目标点。

$\qquad$如果要创建 `first-person-shooter`（FPS），则可能不会使用 `Look-at` 方法来计算视图矩阵。这种情况下，使用一种基于世界空间中的位置、俯仰（绕 $X$ 轴旋转）和偏航（绕 $Y$ 轴旋转）角度（通常我们不这样做）来计算视图矩阵的方法会更加方便。

$\qquad$如果要创建可用于绕3D对象旋转的摄像机，则可能要创建一个 Arcball 摄像机。

$\qquad$我将在以下各节中讨论这3种典型的相机。

# Look At Camera

$\qquad$使用此方法，我们可以直接从相机（眼睛）的世界位置，`up`向量和目标点计算视图矩阵。

$\qquad$此功能的典型实现（假设右手坐标系的相机在 $-Z$ 轴上看）可能看起来像这样：

```cpp
mat4 LookAtRH( vec3 eye, vec3 target, vec3 up )
{
    vec3 zaxis = normal(eye - target);    // The "forward" vector.
    vec3 xaxis = normal(cross(up, zaxis));// The "right" vector.
    vec3 yaxis = cross(zaxis, xaxis);     // The "up" vector.

    // Create a 4x4 orientation matrix from the right, up, and forward vectors
    // This is transposed which is equivalent to performing an inverse
    // if the matrix is orthonormalized (in this case, it is).
    mat4 orientation = {
       vec4( xaxis.x, yaxis.x, zaxis.x, 0 ),
       vec4( xaxis.y, yaxis.y, zaxis.y, 0 ),
       vec4( xaxis.z, yaxis.z, zaxis.z, 0 ),
       vec4(   0,       0,       0,     1 )
    };

    // Create a 4x4 translation matrix.
    // The eye position is negated which is equivalent
    // to the inverse of the translation matrix.
    // T(v)^-1 == T(-v)
    mat4 translation = {
        vec4(   1,      0,      0,   0 ),
        vec4(   0,      1,      0,   0 ),
        vec4(   0,      0,      1,   0 ),
        vec4(-eye.x, -eye.y, -eye.z, 1 )
    };

    // Combine the orientation and translation to compute
    // the final view matrix. Note that the order of
    // multiplication is reversed because the matrices
    // are already inverted.
    return ( orientation * translation );
}
```

$\qquad$可以稍微优化此方法，可以消除矩阵乘法。

```cpp
mat4 LookAtRH( vec3 eye, vec3 target, vec3 up )
{
    vec3 zaxis = normal(eye - target);    // The "forward" vector.
    vec3 xaxis = normal(cross(up, zaxis));// The "right" vector.
    vec3 yaxis = cross(zaxis, xaxis);     // The "up" vector.

    // Create a 4x4 view matrix from the right, up, forward and eye position vectors
    mat4 viewMatrix = {
        vec4(      xaxis.x,            yaxis.x,            zaxis.x,       0 ),
        vec4(      xaxis.y,            yaxis.y,            zaxis.y,       0 ),
        vec4(      xaxis.z,            yaxis.z,            zaxis.z,       0 ),
        vec4(-dot( xaxis, eye ), -dot( yaxis, eye ), -dot( zaxis, eye ),  1 )
    };

    return viewMatrix;
}
```

$\qquad$Nate Robins 的 OpenGL 中使用 gluLookAt 函数的一个很好的例子：http://user.xmission.com/~nate/tutors.html

# FPS Camera

$\qquad$如果要实现 FPS camera，则可能要根据一组欧拉角（俯仰和偏航）和已知的世界位置来计算视图矩阵。相机模型的基本理论是，我们要构建一个相机矩阵，该矩阵首先围绕 $X$ 轴旋转俯仰角，然后围绕 $Y$ 轴旋转偏航角，然后平移到世界上的某个位置。由于我们需要视图矩阵，因此需要计算所得矩阵的逆。

$$
V=(T(RyRx))^{−1}
$$

```cpp
/**
 * FPS camera, right-handed coordinate system.
 */
// Pitch must be in the range of [-90 ... 90] degrees and 
// yaw must be in the range of [0 ... 360] degrees.
// Pitch and yaw variables must be expressed in radians.
mat4 FPSViewRH( vec3 eye, float pitch, float yaw )
{
    // I assume the values are already converted to radians.
    float cosPitch = cos(pitch);
    float sinPitch = sin(pitch);
    float cosYaw = cos(yaw);
    float sinYaw = sin(yaw);

    vec3 xaxis = { cosYaw, 0, -sinYaw };
    vec3 yaxis = { sinYaw * sinPitch, cosPitch, cosYaw * sinPitch };
    vec3 zaxis = { sinYaw * cosPitch, -sinPitch, cosPitch * cosYaw };

    // Create a 4x4 view matrix from the right, up, forward and eye position vectors
    mat4 viewMatrix = {
        vec4(       xaxis.x,            yaxis.x,            zaxis.x,      0 ),
        vec4(       xaxis.y,            yaxis.y,            zaxis.y,      0 ),
        vec4(       xaxis.z,            yaxis.z,            zaxis.z,      0 ),
        vec4( -dot( xaxis, eye ), -dot( yaxis, eye ), -dot( zaxis, eye ), 1 )
    };

    return viewMatrix;
}
```

$\qquad$在此函数中，我们首先计算视图矩阵的轴。这是从围绕 $X$ 轴旋转然后跟随着 $Y$ 轴旋转的串联中得出的。然后，我们利用矩阵的最后一列只是基本矢量与摄像机的眼睛位置的点积这一事实，与以前一样构建视图矩阵。

# Arcball Orbit Camera

# Converting between Camera Transformation Matrix and View Matrix

$\qquad$如果仅具有摄影机转换矩阵 $M$，并且要计算可正确将顶点从世界空间转换为视图空间的视图矩阵 $V$，则只需要进行摄影机转换的逆运算即可。

$$
V=M^{−1}
$$

$\qquad$如果只有视图矩阵 $V$，并且需要找到可用于在场景中放置摄像机视觉表示的摄影机转换矩阵 $M$，则只需取视图矩阵的逆即可。

$$
M = V^{-1}
$$

$\qquad$当您只能访问视图矩阵 $V$ 并且想找出相机在世界空间中的位置时，通常在着色器中使用此方法。在这种情况下，您可以使用倒置视图矩阵的第4列来确定摄影机在世界空间中的位置：

$$
M=V^{−1} \\
eye_{world}=[M_{0,3} \qquad M_{1,3} \qquad M_{2,3}]
$$

$\qquad$当然，建议将摄像机的眼睛位置简单地作为变量传递给着色器，而不是在每次调用顶点着色器或片段着色器时反转视图矩阵。

# Download the Demo

# Conclusion

$\qquad$我希望我已经弄清楚了相机的转换矩阵和视图矩阵之间的区别，以及如何在彼此之间进行转换。知道要处理的矩阵也很重要，这样您才能正确获得相机的眼睛位置。使用相机的世界变换时，眼睛位置是世界变换的第 4 行，但是如果使用视图矩阵，则必须先对矩阵求逆，然后才能提取世界空间中的眼睛位置。



[View-Matrix]:data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABIoAAAI1CAMAAACOg+FdAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA2ZpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDExIDY2LjE0NTY2MSwgMjAxMi8wMi8wNi0xNDo1NjoyNyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDpFQUMzNkFCRTVBNUNFMzExQTJEQkE0QzNFMTM4MUQwRiIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDowRjkwNzZCRTVEOEYxMUUzQUE1NjhENjhGQjRERTE0RCIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDowRjkwNzZCRDVEOEYxMUUzQUE1NjhENjhGQjRERTE0RCIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ1M2IChXaW5kb3dzKSI+IDx4bXBNTTpEZXJpdmVkRnJvbSBzdFJlZjppbnN0YW5jZUlEPSJ4bXAuaWlkOkU1NjM1NkI4OEU1REUzMTFCNDBEQkI5QUU4N0JCODc4IiBzdFJlZjpkb2N1bWVudElEPSJ4bXAuZGlkOkVBQzM2QUJFNUE1Q0UzMTFBMkRCQTRDM0UxMzgxRDBGIi8+IDwvcmRmOkRlc2NyaXB0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

