# Linear Regression — Bài giảng chi tiết

> **Nguồn:** Slide `slides-v1/foundation/03-RegressionSummary.pdf` — Th.S Lê Thanh Sách, Khoa KHMT, ĐH Bách Khoa TP.HCM (05/02/2026)

---

## Mục lục

1. [Architecture View — Nhìn từ góc độ kiến trúc](#1-architecture-view)
2. [Mathematical View — Nhìn từ góc độ toán học](#2-mathematical-view)
3. [Programming View — Nhìn từ góc độ lập trình](#3-programming-view)
4. [Training Methods — Các phương pháp huấn luyện](#4-training-methods)

---

## 1. Architecture View

### Tổng quan kiến trúc mô hình

> 📷 **[Ảnh slide 4]** Sơ đồ kiến trúc: `X (D) → [FC | W, b] → Ŷ (K)`
> Nguồn: slide 4, `03-RegressionSummary.pdf`

**Linear Regression** về mặt kiến trúc chỉ là một lớp duy nhất: lớp **Fully-Connected (FC)**. Đầu vào là vector đặc trưng **X**, đi qua một phép biến đổi tuyến tính với tham số **W** và **b**, cho ra đầu ra dự đoán **Ŷ**.

Hãy tưởng tượng bạn muốn dự đoán giá nhà. Đầu vào là các đặc trưng như diện tích, số phòng ngủ, vị trí — tất cả gộp thành vector **x**. Mô hình chỉ đơn giản là kéo một đường thẳng qua không gian nhiều chiều này để đưa ra dự đoán.

---

### Quy ước ký hiệu (Notations)

Trước khi đi vào toán học, chúng ta cần thống nhất ngôn ngữ ký hiệu:

| Ký hiệu | Ý nghĩa |
|---------|---------|
| Vector | chữ thường in đậm, là **vector cột** (e.g., **x**, **b**) |
| Ma trận | chữ hoa in đậm (e.g., **X**, **W**, **Ŷ**) |
| Batch | xếp chồng các $\mathbf{x}_n^T$ thành các **hàng** |

**Kích thước của các tensor:**

- **Input:** $\mathbf{X} \in \mathbb{R}^{B \times D}$ — hàng thứ $n$ là $\mathbf{x}_n^T$; shape `(B, D)`
  - $B$: **batch size** — số mẫu trong một lần xử lý
  - $D$: **input dimension** — số chiều đặc trưng đầu vào
  - $K$: **output dimension** — số chiều đầu ra

- **Labels (nhãn thực tế):** $\mathbf{Y} \in \mathbb{R}^{B \times K}$

- **Learnable Parameters (tham số học được):**
  - $\mathbf{W} \in \mathbb{R}^{D \times K}$; shape `(D, K)` — ma trận trọng số
  - $\mathbf{b} \in \mathbb{R}^{K}$; shape `(K,)` — vector bias

- **Model Output (đầu ra mô hình):** $\hat{\mathbf{Y}} \in \mathbb{R}^{B \times K}$
  - $\hat{Y}_{n,k}$: dự đoán cho mẫu $n$, đầu ra thứ $k$

> **Tại sao cần phân biệt rõ shape?** Trong deep learning, một sai lầm về shape có thể khiến mô hình chạy không lỗi nhưng cho kết quả sai hoàn toàn. Hiểu shape là kỹ năng sinh tồn.

---

### Fully-Connected Layer (FC Layer)

> 📷 **[Ảnh slide 5]** Sơ đồ mạng nơ-ron với các kết nối $w_{ij}$, bias $b_j$, tổng $\Sigma$ ở mỗi nút đầu ra.
> Nguồn: slide 5, `03-RegressionSummary.pdf`

**FC** là viết tắt của **Fully-Connected** (hay còn gọi là **Dense**). Đây là loại lớp cơ bản nhất trong mạng nơ-ron.

**Nguyên lý hoạt động:** Mỗi nơ-ron đầu ra $y_j$ nhận tín hiệu từ **tất cả** các nơ-ron đầu vào $x_1, x_2, \ldots, x_{N_{in}}$, mỗi kết nối có trọng số riêng $w_{ij}$, rồi cộng thêm bias $b_j$:

$$y_j = \sum_{i=1}^{N_{in}} w_{ij} \cdot x_i + b_j$$

Viết gọn cho cả vector:

$$\mathbf{y} = \mathbf{W}^T \mathbf{x} + \mathbf{b}$$

Và cho cả batch:

$$\hat{\mathbf{Y}} = \mathbf{X}\mathbf{W} + \mathbf{1}\mathbf{b}^T$$

> **Chú ý về transpose:** Khi xét từng mẫu đơn lẻ, ta viết $\mathbf{W}^T\mathbf{x}$. Khi xét batch, **X** đã là dạng hàng nên ta viết $\mathbf{X}\mathbf{W}$. Đây là điểm dễ nhầm lẫn nhất.

**Tương đương trong các thư viện:**

| Framework | API |
|-----------|-----|
| PyTorch | `torch.nn.Linear(in_features, out_features)` |
| Keras/TF | `keras.layers.Dense(units)` |

---

## 2. Mathematical View

### Công thức toán học

**Linear Regression** thực chất là phép tính:

**Cho một mẫu đơn** $\mathbf{x} \in \mathbb{R}^D$:

$$\hat{\mathbf{y}} = \mathbf{W}^T\mathbf{x} + \mathbf{b}$$

**Cho cả batch** $\mathbf{X} \in \mathbb{R}^{B \times D}$ (hàng $n$ là $\mathbf{x}_n^T$):

$$\hat{\mathbf{Y}} = \mathbf{X}\mathbf{W} + \mathbf{1}\mathbf{b}^T$$

> **Câu nói đáng nhớ:** *"Linear Regression is a one-layer neural network without activation."*
>
> Đây là điểm kết nối quan trọng: Linear Regression không phải thứ gì quá xa lạ — nó chính là mạng nơ-ron đơn giản nhất, chỉ gồm một lớp FC mà không có hàm kích hoạt phi tuyến. Khi bạn thêm các lớp và hàm kích hoạt, bạn tiến dần đến Deep Learning.

---

## 3. Programming View

### PyTorch

```python
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, X):  # X: (N, D)
        return self.fc(X)

# Ví dụ: với D chiều đầu vào và K chiều đầu ra
model = LinearModel(input_dim=D, output_dim=K)
```

**Giải thích:**
- `nn.Module`: lớp cơ sở của mọi mô hình trong PyTorch.
- `nn.Linear(input_dim, output_dim)`: tạo một lớp FC, tự động khởi tạo **W** và **b**.
- `forward(X)`: định nghĩa luồng tính toán — khi gọi `model(X)`, PyTorch sẽ gọi hàm này.

### Keras (TensorFlow)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

class LinearModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = layers.Dense(output_dim)

    def call(self, X):  # X: (N, D)
        return self.fc(X)

# Ví dụ
model = LinearModel(input_dim=D, output_dim=K)
# Build the model (optional nhưng nên có)
model.build(input_shape=(None, D))
```

**So sánh PyTorch vs Keras:**
- PyTorch dùng `forward()`, Keras dùng `call()`.
- Keras `Dense` không yêu cầu `input_dim` trong constructor (lazy initialization).
- `model.build()` trong Keras giúp khởi tạo tham số và cho phép xem `model.summary()`.

---

## 4. Training Methods

### Tổng quan — Hai cách huấn luyện

Linear Regression là một trong **số ít mô hình** có thể được huấn luyện theo cả hai cách:

| Phương pháp | Cách hoạt động | Đặc điểm |
|-------------|---------------|-----------|
| **Closed-form (Analytical)** | Giải trực tiếp bằng toán học | Chính xác, không cần lặp, nhưng không mở rộng được |
| **Gradient Descent (Iterative)** | Tối ưu hóa lặp đi lặp lại | Tổng quát cho mọi mô hình deep learning |

---

### 4.1 Closed-form Solution (Nghiệm dạng đóng)

#### Augmented Formulation — Gộp bias vào W

Để đơn giản hóa ký hiệu, ta **gộp bias** vào ma trận trọng số bằng cách mở rộng dữ liệu:

$$\tilde{\mathbf{X}} = [\mathbf{X} \; \mathbf{1}], \quad \tilde{\mathbf{W}} = \begin{bmatrix}\mathbf{W} \\ \mathbf{b}^T\end{bmatrix}$$

Lúc này $\hat{\mathbf{Y}} = \tilde{\mathbf{X}}\tilde{\mathbf{W}}$ — gọn hơn!

#### Hàm mục tiêu (Objective Function)

Ta muốn tìm **W**, **b** sao cho tổng bình phương sai số nhỏ nhất:

$$\arg\min_{\mathbf{W}, \mathbf{b}} \|\hat{\mathbf{Y}} - \mathbf{Y}\|_F^2$$

Trong đó $\|\cdot\|_F$ là **Frobenius norm** — căn bậc hai của tổng bình phương tất cả phần tử, tức là tổng bình phương sai số trên toàn bộ batch và tất cả đầu ra.

> **Tại sao dùng bình phương?** Vì bình phương phạt nặng các sai số lớn hơn, tạo ra hàm mục tiêu trơn và có đạo hàm liên tục — dễ tối ưu hóa về mặt toán học.

Tương đương với bài toán:

$$\arg\min_{\tilde{\mathbf{W}}} \|\tilde{\mathbf{X}}\tilde{\mathbf{W}} - \mathbf{Y}\|_F^2$$

#### Nghiệm Closed-form (Ordinary Least Squares — OLS)

Đặt đạo hàm bằng 0 và giải ra, ta có:

$$\tilde{\mathbf{W}} = (\tilde{\mathbf{X}}^T\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^T\mathbf{Y}$$

Đây là **công thức nghiệm OLS** nổi tiếng. Vế phải $(\tilde{\mathbf{X}}^T\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^T$ còn được gọi là **Moore-Penrose pseudoinverse** của $\tilde{\mathbf{X}}$.

#### Đặc điểm của OLS

- Yêu cầu **nghịch đảo ma trận** — đây là bước đắt nhất.
- **Chi phí tính toán:** $\mathcal{O}(D^3)$ — nếu $D$ tăng gấp đôi, thời gian tăng **8 lần**.
- **Không mở rộng được** (not scalable) khi $D$ lớn (ví dụ ảnh 224×224×3 có $D \approx 150{,}000$).

> **Ví dụ thực tế:** Với dữ liệu nhà đất chỉ có 10 đặc trưng ($D=10$), OLS cho nghiệm tức thì. Nhưng với ảnh 28×28 pixel (MNIST, $D=784$), OLS vẫn khả thi. Tuy nhiên với ảnh ImageNet ($D \approx 150K$), OLS hoàn toàn bất khả thi.

---

### 4.2 Closed-form với Regularization — Ridge Regression

Một vấn đề với OLS thuần túy: ma trận $\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$ có thể **không khả nghịch** (singular) khi:
- Dữ liệu có ít mẫu hơn đặc trưng ($B < D$)
- Các đặc trưng có tương quan cao với nhau (multicollinearity)

**Ridge Regression** thêm vào hàm mục tiêu một số hạng phạt **L2**:

$$\tilde{\mathbf{W}}^* = \arg\min_{\tilde{\mathbf{W}}} \left(\|\tilde{\mathbf{X}}\tilde{\mathbf{W}} - \mathbf{Y}\|_F^2 + \lambda\|\mathbf{W}\|_F^2\right)$$

Nghiệm dạng đóng trở thành:

$$\tilde{\mathbf{W}} = \left(\tilde{\mathbf{X}}^T\tilde{\mathbf{X}} + \lambda\begin{bmatrix}\mathbf{I}_D & \mathbf{0} \\ \mathbf{0}^T & 0\end{bmatrix}\right)^{-1}\tilde{\mathbf{X}}^T\mathbf{Y}$$

> **Tại sao $\lambda > 0$ giải quyết vấn đề?** Thêm $\lambda\mathbf{I}$ vào đường chéo đảm bảo ma trận luôn **positive definite** và do đó **luôn khả nghịch**. Hãy nghĩ đến nó như việc "ổn định" ma trận bằng cách thêm một chút nhiễu có kiểm soát.

**Đặc điểm Ridge Regression:**
- ✅ Cải thiện **numerical stability** (tính ổn định số học)
- ✅ Kiểm soát **overfitting** — W không bị quá lớn
- ✅ Luôn **khả nghịch** khi $\lambda > 0$
- ❌ Vẫn cần nghịch đảo ma trận — không giải quyết vấn đề scalability

---

### 4.3 Gradient Descent (Tối ưu hóa lặp)

> 📷 **[Ảnh slide 18]** Sơ đồ training: `X → [Neural Network Model (W,b)] → Ŷ → [Loss Function] → ℓ`, với Y cũng là đầu vào Loss Function.
> Nguồn: slide 18, `03-RegressionSummary.pdf`

#### Ý tưởng cốt lõi

Thay vì tính toán ngay lập tức, **Gradient Descent** tiếp cận bài toán như một cuộc leo núi ngược: ta muốn đi xuống "thung lũng" của hàm loss bằng cách liên tục bước theo hướng **dốc nhất đi xuống**.

Bài toán tối ưu:

$$(W^*, b^*) = \arg\min_{\mathbf{W}, \mathbf{b}} \mathcal{L}(\mathbf{W}, \mathbf{b})$$

Với **squared loss** (mean squared error):

$$\mathcal{L}(\mathbf{W}, \mathbf{b}) = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_F^2$$

#### Quy tắc cập nhật

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla_{\mathbf{W}}\mathcal{L}$$
$$\mathbf{b}_{t+1} = \mathbf{b}_t - \eta \nabla_{\mathbf{b}}\mathcal{L}$$

#### Giải thích các ký hiệu

| Ký hiệu | Ý nghĩa | Vai trò |
|---------|---------|---------|
| $\mathcal{L}(\mathbf{W}, \mathbf{b})$ | **Loss function** | Đo lường sai số dự đoán — ta muốn tối thiểu hóa cái này |
| $\nabla_{\mathbf{W}}\mathcal{L}$, $\nabla_{\mathbf{b}}\mathcal{L}$ | **Gradients** | Hướng tăng dốc nhất của loss — ta đi **ngược** chiều này |
| $\eta$ (eta) | **Learning rate** | Bước nhảy mỗi lần cập nhật — quá lớn thì vọt qua, quá nhỏ thì hội tụ chậm |

> **Phép ẩn dụ:** Hãy tưởng tượng bạn đang đứng trên một ngọn đồi trong màn sương (không thấy toàn cảnh). Gradient cho bạn biết hướng nào là dốc nhất đi lên. Bạn bước theo **hướng ngược lại** (gradient descent) để đi xuống thung lũng. Learning rate là độ dài mỗi bước.

#### Đặc điểm của Gradient Descent

- ✅ **Không cần nghịch đảo ma trận**
- ✅ **Mở rộng được** cho dataset lớn (dùng mini-batch)
- ✅ **Áp dụng được** cho mọi kiến trúc mạng nơ-ron
- ❌ Cần lặp nhiều lần — không cho nghiệm chính xác tức thì

> **Câu nói đáng nhớ:** *"Gradient Descent is the foundation of deep learning."*
>
> Mọi mô hình deep learning đều được huấn luyện bằng Gradient Descent (hoặc các biến thể như Adam, SGD, RMSProp). Hiểu Gradient Descent là hiểu trái tim của deep learning.

---

## Tóm tắt so sánh hai phương pháp

| | Closed-form (OLS) | Gradient Descent |
|---|---|---|
| **Cách hoạt động** | Giải toán học trực tiếp | Cập nhật lặp đi lặp lại |
| **Tốc độ** | Tức thì (nhưng chậm nếu D lớn) | Cần nhiều epoch |
| **Scalability** | ❌ Không mở rộng được ($\mathcal{O}(D^3)$) | ✅ Mở rộng tốt |
| **Áp dụng cho** | Chỉ Linear Regression | Mọi mô hình neural network |
| **Độ chính xác** | Nghiệm tối ưu toàn cục (exact) | Xấp xỉ (phụ thuộc vào $\eta$, số epoch) |
| **Khi nào dùng** | D nhỏ, dataset vừa | D lớn, deep learning |
