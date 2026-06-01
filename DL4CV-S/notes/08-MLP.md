# Bài 08 — Multilayer Perceptron (MLP)

> **Nguồn slide:** `slides-v1/foundation/08-MLP.pdf` — Thanh-Sach LE, HCMUT, VNU-HCM (05/02/2026)

---

## Mục lục

1. [Recap: Tại sao cần MLP?](#1-recap-tại-sao-cần-mlp)
2. [MLP: Góc nhìn Kiến trúc](#2-mlp-góc-nhìn-kiến-trúc)
3. [MLP: Mô hình Toán học](#3-mlp-mô-hình-toán-học)
4. [MLP: Các thành phần chi tiết](#4-mlp-các-thành-phần-chi-tiết)
   - [4.1 Fully Connected Layer](#41-fully-connected-fc-layer)
   - [4.2 Activation Functions](#42-activation-functions-hàm-kích-hoạt)
   - [4.3 Batch Normalization](#43-batch-normalization)
5. [Tổng kết MLP](#5-tổng-kết-mlp)
6. [Bài Tập Tính Toán](#6-bài-tập-tính-toán)

---

## Giới thiệu

Ở bài trước, chúng ta đã thấy rằng **Linear/Logistic/Softmax Regression đều bị giới hạn bởi tính tuyến tính** — chúng không thể học các quan hệ phi tuyến trong dữ liệu. Giải pháp là thêm một **bộ biến đổi đặc trưng phi tuyến (nonlinear feature transformer)** trước khi áp mô hình tuyến tính.

**Multilayer Perceptron (MLP)** chính là kiến trúc đơn giản nhất hiện thực hóa ý tưởng đó. MLP là nền tảng của mọi mạng neural hiện đại — hiểu MLP sâu sắc là bước đệm để hiểu CNN, Transformer, và các kiến trúc phức tạp hơn.

---

## 1. Recap: Tại sao cần MLP?

### 1.1 Linear Regression — Giới hạn tuyến tính

**Một đầu ra:**

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

**Nhiều đầu ra ($m$ targets):**

$$\hat{\mathbf{y}} = W^\top \mathbf{x} + \mathbf{b}, \qquad W \in \mathbb{R}^{d \times m},\ \mathbf{b} \in \mathbb{R}^m$$

Dù có một hay nhiều đầu ra, mô hình vẫn là **ánh xạ tuyến tính** từ $\mathbf{x}$ — không thể biểu diễn bất kỳ quan hệ phi tuyến nào.

### 1.2 Logistic Regression và Softmax — Vẫn tuyến tính

| Mô hình | Decision Boundary | Giới hạn |
|---------|------------------|---------|
| Logistic | $\mathbf{w}^\top \mathbf{x} + b = 0$ (hyperplane) | Chỉ phân tách tuyến tính |
| Softmax | $(\mathbf{w}_k - \mathbf{w}_j)^\top \mathbf{x} + (b_k - b_j) = 0$ | Vùng lớp là đa diện lồi |

**Cả hai đều tuyến tính trong không gian đặc trưng đầu vào.** Chúng không thể giải quyết bài toán XOR, không thể phân tách các lớp theo vòng tròn lồng nhau, v.v.

### 1.3 Ý tưởng của MLP

> **MLP idea:** Thêm các **hidden layers** đóng vai trò là bộ biến đổi đặc trưng phi tuyến.
>
> - Sau bước biến đổi, với bài toán **hồi quy**: đầu ra có thể được mô hình bởi hàm tuyến tính trong không gian đặc trưng mới.
> - Với bài toán **phân loại**: các lớp trở nên gần như tách được tuyến tính trong không gian mới.
>
> **Bản chất:** Deep Learning = học đặc trưng phi tuyến tốt + áp một linear head đơn giản.

---

## 2. MLP: Góc nhìn Kiến trúc

### 2.1 Pipeline tổng quát

MLP có hai phần chính:

```
x ──→ [Feature Transformer] ──→ [Output Head] ──→ ŷ hoặc p̂
```

**Cho bài toán hồi quy:**

```
x → Feature Transformer (FC layers + Activations) → Regression Head (Linear) → ŷ
```

**Cho bài toán phân loại:**

```
x → Feature Transformer (FC layers + Activations) → Classification Head (Logistic/Softmax) → p̂ = P(y|x)
```

> **Nhận xét quan trọng:** Output head luôn **đơn giản (tuyến tính)**. Toàn bộ sức mạnh phi tuyến nằm ở Feature Transformer.

### 2.2 Bên trong Feature Transformer

Feature Transformer là một chuỗi các cặp **[FC + Activation]** xếp chồng nhau:

```
x → [FC → A] → [FC → A] → ··· → [FC → A] → h
```

- **FC (Fully Connected):** Biến đổi tuyến tính
- **A (Activation):** Hàm kích hoạt phi tuyến

> **Trường hợp đặc biệt — 0 hidden layers:**
>
> Nếu Feature Transformer rỗng (không có cặp FC + A nào), MLP **thoái hóa** thành:
> - **Linear Regression** (nếu dùng regression head)
> - **Logistic Regression** (nếu dùng binary classification head)
> - **Softmax Regression** (nếu dùng multiclass head)
>
> Điều này cho thấy MLP là sự **tổng quát hóa** của tất cả các mô hình tuyến tính cổ điển.

> **Tại sao cần cả FC lẫn Activation?**
> - Chỉ có FC (không có Activation): xếp chồng bao nhiêu lớp FC cũng chỉ tương đương **một lớp FC** (vì tích của các ma trận tuyến tính vẫn là tuyến tính).
> - Chỉ có Activation (không có FC): không học được gì có ý nghĩa.
> - **FC + Activation xen kẽ** = mới tạo ra biểu diễn phi tuyến thực sự.

> 📸 **[Cần ảnh]:** Sơ đồ pipeline: box "x" → box "Feature Transformer" (bên trong ghi FC→A→FC→A→...) → box "Head" → "ŷ/p̂". *(Trang 11–13 slide)*

---

## 3. MLP: Mô hình Toán học

### 3.1 Forward Pass — Lan truyền thuận

Kí hiệu: $L$ hidden layers, mỗi lớp $l$ có kích thước (số neuron) $d_l$.

**Khởi tạo:**
$$\mathbf{h}^{(0)} = \mathbf{x} \qquad \text{(input)}$$

**Mỗi hidden layer $l = 1, \ldots, L$:**

$$\mathbf{h}^{(l)} = \phi\!\left(W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

Trong đó:
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$: ma trận trọng số lớp $l$
- $\mathbf{b}^{(l)} \in \mathbb{R}^{d_l}$: vector bias lớp $l$
- $\phi(\cdot)$: hàm kích hoạt phi tuyến (ReLU, Sigmoid, Tanh, ...)

**Output layer:**

- *Hồi quy:* $\hat{y} = W^{(L+1)} \mathbf{h}^{(L)} + b^{(L+1)}$
- *Phân loại đa lớp:* $\hat{p}_k = \dfrac{\exp(\mathbf{w}_k^\top \mathbf{h}^{(L)} + b_k)}{\sum_j \exp(\mathbf{w}_j^\top \mathbf{h}^{(L)} + b_j)}$

> **Diễn giải:** Mỗi lớp ẩn thực hiện một phép biến đổi **tuyến tính → phi tuyến**. Việc xen kẽ hai phép biến đổi này liên tục qua nhiều lớp cho phép mạng xây dựng các biểu diễn **ngày càng trừu tượng và phức tạp hơn**.

### 3.2 MLP như Hợp thành Hàm số

Có thể viết gọn toàn bộ MLP dưới dạng **hợp thành hàm**:

$$f(\mathbf{x}; \theta) = f^{(L+1)} \circ \phi \circ f^{(L)} \circ \cdots \circ \phi \circ f^{(1)}(\mathbf{x})$$

Trong đó $f^{(l)}(\mathbf{u}) = W^{(l)}\mathbf{u} + \mathbf{b}^{(l)}$ là phép biến đổi tuyến tính tại lớp $l$.

**Thêm lớp → tăng khả năng biểu diễn (representation power).** Mạng sâu hơn có thể mô hình hóa các hàm phức tạp hơn với ít neuron hơn mạng nông.

> **Universal Approximation Theorem (định lý xấp xỉ toàn cục):** Một MLP với **một hidden layer đủ rộng** và hàm kích hoạt phi tuyến có thể xấp xỉ bất kỳ hàm liên tục nào trên miền compact với độ chính xác tùy ý. Đây là cơ sở lý thuyết cho sức mạnh của MLP — nhưng "đủ rộng" trong thực tế có thể cần số neuron cực lớn, nên mạng **sâu (deep)** thường hiệu quả hơn mạng **nông mà rộng (wide)**.

---

## 4. MLP: Các thành phần chi tiết

### 4.1 Fully Connected (FC) Layer

#### Toán học

**Một mẫu:**

$$\mathbf{y} = W\mathbf{x} + \mathbf{b}, \qquad W \in \mathbb{R}^{M \times N},\ \mathbf{x} \in \mathbb{R}^N,\ \mathbf{y} \in \mathbb{R}^M$$

**Mini-batch** (xử lý $B$ mẫu cùng lúc):

$$Y = XW^\top + \mathbf{1}\mathbf{b}^\top, \qquad X \in \mathbb{R}^{B \times N},\ Y \in \mathbb{R}^{B \times M}$$

Trong đó $\mathbf{1} \in \mathbb{R}^B$ là vector toàn 1 — cơ chế **broadcasting** thêm bias vào mỗi hàng.

**Số tham số:** $M \times N$ (weights) $+ M$ (biases) $= M(N+1)$

> **Lưu ý quan trọng:**
> - FC thuần túy là **tuyến tính** — không có phi tuyến.
> - Xếp chồng nhiều FC **mà không có activation** giữa chúng tương đương với **một FC duy nhất** (vì $W_2 W_1 \mathbf{x} = (W_2 W_1)\mathbf{x}$). Đây là lý do bắt buộc phải có activation giữa các lớp.
> - Bias có thể được hấp thụ vào weight bằng cách mở rộng input: $\tilde{\mathbf{x}} = [\mathbf{x}; 1]$, $\tilde{W} = [W\ \mathbf{b}]$.

#### Sơ đồ (N=4, M=3)

```
x₁ ──┐
x₂ ──┤──── [W ∈ R³ˣ⁴, b ∈ R³] ────→ y₁
x₃ ──┤                               y₂
x₄ ──┘                               y₃
1  ──┘ (bias)
```

Mỗi neuron đầu ra $y_j = \sum_{i=1}^{4} W_{j,i} x_i + b_j$ — kết nối đến **tất cả** neuron đầu vào (vì vậy gọi là "Fully Connected").

> 📸 **[Cần ảnh]:** Sơ đồ mạng neural với 4 nút input, 3 nút output, tất cả kết nối với nhau, kèm nhãn $W_{j,i}$. *(Trang 22 slide)*

#### Code

```python
# NumPy
import numpy as np
y = W @ x + b          # single sample: (M,)
Y = X @ W.T + b        # mini-batch: (B, M) — broadcasting

# PyTorch
import torch
import torch.nn as nn

fc = nn.Linear(in_features=128, out_features=64, bias=True)
x = torch.randn(32, 128)   # batch of 32 samples
y = fc(x)                  # shape: (32, 64)

# Tương đương thủ công:
y2 = x @ fc.weight.T + fc.bias

# Keras (TensorFlow)
from tensorflow.keras import layers, models, Input
inp = Input(shape=(128,))
out = layers.Dense(64, use_bias=True)(inp)
```

---

### 4.2 Activation Functions (Hàm kích hoạt)

Hàm kích hoạt là **linh hồn** của MLP — nếu không có chúng, toàn bộ mạng chỉ là một phép biến đổi tuyến tính.

#### Bảng tổng quan

| Activation | Dùng ở đâu | Ưu điểm | Hạn chế |
|-----------|-----------|---------|---------|
| **Sigmoid** | Output (xác suất nhị phân) | Range (0,1), diễn giải xác suất | Vanishing gradient; không zero-centered |
| **Tanh** | Hidden (legacy) | Zero-centered; range (−1,1) | Vẫn bị saturate |
| **ReLU** | Hidden (mặc định) | Đơn giản, nhanh; giảm vanishing gradient | "Dead neurons" |
| **Leaky ReLU** | Hidden | Tránh dead neurons | Thêm hyperparameter $\alpha$ |
| **SiLU/Swish** | Hidden (modern) | Mượt, không đơn điệu; hiệu suất thực nghiệm tốt | Tính toán đắt hơn ReLU |
| **Softmax** | Output (đa lớp) | Chuẩn hóa thành phân phối xác suất | Chỉ dùng ở output |

**Quy tắc ngón tay cái:**
- **Hidden layers:** ReLU hoặc biến thể (Leaky ReLU, SiLU)
- **Output regression:** Identity (không activation) hoặc Sigmoid (nếu cần bounded)
- **Output classification:** Sigmoid (binary) hoặc Softmax (multiclass)

---

#### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Tính chất:**
- Range: $(0, 1)$ — diễn giải tự nhiên là xác suất
- **Vấn đề vanishing gradient:** Khi $|z|$ lớn, $\sigma'(z) \approx 0$. Gradient lan truyền ngược qua nhiều lớp Sigmoid sẽ nhân với những con số gần 0 → gradient biến mất, mạng không học được.
- **Không zero-centered:** Output luôn dương → gradient luôn cùng dấu → zigzag khi update.

> **Phép ẩn dụ:** Sigmoid giống như một công tắc điều chỉnh từ từ — nhưng khi đã bật hoàn toàn hay tắt hoàn toàn, nó ngừng phản ứng với tín hiệu.

> 📸 **[Cần ảnh]:** Đồ thị $\sigma(z)$ và $\sigma'(z)$ — thấy rõ đạo hàm "bẹt" ở hai đầu. *(Trang 24 slide)*

---

#### Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \qquad \frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$$

**Tính chất:**
- Range: $(-1, 1)$ — **zero-centered** (tốt hơn Sigmoid)
- Đạo hàm tối đa là 1 (tại $z=0$), vẫn bị saturate khi $|z|$ lớn
- Là phiên bản "co giãn" của Sigmoid: $\tanh(z) = 2\sigma(2z) - 1$

> 📸 **[Cần ảnh]:** Đồ thị $\tanh(z)$ và đạo hàm — tương tự Sigmoid nhưng đối xứng qua gốc tọa độ. *(Trang 25 slide)*

---

#### ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z), \qquad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$$

**Tại sao ReLU trở thành mặc định?**

1. **Đơn giản và nhanh:** Chỉ là một phép so sánh với 0 — rất nhanh trên GPU.
2. **Giảm vanishing gradient:** Vùng $z > 0$ có gradient = 1, không bị suy giảm qua nhiều lớp.
3. **Sparse activation:** Nhiều neuron output = 0, tạo ra biểu diễn thưa (sparse) — được cho là có lợi cho khả năng tổng quát hóa.

**Vấn đề "Dead Neurons":** Khi $z < 0$, gradient = 0. Nếu tổng input của một neuron luôn âm (do trọng số không may), neuron đó sẽ **ngừng học vĩnh viễn** — gọi là "dead neuron". Có thể xảy ra khi learning rate quá lớn.

> **Phép ẩn dụ:** ReLU giống như một cửa van — chỉ cho tín hiệu dương đi qua, chặn hoàn toàn tín hiệu âm.

> 📸 **[Cần ảnh]:** Đồ thị ReLU — hình dạng "gậy khúc khuỷu" (hinge), đạo hàm là step function. *(Trang 26 slide)*

---

#### Leaky ReLU

$$\text{LReLU}(z) = \begin{cases} z & z \geq 0 \\ \alpha z & z < 0 \end{cases}, \qquad \text{LReLU}'(z) = \begin{cases} 1 & z > 0 \\ \alpha & z < 0 \end{cases}$$

Với $\alpha \in (0, 1)$, thường $\alpha = 0.01$.

**Giải pháp cho dead neurons:** Vùng âm không còn gradient = 0 mà là $\alpha > 0$ — neuron vẫn nhận được tín hiệu, dù nhỏ, và có thể hồi phục.

**Khi $\alpha \to 0$:** Leaky ReLU thoái hóa về ReLU.

> 📸 **[Cần ảnh]:** Đồ thị Leaky ReLU — giống ReLU nhưng có độ dốc nhỏ $\alpha$ ở vùng âm thay vì phẳng. *(Trang 27 slide)*

---

#### SiLU / Swish

$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

$$\frac{d}{dz}\text{SiLU}(z) = \sigma(z) + z \cdot \sigma(z)(1 - \sigma(z))$$

**Tính chất:**
- **Mượt (smooth):** Khác với ReLU có điểm không khả vi tại $z=0$, SiLU mượt hoàn toàn.
- **Không đơn điệu (non-monotonic):** Có vùng $z$ âm nhỏ nơi SiLU âm — tạo ra một loại "cổng" tự điều chỉnh (self-gating).
- **Hiệu suất thực nghiệm mạnh:** Được dùng trong nhiều kiến trúc hiện đại (EfficientNet, nhiều biến thể Transformer).

> **So sánh trực quan:**
> - **ReLU:** Cứng, hiệu quả, đơn giản — như con dao Swiss Army
> - **SiLU:** Mượt, có thể "mờ dần" với giá trị âm — như bộ điều chỉnh analog

> 📸 **[Cần ảnh]:** Đồ thị SiLU — thấy rõ vùng âm nhỏ và độ mượt so với ReLU. *(Trang 28 slide)*

---

### 4.3 Batch Normalization

#### Vấn đề: Internal Covariate Shift

Khi huấn luyện mạng sâu, phân phối đầu vào của mỗi lớp thay đổi liên tục sau mỗi bước cập nhật trọng số — hiện tượng này gọi là **internal covariate shift**. Điều này khiến việc huấn luyện không ổn định và phụ thuộc nhiều vào việc khởi tạo trọng số.

#### Giải pháp: Batch Normalization (BatchNorm)

Chuẩn hóa mỗi đặc trưng **theo chiều batch dimension**:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta$$

Trong đó:
- $\mu_B, \sigma_B$: trung bình và độ lệch chuẩn của batch hiện tại
- $\epsilon$: hằng số nhỏ tránh chia cho 0 (thường $10^{-5}$)
- $\gamma, \beta$: các tham số **học được** (scale và shift) — cho phép mạng "tự chọn" phân phối phù hợp thay vì bị ép về phân phối chuẩn

**Vị trí trong kiến trúc:** Thường đặt **sau FC (hoặc Conv) và trước Activation**:

```
FC → BatchNorm → Activation → (lớp tiếp theo)
```

**Lợi ích:**
- Ổn định huấn luyện — cho phép dùng learning rate lớn hơn
- Giảm phụ thuộc vào khởi tạo trọng số
- Có tác dụng regularization nhẹ (do nhiễu từ batch statistics)

#### LayerNorm — biến thể cho Transformer và RNN

$$\hat{x}_i = \frac{x_i - \mu_{\text{sample}}}{\sqrt{\sigma_{\text{sample}}^2 + \epsilon}}$$

**LayerNorm** chuẩn hóa theo chiều **feature dimension** cho từng mẫu riêng lẻ (thay vì theo batch). Phù hợp hơn với Transformer và RNN vì các mô hình này xử lý chuỗi với độ dài biến đổi — batch statistics không ổn định.

| | BatchNorm | LayerNorm |
|--|-----------|-----------|
| Chuẩn hóa theo | Batch dimension | Feature dimension |
| Thường dùng trong | CNN, MLP | Transformer, RNN |
| Phụ thuộc batch size | Có | Không |

---

## 5. Tổng kết MLP

### 5.1 Tóm tắt kiến trúc

```
Input x
  │
  ▼
[FC] → [BatchNorm] → [Activation]   ← Layer 1 (hidden)
  │
  ▼
[FC] → [BatchNorm] → [Activation]   ← Layer 2 (hidden)
  │
  ▼
    ...
  │
  ▼
[FC]                                 ← Output layer (linear head)
  │
  ▼
ŷ (regression) hoặc Softmax → p̂ (classification)
```

### 5.2 MLP vs Mô hình Tuyến tính

| Đặc điểm | Linear/Logistic/Softmax | MLP |
|----------|------------------------|-----|
| Hypothesis class | Hàm tuyến tính | Hàm phi tuyến (Universal Approximator) |
| Decision boundary | Hyperplane | Có thể là bất kỳ đường cong nào |
| Tham số | Ít | Nhiều hơn (tùy depth/width) |
| Nguy cơ overfitting | Thấp | Cao hơn → cần regularization |
| Yêu cầu dữ liệu | Ít | Nhiều hơn |

### 5.3 Universal Approximation

> **Định lý Xấp xỉ Toàn cục (Universal Approximation Theorem):**
>
> Một MLP với **một hidden layer đủ rộng** và hàm kích hoạt phi tuyến liên tục có thể xấp xỉ **bất kỳ hàm liên tục nào** trên miền compact với độ chính xác tùy ý.

Đây là đảm bảo lý thuyết rằng MLP có đủ sức mạnh biểu diễn. Tuy nhiên, trong thực tế:
- Mạng **sâu (deep)** thường hiệu quả hơn mạng **nông mà rộng (shallow & wide)** cho cùng một số tham số.
- Mạng sâu tận dụng được **hierarchical feature learning** — mỗi lớp xây dựng trên biểu diễn của lớp trước.

### 5.4 Hạn chế và hướng tới

**MLP có những hạn chế thực tế:**
- Với ảnh: mỗi pixel kết nối đến mỗi neuron → số tham số khổng lồ → dễ overfit, không tận dụng cấu trúc không gian.
- Với chuỗi thời gian / text: không có cơ chế tự nhiên để xử lý thứ tự và độ dài biến đổi.

**Các kiến trúc tiếp theo giải quyết những hạn chế này:**
- **CNN:** Khai thác cấu trúc không gian của ảnh (spatial locality + translation invariance)
- **RNN/LSTM:** Xử lý chuỗi thứ tự
- **Transformer:** Cơ chế attention — mọi vị trí đều tương tác với nhau

> **Bức tranh thống nhất:**
>
> $$\boxed{\text{Modern DL} = \underbrace{\Phi(x)}_{\text{Kiến trúc chuyên biệt}} + \underbrace{\text{Linear Head}}_{\text{FC + Loss}}}$$
>
> MLP là trường hợp đơn giản nhất của $\Phi$ — dùng FC + Activation. CNN/Transformer là các $\Phi$ phức tạp hơn được thiết kế cho từng loại dữ liệu đặc thù.

---

## 6. Bài Tập Tính Toán

> Tự làm trước khi mở đáp án.

---

### Bài 1 — Đếm tham số FC Layer và MLP

Một MLP phân loại ảnh MNIST (input 28×28 = 784 chiều, 10 lớp) có kiến trúc:

```
Input (784) → FC1 (256) → ReLU → FC2 (128) → ReLU → FC3 (10) → Softmax
```

**(a)** Tính số tham số (weights + biases) của từng FC layer.

**(b)** Tổng số tham số toàn bộ mô hình.

**(c)** Nếu thay input ảnh 28×28 bằng ảnh 32×32 (1024 chiều), tổng tham số thay đổi bao nhiêu?

**(d)** Nếu dùng FC layer đầu tiên size 1024 thay vì 256, tổng tham số thay đổi như thế nào? Rút ra nhận xét về "bottleneck" kiến trúc.

<details>
<summary>📋 Đáp án Bài 1</summary>

Công thức: $\text{Params}_{FC} = M \times N + M = M(N+1)$ với $N$ = chiều vào, $M$ = chiều ra.

**(a) Tham số từng layer:**

| Layer | N (vào) | M (ra) | Weights | Biases | Tổng |
|-------|---------|--------|---------|--------|------|
| FC1 | 784 | 256 | $256 \times 784 = 200{,}704$ | $256$ | **200,960** |
| FC2 | 256 | 128 | $128 \times 256 = 32{,}768$ | $128$ | **32,896** |
| FC3 | 128 | 10 | $10 \times 128 = 1{,}280$ | $10$ | **1,290** |

**(b) Tổng:**
$$200{,}960 + 32{,}896 + 1{,}290 = \mathbf{235{,}146}$$

**(c) Thay 784 → 1024 (32×32):**

FC1 tham số mới: $256 \times (1024 + 1) = 256 \times 1025 = 262{,}400$

Tăng thêm: $262{,}400 - 200{,}960 = 61{,}440$

Tổng mới: $235{,}146 + 61{,}440 = \mathbf{296{,}586}$

**(d) Thay FC1 size 256 → 1024:**

| Layer | Tham số mới |
|-------|------------|
| FC1 (784→1024) | $1024 \times 785 = 803{,}840$ |
| FC2 (1024→128) | $128 \times 1025 = 131{,}200$ |
| FC3 (128→10) | $1{,}290$ |
| **Tổng** | **936,330** |

Tăng từ 235,146 → 936,330 — gấp **4 lần** chỉ vì tăng một layer!

**Nhận xét:** FC1 là bottleneck tham số — kết nối input lớn (784) với hidden layer lớn (1024). Lớp đầu tiên thường chiếm phần lớn tham số. Vì vậy nhiều kiến trúc thực tế **giảm kích thước ngay từ layer 1** (encoder pattern) hoặc dùng Conv thay FC cho ảnh.

</details>

---

### Bài 2 — Forward pass qua FC Layer (tính tay)

Cho FC layer với:

$$W = \begin{bmatrix} 1 & -1 & 2 \\ 0 & 3 & -1 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$$

Input: $\mathbf{x} = \begin{bmatrix} 2 \\ 1 \\ -1 \end{bmatrix}$

**(a)** Tính $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ (pre-activation).

**(b)** Tính output sau ReLU: $\mathbf{h} = \text{ReLU}(\mathbf{z})$.

**(c)** Tính output sau Sigmoid: $\mathbf{h} = \sigma(\mathbf{z})$. *(Cho: $\sigma(4) \approx 0.982$, $\sigma(-5) \approx 0.007$)*

**(d)** Mini-batch gồm 2 mẫu $X = \begin{bmatrix} 2 & 1 & -1 \\ 0 & -1 & 3 \end{bmatrix}$. Tính $Y = XW^\top + \mathbf{b}^\top$.

<details>
<summary>📋 Đáp án Bài 2</summary>

**(a) Pre-activation $\mathbf{z} = W\mathbf{x} + \mathbf{b}$:**

$$z_1 = 1 \times 2 + (-1) \times 1 + 2 \times (-1) + 1 = 2 - 1 - 2 + 1 = \mathbf{0}$$
$$z_2 = 0 \times 2 + 3 \times 1 + (-1) \times (-1) + (-2) = 0 + 3 + 1 - 2 = \mathbf{2}$$

$$\mathbf{z} = \begin{bmatrix} 0 \\ 2 \end{bmatrix}$$

**(b) Sau ReLU:**

$$\text{ReLU}(0) = \max(0, 0) = 0, \quad \text{ReLU}(2) = \max(0, 2) = 2$$

$$\mathbf{h}_{ReLU} = \begin{bmatrix} 0 \\ 2 \end{bmatrix}$$

**(c) Sau Sigmoid** — nhưng trước tiên cần $\mathbf{z}$ từ (a) là $[0, 2]$:

$$\sigma(0) = \frac{1}{1+e^0} = \frac{1}{2} = 0.5, \quad \sigma(2) \approx 0.880$$

$$\mathbf{h}_{\sigma} = \begin{bmatrix} 0.500 \\ 0.880 \end{bmatrix}$$

*(Ghi chú: $\sigma(2) = 1/(1+e^{-2}) = 1/(1+0.135) \approx 0.880$)*

**(d) Mini-batch** $X \in \mathbb{R}^{2 \times 3}$, $W^\top \in \mathbb{R}^{3 \times 2}$:

$$Y = XW^\top + \mathbf{b}^\top$$

$W^\top = \begin{bmatrix} 1 & 0 \\ -1 & 3 \\ 2 & -1 \end{bmatrix}$

Mẫu 1 ($\mathbf{x} = [2, 1, -1]$): đã tính ở (a) → $[0, 2]$

Mẫu 2 ($\mathbf{x} = [0, -1, 3]$):
$$z_1 = 1(0) + (-1)(-1) + 2(3) + 1 = 0 + 1 + 6 + 1 = \mathbf{8}$$
$$z_2 = 0(0) + 3(-1) + (-1)(3) + (-2) = 0 - 3 - 3 - 2 = \mathbf{-8}$$

$$Y = \begin{bmatrix} 0 & 2 \\ 8 & -8 \end{bmatrix}$$

**Shape check:** $X(2×3) \times W^\top(3×2) = (2×2)$ → thêm bias $(1×2)$ broadcast → $Y(2×2)$ ✓

</details>

---

### Bài 3 — Activation Functions: giá trị và đạo hàm

**(a)** Điền vào bảng sau (tính tay đến 3 chữ số thập phân):

| $z$ | Sigmoid $\sigma(z)$ | $\sigma'(z)$ | Tanh$(z)$ | Tanh$'(z)$ | ReLU$(z)$ | ReLU$'(z)$ |
|-----|---------------------|-------------|-----------|-----------|-----------|-----------|
| $-3$ | ? | ? | ? | ? | ? | ? |
| $0$ | ? | ? | ? | ? | ? | ? |
| $1$ | ? | ? | ? | ? | ? | ? |
| $3$ | ? | ? | ? | ? | ? | ? |

*(Cho: $e^{-1} \approx 0.368$, $e^{-3} \approx 0.050$, $e^1 \approx 2.718$, $e^3 \approx 20.09$)*

**(b)** Tại $z = 3$, gradient qua Sigmoid là bao nhiêu? Nếu mạng có 5 lớp Sigmoid, gradient sẽ bị nhân bao nhiêu lần qua 5 lớp đó? Tính giá trị và giải thích vấn đề vanishing gradient.

**(c)** Một neuron ReLU có $z = -2$ sau forward pass. Gradient từ lớp sau truyền về là $\delta = 5$. Gradient truyền qua neuron này là bao nhiêu? Điều gì xảy ra?

<details>
<summary>📋 Đáp án Bài 3</summary>

**(a) Bảng giá trị:**

Công thức:
- $\sigma(z) = 1/(1+e^{-z})$, $\sigma'(z) = \sigma(z)(1-\sigma(z))$
- $\tanh(z) = (e^z-e^{-z})/(e^z+e^{-z})$, $\tanh'(z) = 1 - \tanh^2(z)$
- $\text{ReLU}(z) = \max(0,z)$, $\text{ReLU}'(z) = \mathbf{1}[z>0]$

| $z$ | $\sigma(z)$ | $\sigma'(z)$ | $\tanh(z)$ | $\tanh'(z)$ | ReLU | ReLU' |
|-----|------------|-------------|-----------|------------|------|-------|
| $-3$ | $1/(1+20.09) \approx 0.047$ | $0.047×0.953 \approx 0.045$ | $(0.050-20.09)/(0.050+20.09) \approx -0.995$ | $1-(-0.995)^2 \approx 0.010$ | $0$ | $0$ |
| $0$ | $1/2 = 0.500$ | $0.5×0.5 = 0.250$ | $0$ | $1-0 = 1.000$ | $0$ | $0^*$ |
| $1$ | $1/(1+0.368) \approx 0.731$ | $0.731×0.269 \approx 0.197$ | $(2.718-0.368)/(2.718+0.368) \approx 0.762$ | $1-0.762^2 \approx 0.419$ | $1$ | $1$ |
| $3$ | $1/(1+0.050) \approx 0.952$ | $0.952×0.048 \approx 0.046$ | $(20.09-0.050)/(20.09+0.050) \approx 0.995$ | $1-0.995^2 \approx 0.010$ | $3$ | $1$ |

*(* ReLU không khả vi tại $z=0$; thường định nghĩa subgradient = 0 hoặc 1)*

**Nhận xét quan trọng:**
- Sigmoid và Tanh đều bị **saturate** tại $z = \pm 3$: $\sigma'(3) \approx 0.046$, $\tanh'(3) \approx 0.010$ — rất nhỏ.
- ReLU: gradient chỉ là 0 hoặc 1 — không bị saturate vùng dương.

**(b) Vanishing gradient qua 5 lớp Sigmoid tại $z = 3$:**

Gradient mỗi lớp: $\sigma'(3) \approx 0.046$

Qua 5 lớp: $0.046^5 \approx 0.046 \times 0.046 \times 0.046 \times 0.046 \times 0.046 \approx \mathbf{2 \times 10^{-7}}$

Gradient gần như **triệt tiêu hoàn toàn** — lớp đầu gần như không nhận được tín hiệu học từ loss. Đây là lý do mạng sâu với Sigmoid **rất khó huấn luyện** trước khi ReLU ra đời.

**(c) Dead neuron:**

$\text{ReLU}'(-2) = 0$ (vì $z < 0$)

Gradient lan truyền ngược: $\delta_{\text{back}} = \delta \times \text{ReLU}'(-2) = 5 \times 0 = \mathbf{0}$

Gradient = 0 → trọng số của neuron này **không được cập nhật** trong bước này. Nếu neuron này **luôn** có $z < 0$ (do trọng số khởi tạo tệ hoặc learning rate quá lớn), nó sẽ **không bao giờ học** — đây là "dead neuron". Leaky ReLU giải quyết bằng cách cho gradient nhỏ $\alpha$ khi $z < 0$.

</details>

---

### Bài 4 — Forward Pass toàn MLP (tính tay)

Cho MLP 2 hidden layers phân loại nhị phân:

**Layer 1:** $W^{(1)} = \begin{bmatrix} 1 & 2 \\ -1 & 1 \end{bmatrix}$, $\mathbf{b}^{(1)} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, activation = ReLU

**Layer 2:** $W^{(2)} = \begin{bmatrix} 1 & -2 \end{bmatrix}$, $b^{(2)} = 0.5$, activation = Sigmoid

Input: $\mathbf{x} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$

**(a)** Tính $\mathbf{h}^{(1)}$ (output của hidden layer 1 sau ReLU).

**(b)** Tính $z^{(2)}$ (pre-activation của output layer).

**(c)** Tính $\hat{p} = \sigma(z^{(2)})$. Nếu ngưỡng $\tau = 0.5$, mô hình phân loại mẫu này vào lớp nào?

*(Cho: $\sigma(-0.5) \approx 0.378$)*

<details>
<summary>📋 Đáp án Bài 4</summary>

**(a) Hidden layer 1:**

$$\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)} = \begin{bmatrix} 1 & 2 \\ -1 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ -1 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

$$z^{(1)}_1 = 1(1) + 2(-1) + 0 = 1 - 2 = -1$$
$$z^{(1)}_2 = -1(1) + 1(-1) + 1 = -1 - 1 + 1 = -1$$

$$\mathbf{z}^{(1)} = \begin{bmatrix} -1 \\ -1 \end{bmatrix}$$

$$\mathbf{h}^{(1)} = \text{ReLU}\left(\begin{bmatrix} -1 \\ -1 \end{bmatrix}\right) = \begin{bmatrix} \max(0,-1) \\ \max(0,-1) \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

**(b) Output layer pre-activation:**

$$z^{(2)} = W^{(2)}\mathbf{h}^{(1)} + b^{(2)} = \begin{bmatrix} 1 & -2 \end{bmatrix}\begin{bmatrix} 0 \\ 0 \end{bmatrix} + 0.5 = 0 + 0.5 = \mathbf{0.5}$$

**(c) Xác suất và quyết định:**

$$\hat{p} = \sigma(0.5) = \frac{1}{1+e^{-0.5}} \approx \frac{1}{1+0.607} \approx \mathbf{0.622}$$

Vì $\hat{p} = 0.622 > \tau = 0.5$ → mô hình phân loại vào **lớp 1 (dương)**.

**Quan sát thú vị:** Cả hai neuron của hidden layer đều bị "chết" (output = 0) vì input trước activation đều âm. Nhờ bias $b^{(2)} = 0.5$, output layer vẫn cho ra $z^{(2)} = 0.5$ và phân loại được — nhưng đây là dấu hiệu của **mạng chưa học hiệu quả**. Trong thực tế cần kiểm tra dead neurons và có thể cần re-init hoặc dùng Leaky ReLU.

</details>

---

### Bài 5 — Batch Normalization

Một batch 4 mẫu, một feature (chiều) duy nhất:

$$\mathbf{x} = [2,\ 4,\ 6,\ 8]$$

BatchNorm có tham số học được: $\gamma = 2$, $\beta = -1$.

**(a)** Tính $\mu_B$ và $\sigma_B^2$ của batch.

**(b)** Tính $\hat{\mathbf{x}}$ (sau chuẩn hóa, dùng $\epsilon = 0$).

**(c)** Tính output $\mathbf{y} = \gamma \hat{\mathbf{x}} + \beta$.

**(d)** Nếu không có $\gamma$ và $\beta$ (chỉ chuẩn hóa về $\mathcal{N}(0,1)$), mạng bị mất đi điều gì? Tại sao $\gamma, \beta$ quan trọng?

<details>
<summary>📋 Đáp án Bài 5</summary>

**(a) Thống kê batch:**

$$\mu_B = \frac{2+4+6+8}{4} = \frac{20}{4} = \mathbf{5}$$

$$\sigma_B^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = \frac{9+1+1+9}{4} = \frac{20}{4} = \mathbf{5}$$

$$\sigma_B = \sqrt{5} \approx 2.236$$

**(b) Chuẩn hóa** ($\epsilon = 0$):

$$\hat{x}_i = \frac{x_i - \mu_B}{\sigma_B}$$

| $x_i$ | $x_i - \mu_B$ | $\hat{x}_i$ |
|--------|--------------|------------|
| 2 | $-3$ | $-3/2.236 \approx -1.342$ |
| 4 | $-1$ | $-1/2.236 \approx -0.447$ |
| 6 | $+1$ | $+1/2.236 \approx +0.447$ |
| 8 | $+3$ | $+3/2.236 \approx +1.342$ |

Kiểm tra: $\sum \hat{x}_i = 0$, $\text{Var}(\hat{\mathbf{x}}) = 1$ ✓ (phân phối chuẩn hóa)

**(c) Scale và Shift** ($\gamma = 2$, $\beta = -1$):

$$y_i = 2\hat{x}_i - 1$$

| $\hat{x}_i$ | $y_i = 2\hat{x}_i - 1$ |
|------------|----------------------|
| $-1.342$ | $2(-1.342) - 1 = -3.684$ |
| $-0.447$ | $2(-0.447) - 1 = -1.894$ |
| $+0.447$ | $2(0.447) - 1 = -0.106$ |
| $+1.342$ | $2(1.342) - 1 = +1.684$ |

**(d) Vai trò của $\gamma$ và $\beta$:**

Nếu chỉ chuẩn hóa về $\mathcal{N}(0,1)$: **mọi lớp BN đều buộc feature có mean=0, std=1** — mạng mất khả năng tự điều chỉnh phân phối. Ví dụ:
- Nếu activation tiếp theo là Sigmoid, vùng tuyến tính của Sigmoid là $z \in (-1, 1)$ → chuẩn hóa về $(0,1)$ là tốt.
- Nhưng nếu cần mean khác 0 hoặc std khác 1 để tối ưu hơn, mạng không tự điều chỉnh được.

$\gamma$ và $\beta$ là **tham số học được** — mạng tự chọn phân phối tốt nhất. Trong trường hợp đặc biệt $\gamma = \sigma_B$, $\beta = \mu_B$, BN trở thành identity (giữ nguyên input) — mạng có thể "tắt" BN nếu không cần. Đây là thiết kế linh hoạt.

</details>

---

### Bài 6 — So sánh và Lý luận Kiến trúc

**(a)** MLP với 3 hidden layers, mỗi layer 100 neurons, không có activation function (chỉ dùng FC thuần). Kiến trúc này tương đương với gì? Viết ra tường minh.

**(b)** Cho hai MLP có cùng số tham số:
- **Mạng A:** 1 hidden layer, 1000 neurons (wide & shallow)
- **Mạng B:** 4 hidden layers, mỗi layer 100 neurons (deep & narrow)

Trong thực tế, mạng nào thường hoạt động tốt hơn? Tại sao theo Universal Approximation Theorem và hierarchical learning?

**(c)** Một mạng MLP huấn luyện trên ảnh 28×28. Nếu ta dịch chuyển ảnh 1 pixel sang phải trước khi đưa vào mạng, output sẽ thay đổi đáng kể không? So sánh với CNN (gợi ý: weight sharing).

<details>
<summary>📋 Đáp án Bài 6</summary>

**(a) MLP không có activation:**

$$\mathbf{h}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$$
$$\mathbf{h}^{(2)} = W^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)} = W^{(2)}(W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)}$$
$$= \underbrace{(W^{(2)}W^{(1)})}_{W'}\mathbf{x} + \underbrace{W^{(2)}\mathbf{b}^{(1)} + \mathbf{b}^{(2)}}_{\mathbf{b}'}$$

Sau 3 lớp:
$$\mathbf{h}^{(3)} = W^{(3)}W^{(2)}W^{(1)}\mathbf{x} + \text{bias terms} = W_{eff}\mathbf{x} + \mathbf{b}_{eff}$$

**Kết quả: Tương đương một FC layer duy nhất** $W_{eff} = W^{(3)}W^{(2)}W^{(1)}$. Ba lớp FC không có activation = vô nghĩa, tốn tham số mà không học được gì phi tuyến hơn một lớp. Đây là lý do **activation function là bắt buộc** giữa các lớp.

**(b) Wide & Shallow vs Deep & Narrow:**

*Universal Approximation Theorem* đảm bảo **cả hai** đều có thể xấp xỉ hàm phức tạp nếu đủ tham số — nhưng với số tham số bằng nhau:

**Mạng B (deep) thường tốt hơn** vì:

1. **Hierarchical feature learning:** Lớp 1 học cạnh/texture cơ bản, lớp 2 học các phần của đối tượng, lớp 3 học cấu trúc cao cấp hơn — mỗi lớp xây dựng trên output của lớp trước. Mạng nông phải học tất cả trực tiếp từ input.

2. **Parameter efficiency:** Hàm phức tạp có thể cần **số neuron mũ** trong mạng nông, nhưng chỉ cần **số neuron đa thức** trong mạng sâu.

3. **Thực nghiệm:** Hầu hết kết quả state-of-the-art đều từ mạng **sâu** — ResNet (50-200 layers), Transformer (12-96 blocks).

*Nhưng deep networks có vấn đề:* Vanishing gradient, khó optimize → cần BatchNorm, skip connections (ResNet), learning rate scheduling.

**(c) MLP vs CNN với dịch chuyển ảnh:**

**MLP:** Mỗi pixel $i$ kết nối với neuron $j$ qua trọng số $W_{j,i}$ riêng biệt. Pixel $(5, 5)$ và pixel $(5, 6)$ dùng **trọng số khác nhau** hoàn toàn. Khi dịch ảnh 1 pixel, tất cả pixel thay đổi vị trí → input đến mạng thay đổi hoàn toàn → output thay đổi đáng kể. MLP **không có translation invariance** — phải học lại cho mỗi vị trí.

**CNN:** Cùng filter (với cùng trọng số) được áp lên tất cả vị trí ảnh. Nếu ảnh dịch 1 pixel, feature map cũng dịch 1 pixel, nhưng **đặc trưng được phát hiện là như nhau**. CNN có **translation equivariance** nhờ weight sharing — cùng pattern ở vị trí khác vẫn được nhận ra.

**Đây là lý do cốt lõi tại sao CNN vượt trội MLP cho ảnh:**
- MLP: $784 \times 256 = 200{,}704$ params để xử lý một ảnh 28×28
- CNN 3×3 với 32 filters: chỉ $32 \times (3×3×1 + 1) = 320$ params — **chia sẻ** qua toàn bộ ảnh

</details>

---

### Tổng hợp công thức và mẹo thi

| Chủ đề | Công thức / Quy tắc | Mẹo nhớ |
|--------|---------------------|---------|
| **FC params** | $M(N+1)$ — $N$ vào, $M$ ra | Weights = $M \times N$, Biases = $M$ |
| **FC forward** | $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, batch: $Y = XW^\top + \mathbf{b}^\top$ | Shape: $(B,N) \times (N,M) = (B,M)$ |
| **Sigmoid** | $\sigma(z) = 1/(1+e^{-z})$, $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | Max đạo hàm = 0.25 tại $z=0$ |
| **Tanh** | $\tanh'(z) = 1 - \tanh^2(z)$ | Max đạo hàm = 1 tại $z=0$ |
| **ReLU** | $\max(0,z)$, gradient = 1 nếu $z>0$, = 0 nếu $z<0$ | Dead neuron khi $z < 0$ mãi |
| **Vanishing grad** | $(0.046)^5 \approx 10^{-7}$ qua 5 lớp Sigmoid | Tại sao ReLU thay thế Sigmoid |
| **BatchNorm** | Chuẩn hóa: $\hat{x} = (x-\mu_B)/\sigma_B$; Scale-shift: $y = \gamma\hat{x}+\beta$ | $\gamma, \beta$ là tham số học được |
| **FC không activation** | $n$ lớp FC liên tiếp = 1 FC | Không phi tuyến = vô nghĩa |
| **Tại sao deep > wide** | Hierarchical learning + parameter efficiency | ResNet 50 layer >> 1 layer wide |

## Ghi chú về ảnh cần đính kèm

| # | Mô tả ảnh | Trang slide | Gợi ý nguồn |
|---|-----------|-------------|-------------|
| 1 | Pipeline: x → Feature Transformer → Head → output (dạng block diagram) | Trang 11 | Vẽ tay hoặc draw.io |
| 2 | Bên trong Feature Transformer: chuỗi FC→A→FC→A→... | Trang 12–13 | Trang 12–13 slide |
| 3 | Sơ đồ FC layer N=4, M=3 với tất cả kết nối | Trang 22 | Trang 22 slide; hoặc vẽ bằng `networkx` |
| 4 | Đồ thị Sigmoid: $\sigma(z)$ và $\sigma'(z)$ — thấy rõ vùng bão hòa | Trang 24 | Trang 24 slide; `plt.plot` |
| 5 | Đồ thị Tanh: $\tanh(z)$ và đạo hàm | Trang 25 | Trang 25 slide; `plt.plot` |
| 6 | Đồ thị ReLU: hình "hockey stick" và step function đạo hàm | Trang 26 | Trang 26 slide; `plt.plot` |
| 7 | Đồ thị Leaky ReLU: so sánh với ReLU, độ dốc $\alpha$ vùng âm | Trang 27 | Trang 27 slide; `plt.plot` |
| 8 | Đồ thị SiLU/Swish: độ mượt và vùng âm nhỏ | Trang 28 | Trang 28 slide; `plt.plot` |
| 9 | So sánh 5 activation functions trên cùng một đồ thị | Không trong slide | `plt.plot` với 5 đường màu khác nhau |
