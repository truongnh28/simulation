# Logistic Regression — Bài giảng chi tiết

> **Nguồn:** Slide `slides-v1/foundation/04-ClassificationSummary.pdf` — Th.S Lê Thanh Sách, Khoa KHMT, ĐH Bách Khoa TP.HCM (05/02/2026)

---

## Mục lục

1. [Architecture View — Kiến trúc mô hình phân loại](#1-architecture-view)
2. [Activation Functions — Hàm kích hoạt](#2-activation-functions)
3. [Mathematical View — Công thức toán học](#3-mathematical-view)
4. [Programming View — Lập trình thực tế](#4-programming-view)
5. [Training Methods — Phương pháp huấn luyện](#5-training-methods)

---

## 1. Architecture View

### Sự khác biệt cơ bản giữa Regression và Classification

Trong **Linear Regression**, đầu ra là một số thực bất kỳ (giá nhà, nhiệt độ, ...). Nhưng trong bài toán **Classification**, ta cần đầu ra là **xác suất** — một con số trong khoảng $[0, 1]$ — hoặc một **phân phối xác suất** trên các lớp.

Vì vậy, kiến trúc **Logistic Regression** thêm một bước so với Linear Regression: sau lớp FC, ta đưa kết quả qua một **hàm kích hoạt (Activation Function)** để ép đầu ra về dạng xác suất.

> 📷 **[Ảnh slide 4 (Classification)]** Sơ đồ kiến trúc: `X → [FC | W,b] → Z → [Activation] → Ŷ`
> Nguồn: slide 4, `04-ClassificationSummary.pdf`

---

### Quy ước ký hiệu

| Ký hiệu | Shape | Ý nghĩa |
|---------|-------|---------|
| $\mathbf{X} \in \mathbb{R}^{B \times D}$ | `(B, D)` | Input batch (hàng $n$ là $\mathbf{x}_n^T$) |
| $\mathbf{Z} \in \mathbb{R}^{B \times K}$ | `(B, K)` | **Logits** — đầu ra của lớp FC (chưa qua activation) |
| $\hat{\mathbf{Y}} \in \mathbb{R}^{B \times K}$ | `(B, K)` | **Probabilities** — đầu ra mô hình sau activation |
| $\mathbf{Y}$ | — | Nhãn thực tế |
| $\mathbf{W} \in \mathbb{R}^{D \times K}$, $\mathbf{b} \in \mathbb{R}^K$ | — | Tham số học được |
| $N$ | — | Số mẫu (batch size) |
| $D$ | — | Số chiều đặc trưng (input dimension) |
| $K$ | — | Số lớp / số nhãn (output dimension) |

> **Khái niệm "Logit":** $Z_{n,k}$ là điểm số "thô" mà mô hình tính được cho mẫu $n$ thuộc lớp $k$. Nó chưa phải xác suất vì có thể âm hoặc lớn hơn 1. Hàm activation sẽ chuyển logit thành xác suất.

---

### Ba loại bài toán Classification

Tùy vào cấu trúc của bài toán, ta chọn số đầu ra $K$ và loại activation khác nhau:

| Loại | Ví dụ | $K$ outputs | Activation |
|------|-------|-------------|-----------|
| **Binary** (nhị phân) | Email spam/không spam | $K=1$ (một logit) | Sigmoid |
| **Single-label** (đa lớp, chọn 1) | Nhận diện chữ số 0-9 | $K>1$ logits | Softmax |
| **Multi-label** (đa nhãn, chọn nhiều) | Ảnh có cả "người" lẫn "xe" | $K$ logits | Sigmoid mỗi lớp |

> **Phép ẩn dụ hay:**
> - **Binary**: câu hỏi đúng/sai — "Con vật này có phải là mèo không?"
> - **Single-label**: câu hỏi chọn một — "Con vật này là mèo, chó, hay thỏ?"
> - **Multi-label**: câu hỏi chọn nhiều — "Trong ảnh này có những gì?" (có thể có cả mèo lẫn ghế sofa)

---

### Luồng tính toán tổng quát

**Bước 1 — FC (Linear Transformation):**
$$\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{1}\mathbf{b}^T, \quad \mathbf{Z} \in \mathbb{R}^{B \times K}$$

**Bước 2 — Activation (Probabilistic Interpretation):**
$$\hat{\mathbf{Y}} = \text{Activation}(\mathbf{Z})$$

---

## 2. Activation Functions

### 2.1 Sigmoid — Dành cho Binary Classification

> 📷 **[Ảnh slide 7]** Đồ thị hàm Sigmoid: đường cong hình chữ S, trục x là logit $z \in [-10, 10]$, trục y là $\sigma(z) \in (0, 1)$, giao điểm tại $(0, 0.5)$.
> Nguồn: slide 7, `04-ClassificationSummary.pdf`

**Định nghĩa:**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Cơ chế hoạt động:**
- Khi $z \to +\infty$: $\sigma(z) \to 1$ (mô hình rất tự tin thuộc lớp dương)
- Khi $z \to -\infty$: $\sigma(z) \to 0$ (mô hình rất tự tin thuộc lớp âm)
- Khi $z = 0$: $\sigma(z) = 0.5$ (mô hình hoàn toàn không chắc)

**Diễn giải xác suất:**
- Input: **logit** $z = \mathbf{w}^T\mathbf{x} + b$
- Output: $\hat{y} \in (0, 1)$
- Ý nghĩa: $\hat{y} = P(y = 1 \mid \mathbf{x})$ — xác suất mẫu $\mathbf{x}$ thuộc lớp dương

> **Tại sao gọi là "logistic" regression?** Vì "logit" là nghịch đảo của sigmoid. Nếu $\hat{y} = \sigma(z)$, thì $z = \log\frac{\hat{y}}{1-\hat{y}}$ — đây chính là "log-odds" hay "logit". Mô hình học một hàm tuyến tính trên không gian logit.

---

### 2.2 Softmax — Dành cho Single-label Classification

**Định nghĩa:**

$$\hat{y}_k = \left(\text{softmax}(\mathbf{z})\right)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad k = 1, \ldots, K$$

**Cơ chế hoạt động:** Softmax "chuẩn hóa" toàn bộ vector logit thành phân phối xác suất. Bằng cách lấy exponential, nó **khuếch đại** sự chênh lệch giữa các logit — lớp có logit cao nhất sẽ được gán xác suất gần 1.

**Tính chất:**
- $\hat{y}_k \geq 0$ — mỗi xác suất là không âm
- $\sum_{k=1}^{K} \hat{y}_k = 1$ — tổng xác suất bằng 1 (phân phối hợp lệ)
- Diễn giải: **phân phối categorical** — $P(y = k \mid \mathbf{x}) = \hat{y}_k$

> **Phép ẩn dụ:** Nếu một giáo viên cho điểm $z_k$ cho K học sinh, Softmax chuyển điểm thô thành "thị phần" — học sinh nào điểm cao nhất được "chú ý" nhất, nhưng tất cả cùng nhận được một phần.

> **Quan hệ Binary ↔ Softmax:** Sigmoid cho $K=1$ tương đương với Softmax cho $K=2$. Đây không phải ngẫu nhiên — chúng đều xuất phát từ cùng một lý thuyết xác suất (exponential family).

---

### Bảng tổng hợp Activation theo loại bài toán

| Loại bài toán | FC Output | Activation | Model Output |
|---------------|-----------|------------|--------------|
| Binary Classification | $\mathbf{Z} \in \mathbb{R}^{B \times 1}$ | **Sigmoid** | $\hat{y} = \sigma(z) \in (0,1)$ |
| Single-label Classification | $\mathbf{Z} \in \mathbb{R}^{B \times K}$ | **Softmax** | $\hat{Y}_{n,k} = \frac{e^{Z_{n,k}}}{\sum_j e^{Z_{n,j}}}$ |
| Multi-label Classification | $\mathbf{Z} \in \mathbb{R}^{B \times K}$ | **Sigmoid** (per class) | $\hat{Y}_{n,k} = \sigma(Z_{n,k})$ |

> **Câu nói đáng nhớ:** *"Activation determines the probabilistic interpretation. Same architecture, different learning problems."*
>
> Một điều đẹp đẽ: kiến trúc FC + Activation xử lý được cả ba loại bài toán — chỉ thay đổi lớp activation là thay đổi toàn bộ ngữ nghĩa học máy của mô hình.

---

## 3. Mathematical View

### Công thức — Single Sample

Với $\mathbf{x} \in \mathbb{R}^D$:

1. Tính **logits**:
$$\mathbf{z} = \mathbf{W}^T\mathbf{x} + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{D \times K}, \; \mathbf{b} \in \mathbb{R}^K$$

2. Tính **prediction**:
$$\hat{\mathbf{y}} = \text{Activation}(\mathbf{z})$$

### Công thức — Batch

$$\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{1}\mathbf{b}^T, \quad \hat{\mathbf{Y}} = \text{Activation}(\mathbf{Z})$$

> **Câu nói đáng nhớ:** *"Logistic regression (and softmax / multi-label logistic) are linear models followed by a non-linear activation."*
>
> Đây là insight quan trọng: Logistic Regression **linear** ở phần FC, nhưng **phi tuyến** ở phần activation. Chính hàm activation phi tuyến này cho phép mô hình tạo ra **ranh giới quyết định phi tuyến** trong không gian xác suất, dù ranh giới trong không gian đặc trưng vẫn là tuyến tính.

---

## 4. Programming View

### 4.1 PyTorch — Binary Classification

```python
import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, X):  # X: (N, D)
        return self.fc(X)  # trả về logits, shape: (N, 1)

model = BinaryClassifier(input_dim=D)
```

**Loss function:** `nn.BCEWithLogitsLoss()` — Sigmoid được tích hợp **bên trong** loss.

---

### 4.2 PyTorch — Single-label Classification (Softmax Regression)

```python
import torch
import torch.nn as nn

class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, X):  # X: (N, D)
        return self.fc(X)  # trả về logits, shape: (N, K)

model = SoftmaxClassifier(input_dim=D, num_classes=K)
```

**Loss function:** `nn.CrossEntropyLoss()` — Softmax được tích hợp **bên trong** loss.

---

### 4.3 PyTorch — Multi-label Classification

```python
import torch
import torch.nn as nn

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, X):  # X: (N, D)
        return self.fc(X)  # trả về logits, shape: (N, K)

model = MultiLabelClassifier(input_dim=D, num_labels=K)
```

**Loss function:** `nn.BCEWithLogitsLoss()` — Sigmoid áp dụng **từng nhãn độc lập**.

---

### 4.4 Bảng tổng hợp Loss Function trong PyTorch

| Bài toán | Model output (logits) | Loss Function | Activation trong Loss |
|----------|-----------------------|---------------|----------------------|
| Binary Classification | $\mathbf{Z} \in \mathbb{R}^{B \times 1}$ | `BCEWithLogitsLoss` | Sigmoid |
| Single-label | $\mathbf{Z} \in \mathbb{R}^{B \times K}$ | `CrossEntropyLoss` | Softmax |
| Multi-label | $\mathbf{Z} \in \mathbb{R}^{B \times K}$ | `BCEWithLogitsLoss` | Sigmoid (per class) |
| Regression | $\hat{\mathbf{Y}} \in \mathbb{R}^{N \times K}$ | `MSELoss` | None |

---

### ⚠️ Lưu ý quan trọng trong PyTorch

> **KHÔNG được thêm activation vào đầu ra của model khi dùng các loss trên!**

```python
# SAI — Double sigmoid!
def forward(self, X):
    return torch.sigmoid(self.fc(X))  # ❌
# Rồi dùng BCEWithLogitsLoss -> sigmoid bị áp dụng 2 lần

# ĐÚNG — chỉ trả về logits
def forward(self, X):
    return self.fc(X)  # ✅
# BCEWithLogitsLoss tự áp dụng sigmoid một lần
```

**Hậu quả của double activation:**
- Gradient sai → mô hình học không đúng
- Training chậm hoặc không hội tụ
- Đây là một trong những lỗi phổ biến nhất của người mới học PyTorch

---

### 4.5 Keras (TensorFlow)

```python
import tensorflow as tf
from tensorflow.keras import layers

# Binary classification
model_binary = tf.keras.Sequential([
    layers.Dense(1, activation="sigmoid", input_shape=(D,))
])

# Single-label classification
model_softmax = tf.keras.Sequential([
    layers.Dense(K, activation="softmax", input_shape=(D,))
])

# Multi-label classification
model_multilabel = tf.keras.Sequential([
    layers.Dense(K, activation="sigmoid", input_shape=(D,))
])
```

**Triết lý thiết kế Keras** (ngược với PyTorch):
- Trong Keras, **activation là một phần của model** — bạn khai báo activation ngay trong lớp Dense.
- Loss function của Keras **mặc định nhận xác suất** làm input, không phải logits.
- Vì vậy, **phải nhớ thêm activation** vào model, không thể quên.

**Bảng Activation vs Loss trong Keras:**

| Bài toán | Last Layer | Activation | Loss Function |
|----------|-----------|------------|---------------|
| Binary | `Dense(1)` | `sigmoid` | `BinaryCrossentropy` |
| Single-label | `Dense(K)` | `softmax` | `CategoricalCrossentropy` |
| Multi-label | `Dense(K)` | `sigmoid` | `BinaryCrossentropy` |
| Regression | `Dense(K)` | None (Linear) | `MeanSquaredError` |

> **Một số Keras losses có `from_logits=True`** nếu bạn muốn theo phong cách PyTorch (output logits, tích hợp activation vào loss). Cả hai cách đều hợp lệ, nhưng phải nhất quán.

---

## 5. Training Methods

### Tại sao Classification không có Closed-form?

Nhớ lại rằng Linear Regression có nghiệm dạng đóng OLS. Câu hỏi tự nhiên là: Logistic Regression có không?

**Câu trả lời: Không.** Lý do: hàm kích hoạt Sigmoid/Softmax là **phi tuyến**. Khi bạn đặt đạo hàm của loss function bằng 0 và cố gắng giải, bạn không thể ra nghiệm dạng đóng nữa. Phương trình trở nên quá phức tạp.

**Kết luận:** Classification **bắt buộc phải dùng Gradient Descent** (tối ưu hóa lặp).

> **Nguyên nhân sâu xa:** OLS hoạt động được cho Linear Regression vì hàm mục tiêu (MSE) là **hàm lồi bậc hai** (quadratic convex) theo tham số — có một điểm cực tiểu duy nhất và ta có thể tính trực tiếp. Khi thêm Sigmoid/Softmax, hàm mục tiêu không còn dạng đó nữa.

---

### Gradient Descent cho Classification

> 📷 **[Ảnh slide 23]** Sơ đồ training phân loại: `X → [Neural Network (W,b)] → Ŷ → [Loss Function] → ℓ`, với Y cũng đi vào Loss.
> Nguồn: slide 23, `04-ClassificationSummary.pdf`

**Bài toán tối ưu:**

$$(W^*, b^*) = \arg\min_{\mathbf{W}, \mathbf{b}} \mathcal{L}(\hat{\mathbf{Y}}, \mathbf{Y})$$

**Quy tắc cập nhật** (giống Linear Regression):

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta\nabla_{\mathbf{W}}\mathcal{L}$$
$$\mathbf{b}_{t+1} = \mathbf{b}_t - \eta\nabla_{\mathbf{b}}\mathcal{L}$$

Điểm khác biệt nằm ở **hàm loss** và cách tính gradient, không phải ở cấu trúc cập nhật.

---

### Các hàm Loss cho Classification

#### Binary Cross-Entropy (BCE)

Dùng cho **Binary Classification** ($y_i \in \{0, 1\}$):

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

**Giải thích trực quan:**
- Nếu $y_i = 1$ (mẫu dương): ta muốn $\hat{y}_i \to 1$, nên $-\log\hat{y}_i$ phạt nếu $\hat{y}_i$ thấp.
- Nếu $y_i = 0$ (mẫu âm): ta muốn $\hat{y}_i \to 0$, nên $-\log(1-\hat{y}_i)$ phạt nếu $\hat{y}_i$ cao.
- Đây là **log-likelihood âm** của mô hình Bernoulli — có nền tảng lý thuyết xác suất vững chắc.

#### Categorical Cross-Entropy (Softmax + CE)

Dùng cho **Single-label Classification** ($y_{ik} \in \{0,1\}$, one-hot encoded):

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik}\log\hat{y}_{ik}$$

**Nhận xét thú vị:** Vì $y_{ik}$ là one-hot (chỉ một $k$ bằng 1), tổng trên $k$ thực ra chỉ còn một số hạng không bằng 0: $-\log\hat{y}_{i, y_i}$ — nghĩa là loss chỉ phạt **xác suất của lớp đúng**. Ta muốn xác suất lớp đúng càng cao càng tốt.

#### Multi-label Binary Cross-Entropy

Dùng cho **Multi-label Classification** (mỗi nhãn độc lập):

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}\left[y_{ik}\log\hat{y}_{ik} + (1-y_{ik})\log(1-\hat{y}_{ik})\right]$$

Đây giống BCE nhưng áp dụng **độc lập trên từng nhãn** — mỗi nhãn là một bài toán binary riêng.

---

## Kết luận và Key Takeaways

### Tổng hợp điểm cốt lõi

| | Linear Regression | Logistic Regression |
|---|---|---|
| **Kiến trúc** | FC | FC + Activation |
| **Activation** | Không có | Sigmoid / Softmax |
| **Đầu ra** | Số thực tùy ý | Xác suất $\in (0,1)$ |
| **Loss** | MSE | BCE / Cross-Entropy |
| **Training** | Closed-form **hoặc** GD | **Chỉ** Gradient Descent |
| **Khi nào dùng** | Dự đoán giá trị liên tục | Phân loại |

> **Câu nói đáng nhớ để kết bài:** *"Activation + Loss define the learning problem."*
>
> Khi bạn chọn một Activation và một Loss function, bạn đang định nghĩa **mô hình xác suất** mà neural network đang cố gắng học. Đây không phải là quyết định kỹ thuật đơn thuần — mà là quyết định mang tính **toán học và xác suất** về bản chất của bài toán bạn muốn giải.

### Checklist thực hành

Khi implement một bài toán classification, hãy tự hỏi:

- [ ] **Loại bài toán là gì?** Binary, Single-label, hay Multi-label?
- [ ] **Số đầu ra $K$?** ($K=1$ cho Binary, $K=$ số lớp cho các loại còn lại)
- [ ] **Activation nào?** Sigmoid (Binary/Multi-label) hay Softmax (Single-label)?
- [ ] **PyTorch hay Keras?**
  - PyTorch: **KHÔNG** thêm activation vào model, dùng `BCEWithLogitsLoss` / `CrossEntropyLoss`
  - Keras: **PHẢI** thêm activation vào model, dùng `BinaryCrossentropy` / `CategoricalCrossentropy`
