# Bài 01: Problem Formulation — Phát Biểu Bài Toán

> **Nguồn slide:** `slides-v1/foundation/01-ProblemFormulation.pdf`
> **Giảng viên:** Thanh-Sach LE — ltsach@hcmut.edu.vn
> **Khoa KHMT, ĐHBK TP.HCM — 05/02/2026**

---

## Mục Lục

1. [Quy Ước Ký Hiệu](#1-quy-ước-ký-hiệu)
2. [Machine Learning là gì?](#2-machine-learning-là-gì)
3. [Học Có Giám Sát](#3-học-có-giám-sát)
4. [Bài Toán Phân Loại](#4-bài-toán-phân-loại-classification)
5. [Bài Toán Hồi Quy](#5-bài-toán-hồi-quy-regression)
6. [Quy Ước Batch và Ma Trận](#6-quy-ước-batch-và-ma-trận)
7. [Mục Tiêu Học](#7-mục-tiêu-học-learning-objective)
8. [Phân Phối Dữ Liệu và Khái Quát Hóa](#8-phân-phối-dữ-liệu-và-khái-quát-hóa)
9. [Phân Chia Tập Dữ Liệu](#9-phân-chia-tập-dữ-liệu)
10. [Không Gian Đầu Ra](#10-không-gian-đầu-ra-output-space)
11. [Mô Hình Tham Số và Inductive Bias](#11-mô-hình-tham-số-và-inductive-bias)
12. [Hàm Mất Mát](#12-hàm-mất-mát-loss-functions)

---

## 1. Quy Ước Ký Hiệu

> 📌 **Ảnh:** Slide trang 3 — `01-ProblemFormulation.pdf#page=3`
> Bảng tóm tắt ký hiệu vectors, matrices, output/labels, parameters & sizes.

Trước khi đi vào bất kỳ nội dung kỹ thuật nào, chúng ta cần thống nhất **ngôn ngữ toán học** dùng xuyên suốt khóa học. Đây không phải là việc học thuộc ký hiệu — đây là việc xây dựng một *ngôn ngữ chung* để diễn đạt ý tưởng một cách chính xác và không nhầm lẫn.

### 1.1 Vectors và Ma Trận

| Đối tượng | Ký hiệu | Ví dụ |
|---|---|---|
| **Vector** (column vector) | Chữ thường in đậm: **x**, **w**, **b** | **x** ∈ ℝ^{d×1} |
| **Ma trận / Tensor** | Chữ hoa in đậm: **X**, **W**, **Z** | **X** ∈ ℝ^{B×D} |

Theo quy ước, **mọi vector** trong khóa học này đều là **vector cột** ($d \times 1$). Điều này quan trọng khi tính toán ma trận.

### 1.2 Quy Ước Batch

Khi có nhiều mẫu dữ liệu, ta xếp chồng các **transpose** của từng vector thành từng hàng:

$$\mathbf{X} = \begin{bmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_B^T \end{bmatrix} \in \mathbb{R}^{B \times D}$$

**Mỗi hàng = một mẫu dữ liệu.** Quy ước này khớp với PyTorch và Keras vì các phép tính song song trên GPU được tối ưu khi data lưu *row-contiguous*.

**Shape theo từng loại dữ liệu:**

| Loại | Shape một mẫu | Shape một batch |
|---|---|---|
| Tabular | $(D,)$ | $(B, D)$ |
| Image | $(C, H, W)$ | $(B, C, H, W)$ |
| Sequence / Text | $(T,)$ hoặc $(T, D)$ | $(B, T)$ hoặc $(B, T, D)$ |

### 1.3 Ký Hiệu Đầu Ra và Tham Số

| Ký hiệu | Ý nghĩa |
|---|---|
| $\hat{\mathbf{Y}}$ hoặc $\hat{y}$ | **Đầu ra dự đoán** của mô hình |
| $\mathbf{Y}$ hoặc $y$ | **Nhãn thực tế** (ground-truth) |
| $\mathbf{Z}$ | **Logits** — giá trị thô trước hàm softmax/sigmoid |
| $\hat{\mathbf{P}}$ | **Xác suất** đầu ra sau hàm kích hoạt |
| $\boldsymbol{\theta}$ | **Toàn bộ tham số** học được của mô hình |
| $\mathbf{W}$, $\mathbf{b}$ | **Weights** (trọng số) và **bias** |
| $B, D, K, N$ | Batch size, số features, số classes, tổng số mẫu |

---

## 2. Machine Learning là gì?

> 📌 **Ảnh:** Slide trang 5 — `01-ProblemFormulation.pdf#page=5`
> Hộp "General Setting" với công thức D = {(x_i, y_i)} và f(x) ≈ y.

Hãy cùng suy nghĩ về cách lập trình truyền thống. Muốn viết chương trình phân loại ảnh mèo/chó, bạn phải viết hàng trăm luật: "nếu có tai nhọn thì là mèo"... Đây là công việc bất khả thi cho bài toán phức tạp.

**Machine learning** đưa ra triết lý hoàn toàn khác: *Thay vì viết luật, hãy cho máy học từ dữ liệu.*

### Cài Đặt Tổng Quát (General Setting)

Cho trước bộ dữ liệu $N$ cặp đầu vào — đầu ra:

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$$

Mục tiêu: học hàm $f(\cdot)$ sao cho:

$$f(\mathbf{x}) \approx y$$

Hàm $f$ được **tham số hóa** (parameterized) và **học từ dữ liệu** — không phải do con người viết tay từng luật.

> 💡 **Triết lý cốt lõi:** *Machine learning is about learning a mapping from data, not writing explicit rules.*

---

## 3. Học Có Giám Sát

> 📌 **Ảnh:** Slide trang 6 — `01-ProblemFormulation.pdf#page=6`
> Hai hộp: "Supervised Learning" (input features + ground-truth label → learn f: x↦y) và "Two Main Problem Types" (Classification, Regression).

**Học có giám sát (Supervised Learning)** — tên gọi "có giám sát" vì mỗi mẫu đầu vào đều có một **nhãn** (label) đi kèm, giống như giáo viên đang chỉnh sửa từng câu trả lời của học sinh.

**Dữ liệu huấn luyện gồm:**
- **Input features $\mathbf{x}$**: ảnh, văn bản, dữ liệu bảng, tín hiệu âm thanh...
- **Ground-truth label $y$**: nhãn lớp, giá trị số thực...

**Mục tiêu:** Học hàm $f: \mathbf{x} \mapsto y$ sao cho **khái quát hóa tốt** trên **dữ liệu chưa thấy** (unseen data).

Hai dạng bài toán chính: **Classification** và **Regression**.

---

## 4. Bài Toán Phân Loại (Classification)

> 📌 **Ảnh:** Slide trang 7 — `01-ProblemFormulation.pdf#page=7`
> Hộp "Classification Problem" với công thức y ∈ {1,...,K} và P(y=k|x); hộp "Common Classification Types".

**Đặc điểm:** Đầu ra $y$ thuộc về một **tập hữu hạn** các lớp:

$$y \in \{1, 2, \ldots, K\}$$

Mô hình tính toán **phân phối xác suất** trên tất cả các lớp:

$$P(y = k \mid \mathbf{x})$$

Câu trả lời cuối cùng: $\hat{y} = \arg\max_k P(y = k \mid \mathbf{x})$.

### Ba Biến Thể Phổ Biến

| Dạng | Điều kiện | Ví dụ |
|---|---|---|
| **Binary classification** | $K = 2$ | Spam/không spam; lành/ác tính |
| **Single-label multi-class** | $K > 2$, mỗi mẫu thuộc đúng 1 lớp | Nhận dạng chữ số 0–9, ImageNet |
| **Multi-label classification** | Mỗi mẫu có thể thuộc *nhiều* lớp | Ảnh vừa là "chó" vừa là "ngoài trời" |

> 💡 **Ẩn dụ:** Nếu classification là bài thi trắc nghiệm: binary = "đúng/sai", multi-class = "chọn 1 trong 4", multi-label = "chọn tất cả đáp án đúng".

---

## 5. Bài Toán Hồi Quy (Regression)

> 📌 **Ảnh:** Slide trang 8 — `01-ProblemFormulation.pdf#page=8`
> Hộp "Regression Problem" với công thức y ∈ ℝ hoặc y ∈ ℝ^K và ŷ = f(x); hộp "Typical Examples".

**Đặc điểm:** Đầu ra $y$ là một **giá trị liên tục**:

$$y \in \mathbb{R} \quad \text{hoặc} \quad \mathbf{y} \in \mathbb{R}^K$$

Mô hình dự đoán trực tiếp một số thực:

$$\hat{y} = f(\mathbf{x})$$

**Ví dụ điển hình:**
- **Price prediction**: Đầu vào là diện tích, vị trí → đầu ra là giá nhà (VNĐ)
- **Temperature forecasting**: Dữ liệu khí tượng → nhiệt độ ngày mai (°C)
- **Signal reconstruction**: Tín hiệu nhiễu → tín hiệu sạch

> 💡 **Điểm phân biệt:** Classification hỏi "**cái này là gì?**" (nhãn). Regression hỏi "**cái này bao nhiêu?**" (số thực).

---

## 6. Quy Ước Batch và Ma Trận

> 📌 **Ảnh:** Slide trang 9 — `01-ProblemFormulation.pdf#page=9`
> Hộp "Vector and Matrix Convention" và "Notation" với công thức X ∈ ℝ^{B×D}, Y ∈ ℝ^{B×K}.

Khi huấn luyện mô hình, dữ liệu được đưa vào theo **lô (batch)**. Hai ma trận chính:

$$\mathbf{X} \in \mathbb{R}^{B \times D} \quad \text{(hàng } n \text{ là } \mathbf{x}_n^T\text{)}$$

$$\mathbf{Y} \in \mathbb{R}^{B \times K}$$

trong đó:
- $B$ = batch size
- $D$ = số chiều đặc trưng đầu vào
- $K$ = số chiều đầu ra (hoặc số lớp)

**Quy tắc nhớ:** Mọi vector đều là cột; khi stack thành batch thì lấy transpose → xếp thành hàng.

---

## 7. Mục Tiêu Học (Learning Objective)

> 📌 **Ảnh:** Slide trang 10 — `01-ProblemFormulation.pdf#page=10`
> Hộp "General Learning Objective" với công thức θ* = argmin_θ L(f_θ(X), Y).

Mọi thuật toán machine learning đều quy về một bài toán tối ưu hóa:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \mathcal{L}\!\left(f_{\boldsymbol{\theta}}(\mathbf{X}),\, \mathbf{Y}\right)$$

**Đọc:** *Tìm bộ tham số $\boldsymbol{\theta}^*$ sao cho hàm mất mát $\mathcal{L}$ (đo sự sai lệch giữa dự đoán và nhãn thực tế) là nhỏ nhất.*

> 💡 Các bài toán khác nhau dùng hàm mất mát $\mathcal{L}$ khác nhau, nhưng tất cả **chia sẻ cùng nguyên lý học** này.

---

## 8. Phân Phối Dữ Liệu và Khái Quát Hóa

> 📌 **Ảnh:** Slide trang 11 — `01-ProblemFormulation.pdf#page=11`
> Hộp "Underlying Data Distribution" với (x,y)∼P(x,y) và Expected Risk; hộp "What We Actually Minimize" với empirical risk (1/N)∑ℓ.

Đây là một trong những khái niệm **sâu sắc nhất** mà nhiều người học xong vẫn chưa thực sự hiểu.

### Giả Thiết Nền Tảng

Mọi mẫu dữ liệu đều được lấy mẫu từ một *phân phối ẩn* $\mathcal{P}(\mathbf{x}, y)$:

$$(\mathbf{x}, y) \sim \mathcal{P}(\mathbf{x}, y)$$

### Mục Tiêu Thực Sự: Minimize Expected Risk

$$\mathbb{E}_{(\mathbf{x},y) \sim \mathcal{P}}\left[\ell\!\left(f_{\boldsymbol{\theta}}(\mathbf{x}),\, y\right)\right]$$

### Điều Chúng Ta Thực Sự Làm: Minimize Empirical Risk

Vì không biết $\mathcal{P}$ hoàn toàn, ta xấp xỉ bằng tập huấn luyện hữu hạn:

$$\frac{1}{N} \sum_{i=1}^{N} \ell\!\left(f_{\boldsymbol{\theta}}(\mathbf{x}_i),\, y_i\right)$$

> 💡 **Ẩn dụ:** Bạn học thi bằng đề cũ (empirical risk). Mục tiêu thực sự là làm tốt bài thi thật (expected risk). Học thuộc đề cũ mà không hiểu bản chất = **overfitting**. Khái quát hóa tốt = học được *quy luật*, không phải học thuộc *ví dụ*.

---

## 9. Phân Chia Tập Dữ Liệu

> 📌 **Ảnh:** Slide trang 12 — `01-ProblemFormulation.pdf#page=12`
> Hộp "Dataset Splitting" (Train/Val/Test) và "Important Rule" (no data leakage, test touched once).

Để đánh giá khả năng khái quát hóa, **không bao giờ** đánh giá mô hình trên dữ liệu nó đã học.

| Tập | Vai trò |
|---|---|
| **Training set** | Học (cập nhật) tham số mô hình |
| **Validation set** | Tune hyperparameter, early stopping, chọn kiến trúc |
| **Test set** | Đánh giá hiệu năng **cuối cùng, một lần duy nhất** |

### Quy Tắc Bất Di Bất Dịch

- **Test data TUYỆT ĐỐI KHÔNG ĐƯỢC** dùng trong huấn luyện hay điều chỉnh
- Tránh **data leakage** — thông tin từ test set "thấm" vào quá trình học

> ⚠️ *Test set chỉ được chạm vào đúng một lần.* Nếu bạn nhìn kết quả test set nhiều lần để quyết định thay đổi gì, bạn đã biến nó thành validation set — kết quả không còn đáng tin cậy.

---

## 10. Không Gian Đầu Ra (Output Space)

> 📌 **Ảnh:** Slide trang 13 — `01-ProblemFormulation.pdf#page=13`
> Bảng "Regression" (scalar/vector output) và "Classification" (Binary/Multi-class/Multi-label) với câu kết "output space determines activation functions and loss functions".

| Bài toán | Không gian đầu ra | Activation đầu ra | Loss function |
|---|---|---|---|
| Regression (scalar) | $y \in \mathbb{R}$ | Linear (không có) | MSE |
| Regression (vector) | $\mathbf{y} \in \mathbb{R}^K$ | Linear | MSE |
| Binary classification | $y \in \{0, 1\}$ | **Sigmoid** | Binary Cross-Entropy |
| Multi-class (single-label) | $y \in \{1, \ldots, K\}$ | **Softmax** | Categorical Cross-Entropy |
| Multi-label | $\mathbf{y} \in \{0,1\}^K$ | **Sigmoid** (per-class) | Binary Cross-Entropy |

> 💡 **Nguyên tắc vàng:** *Output space quyết định activation function và loss function.* Đây là lý do tại sao bước phát biểu bài toán rõ ràng phải làm **trước khi** viết bất kỳ dòng code nào.

---

## 11. Mô Hình Tham Số và Inductive Bias

> 📌 **Ảnh:** Slide trang 14 — `01-ProblemFormulation.pdf#page=14`
> Hộp "Parametric Model" (f_θ(x), θ = W, b) và "Inductive Bias" (examples: Linear/CNN/Transformer).

### Mô Hình Tham Số (Parametric Model)

$$f_{\boldsymbol{\theta}}(\mathbf{x})$$

- $\boldsymbol{\theta}$: bộ tham số học được — gồm **weights $\mathbf{W}$** và **bias $\mathbf{b}$**
- Huấn luyện = tìm $\boldsymbol{\theta}$ tốt nhất

### Thiên Kiến Quy Nạp (Inductive Bias)

Để học từ dữ liệu hữu hạn và khái quát hóa, mô hình **phải có** một số giả định tiên nghiệm về cấu trúc của thế giới:

| Kiến trúc | Inductive Bias |
|---|---|
| **Linear model** | Quan hệ đầu vào–đầu ra là **tuyến tính** |
| **CNN** | Đặc trưng quan trọng mang tính **cục bộ** (locality) và **bất biến tịnh tiến** (translation invariance) |
| **Transformer** | Các phần tử tương tác nhau qua cơ chế **attention** |

> 💡 **No Free Lunch Theorem:** *Learning is impossible without some form of inductive bias.* Lý do ta chọn CNN cho ảnh, Transformer cho văn bản không phải ngẫu nhiên — inductive bias của chúng *phù hợp* với cấu trúc vốn có của từng loại dữ liệu.

---

## 12. Hàm Mất Mát (Loss Functions)

> 📌 **Ảnh:** Slide trang 15 — `01-ProblemFormulation.pdf#page=15`
> Hộp "Regression" với MSE: ℓ(y, ŷ) = (y−ŷ)²; hộp "Classification" với Binary/Categorical Cross-Entropy.

**Hàm mất mát** định nghĩa *thế nào là một dự đoán tốt*.

### Regression — Mean Squared Error (MSE)

$$\ell(y, \hat{y}) = (y - \hat{y})^2$$

Phạt nặng những sai số lớn (do bình phương). Phù hợp khi muốn tránh các sai lệch cực đoan.

### Classification — Cross-Entropy

- **Binary Cross-Entropy:** Dùng cho binary hoặc multi-label (sigmoid đầu ra)

$$\ell(y, \hat{p}) = -[y \log \hat{p} + (1-y) \log(1-\hat{p})]$$

- **Categorical Cross-Entropy:** Dùng cho multi-class single-label (softmax đầu ra)

$$\ell(y, \hat{\mathbf{p}}) = -\sum_{k=1}^{K} \mathbf{1}[y=k] \log \hat{p}_k$$

> 💡 *Loss function định nghĩa điều bạn muốn tối ưu. Chọn sai loss = chọn sai mục tiêu huấn luyện = mô hình học được điều bạn không mong muốn.*

---

## Tóm Tắt Bài 01

| Khái niệm | Nội dung cốt lõi |
|---|---|
| **Notation** | Vector: **x** (chữ thường đậm); Matrix: **X** (chữ hoa đậm); Batch: stack $\mathbf{x}_i^T$ as rows |
| **ML Problem** | Học $f: \mathbf{x} \mapsto y$ từ $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ |
| **Classification** | $y \in \{1,...,K\}$; binary / multi-class / multi-label |
| **Regression** | $y \in \mathbb{R}$ hoặc $\mathbf{y} \in \mathbb{R}^K$ |
| **Learning Objective** | $\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \mathcal{L}(f_{\boldsymbol{\theta}}(\mathbf{X}), \mathbf{Y})$ |
| **Generalization** | Minimize empirical risk ≈ minimize expected risk |
| **Data Splits** | Train (học params) → Val (tune) → Test (báo cáo, 1 lần duy nhất) |
| **Output Space** | Quyết định activation function và loss function |
| **Inductive Bias** | CNN → locality; Transformer → attention; thiếu bias = không thể học |
| **Loss Functions** | MSE (regression); Cross-Entropy (classification) |
