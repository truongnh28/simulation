# Bài 07 — Model Capacity và Inductive Bias

> **Nguồn slide:** `slides-v1/foundation/07-ModelCapacity.pdf` — Thanh-Sach LE, HCMUT, VNU-HCM (05/02/2026)

---

## Mục lục

1. [Model Capacity của Linear Regression](#1-model-capacity-của-linear-regression)
2. [Model Capacity của Logistic & Softmax Regression](#2-model-capacity-của-logistic--softmax-regression)
3. [Overfitting, Underfitting và Bias–Variance Tradeoff](#3-overfitting-underfitting-và-biasvariance-tradeoff)
4. [Feature Transformation: The Missing Piece](#4-feature-transformation-the-missing-piece)
5. [Representation Learning](#5-representation-learning)

---

## Giới thiệu

Trong các bài trước, chúng ta đã học cách xây dựng mô hình (Linear Regression, Logistic Regression), định nghĩa hàm mất mát, và đo lường hiệu suất. Nhưng một câu hỏi nền tảng hơn vẫn còn bỏ ngỏ:

> **"Mô hình của tôi có thể học được gì? Và tại sao nó lại thất bại với một số bài toán?"**

Bài học này trả lời câu hỏi đó thông qua khái niệm **model capacity** — khả năng biểu diễn hàm số của một mô hình. Hiểu rõ capacity giúp chúng ta chẩn đoán vấn đề (underfitting hay overfitting?) và biết cần bổ sung gì để mô hình học tốt hơn.

---

## 1. Model Capacity của Linear Regression

### 1.1 Linear Regression học được gì?

Linear Regression dự đoán theo công thức:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

Đây là một **hàm affine** (tuyến tính cộng hằng số) của đầu vào $\mathbf{x}$. Dù bạn có thay đổi dữ liệu huấn luyện, điều chỉnh learning rate, hay chạy nhiều epochs đến đâu — đầu ra vẫn **luôn luôn là một hàm tuyến tính** của input.

**Hệ quả hình học:**
- Với input 1 chiều: đường dự đoán luôn là **đường thẳng**.
- Với input 2 chiều: bề mặt dự đoán luôn là một **mặt phẳng**.
- Với input $D$ chiều: dự đoán luôn là một **siêu phẳng (hyperplane)**.

### 1.2 Những gì Linear Regression KHÔNG học được

Vì bị ràng buộc bởi tính tuyến tính, Linear Regression không thể biểu diễn:

- **Hàm cong** như $\sin(x)$, $x^2$, $e^x$
- **Mẫu tuần hoàn** (periodic patterns)
- **Quan hệ phi tuyến từng mảnh** (piecewise nonlinear)

> **Phép ẩn dụ:** Linear Regression giống như một thợ vẽ chỉ được phép dùng thước kẻ. Dù anh ta có tài năng đến đâu, mọi đường anh ta vẽ đều thẳng — anh ta không thể vẽ được đường cong, dù đường cong đó có đơn giản đến mấy.

> **Điểm mấu chốt:** Hạn chế này **không phải** do hàm mất mát (MSE), không phải do thuật toán tối ưu, không phải do thiếu dữ liệu. Nguồn gốc của vấn đề là **lớp giả thuyết (hypothesis class)** — tập hợp các hàm mà mô hình có thể biểu diễn — và nó chỉ chứa các hàm tuyến tính.

> 📸 **[Cần ảnh]:** Hai đồ thị song song: (1) dữ liệu theo đường cong (sin, parabola) với đường thẳng của Linear Regression cố khớp — minh họa sự thất bại; (2) cùng dữ liệu nhưng sau feature transformation thành công. *(Trang 4–5 slide)*

---

## 2. Model Capacity của Logistic & Softmax Regression

### 2.1 Logistic Regression: Bộ phân loại tuyến tính nhị phân

Với $y \in \{0, 1\}$, Logistic Regression dự đoán:

$$\hat{p} = \sigma(\mathbf{w}^\top \mathbf{x} + b)$$

**Decision boundary** — ranh giới phân chia hai lớp — là tập hợp các điểm thỏa mãn $\hat{p} = 0.5$, tức là:

$$\mathbf{w}^\top \mathbf{x} + b = 0$$

Đây là một **siêu phẳng** trong không gian input. Logistic Regression **luôn luôn tạo ra decision boundary tuyến tính** — bất kể độ phức tạp của dữ liệu thực tế.

> **Ví dụ thực tế:** Bài toán XOR — hai lớp được phân bố theo dạng bàn cờ (điểm tại (0,0) và (1,1) là lớp A; điểm tại (1,0) và (0,1) là lớp B). Không có đường thẳng nào có thể phân chia chúng. Logistic Regression thất bại hoàn toàn với XOR.

### 2.2 Softmax Regression: Bộ phân loại tuyến tính đa lớp

Với $K$ lớp, Softmax Regression tính điểm số cho mỗi lớp:

$$z_k = \mathbf{w}_k^\top \mathbf{x} + b_k, \qquad k = 1, \ldots, K$$

**Decision boundary giữa lớp $i$ và lớp $j$** là:

$$(\mathbf{w}_i - \mathbf{w}_j)^\top \mathbf{x} + (b_i - b_j) = 0$$

Đây vẫn là một siêu phẳng. Hệ quả:

- Mỗi **vùng lớp (class region)** là một **đa diện lồi (convex polytope)** — tập hợp giao của các nửa không gian tuyến tính.
- Không thể biểu diễn các vùng lớp **cong** hay **không liên thông**.

> **Phép ẩn dụ:** Softmax Regression giống như phân chia một tờ bản đồ bằng những đường thẳng — mỗi quốc gia (lớp) là một vùng đa giác. Nhưng nếu "biên giới tự nhiên" là dòng sông cong (non-linear boundary), các đường thẳng không thể mô tả chính xác.

**Kết luận chung cho cả hai:**

> Logistic Regression và Softmax Regression đều là **bộ phân loại tuyến tính** trong không gian đặc trưng hiện tại. Chúng tổng quát hóa lẫn nhau (Logistic là trường hợp $K=2$ của Softmax), nhưng cùng chia sẻ hạn chế: **không thể học decision boundary phi tuyến**.

> 📸 **[Cần ảnh]:** (1) Logistic Regression với đường thẳng phân chia hai lớp vs dữ liệu hình tròn không tách được; (2) Softmax với các vùng đa giác lồi cho 3 lớp. *(Trang 9–13 slide)*

---

## 3. Overfitting, Underfitting và Bias–Variance Tradeoff

### 3.1 Underfitting — Mô hình quá đơn giản

**Underfitting** xảy ra khi mô hình **không đủ capacity** để nắm bắt cấu trúc thực sự trong dữ liệu.

**Dấu hiệu nhận biết:**
- **Training error cao** — ngay cả trên dữ liệu huấn luyện cũng sai nhiều.
- **Validation error cao** — tất nhiên cũng sai trên dữ liệu mới.

**Ví dụ điển hình:** Dùng Linear Regression để xấp xỉ một đường cong hình sin. Đường thẳng tốt nhất vẫn sẽ sai rất nhiều điểm.

### 3.2 Overfitting — Mô hình quá phức tạp

**Overfitting** xảy ra khi mô hình **quá nhiều capacity** đến mức học thuộc lòng cả nhiễu trong dữ liệu huấn luyện, thay vì học quy luật thực sự.

**Dấu hiệu nhận biết:**
- **Training error rất thấp** — mô hình "khớp hoàn hảo" với dữ liệu huấn luyện.
- **Validation error cao** — mô hình thất bại trên dữ liệu chưa thấy.

**Ví dụ điển hình:** Đa thức bậc 15 khớp qua tất cả 10 điểm huấn luyện — đường cong uốn lượn kỳ dị, dự đoán tệ ở mọi điểm khác.

> **Phép ẩn dụ:** Underfitting giống như học sinh không học bài, đi thi làm sai cả bài dễ. Overfitting giống như học sinh học vẹt đúng từng dòng đề cương nhưng không hiểu — gặp đề biến tấu một chút là không làm được.

**Mục tiêu:** Tìm một độ phức tạp mô hình "vừa đủ" — **tổng quát hóa tốt (good generalization)** sang dữ liệu chưa thấy.

> 📸 **[Cần ảnh]:** Ba đồ thị kinh điển cạnh nhau — underfitting (đường thẳng cho dữ liệu cong), just-right (đường cong khớp tốt), overfitting (đường uốn lượn qua từng điểm). *(Trang 15 slide; hoặc ảnh cực kỳ phổ biến trong mọi sách ML)*

---

### 3.3 Bias–Variance Tradeoff — Phân tích định lượng

Đây là một trong những kết quả lý thuyết quan trọng nhất trong học máy. Sai số dự đoán kỳ vọng có thể phân tích thành ba thành phần:

$$\mathbb{E}\left[(\hat{y} - y)^2\right] = \underbrace{\left(\mathbb{E}[\hat{y}] - y\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[(\hat{y} - \mathbb{E}[\hat{y}])^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}$$

Hãy hiểu từng thành phần:

#### Bias² — Sai lệch hệ thống

**Bias** đo lường sai lệch **trung bình** của mô hình so với giá trị thực, tính qua nhiều tập huấn luyện khác nhau.

- **Bias cao** → Mô hình có giả định sai (wrong assumptions) về dạng hàm cần học → **Underfitting**.
- **Ví dụ:** Dùng Linear Regression để học $y = x^2$. Dù có bao nhiêu dữ liệu, trung bình dự đoán vẫn là một đường thẳng — luôn lệch xa đường cong.

> **Phép ẩn dụ:** Bias giống như cây súng bị lệch nòng — dù bắn bao nhiêu viên, trung tâm các điểm chạm vẫn lệch khỏi tâm bia.

#### Variance — Sự nhạy cảm với dữ liệu

**Variance** đo lường mức độ **biến động** của dự đoán khi thay đổi tập huấn luyện.

- **Variance cao** → Mô hình quá nhạy cảm với tập huấn luyện cụ thể → **Overfitting**.
- **Ví dụ:** Đa thức bậc cao — mỗi lần thêm/bớt một điểm dữ liệu, đường cong thay đổi hoàn toàn.

> **Phép ẩn dụ:** Variance giống như tay súng run tay — mỗi lần bắn trúng một chỗ khác nhau, dù trung tâm có thể đúng.

#### Irreducible Noise $\sigma^2$

Là nhiễu ngẫu nhiên vốn có trong dữ liệu — không thể loại bỏ bằng bất kỳ mô hình nào. Đây là "sàn" của sai số.

#### Sự đánh đổi (The Tradeoff)

| Tăng model capacity | Bias | Variance |
|---------------------|------|----------|
| Mô hình đơn giản hơn | Tăng ↑ | Giảm ↓ |
| Mô hình phức tạp hơn | Giảm ↓ | Tăng ↑ |

**Đây là sự đánh đổi không thể tránh khỏi.** Không có mô hình nào vừa Bias = 0 vừa Variance = 0 (trừ khi bài toán quá đơn giản).

> **Insight sâu sắc:** Mục tiêu của regularization (L1, L2, Dropout...) chính là **kiểm soát Variance** bằng cách hy sinh một chút Bias — để tổng sai số kỳ vọng nhỏ nhất có thể.

> 📸 **[Cần ảnh]:** Đồ thị "U-shape" cổ điển — trục x là model complexity, trục y là error; đường Training Error giảm dần, đường Validation Error có dạng chữ U, điểm tối ưu ở đáy chữ U. *(Trang 15–16 slide; phổ biến trong mọi sách ML)*

> 📸 **[Cần ảnh]:** Hình 2×2 minh họa Bias/Variance qua phép ẩn dụ bắn bia: (1) Low Bias/Low Variance = lý tưởng, (2) High Bias/Low Variance = lệch nhưng tập trung, (3) Low Bias/High Variance = đúng trung bình nhưng phân tán, (4) High Bias/High Variance = tệ nhất. *(Cực kỳ phổ biến trong textbook ML)*

---

## 4. Feature Transformation: The Missing Piece

### 4.1 Tại sao mô hình tuyến tính thất bại?

Chúng ta đã xác định được vấn đề: **Linear/Logistic/Softmax Regression đều bị giới hạn bởi tính tuyến tính trong không gian đặc trưng hiện tại.**

Điều quan trọng cần nhấn mạnh:

> **Nút thắt cổ chai (bottleneck) KHÔNG phải là:**
> - Optimizer (SGD, Adam hay bất kỳ thuật toán tối ưu nào)
> - Kích thước dataset (thêm dữ liệu không giúp ích)
> - Hàm mất mát
>
> **Nút thắt cổ chai LÀ:** lớp giả thuyết (hypothesis class) — tập hợp các hàm mà mô hình có khả năng biểu diễn. Với mô hình tuyến tính, lớp này chỉ chứa các hàm tuyến tính → **cần thay đổi không gian đặc trưng**.

### 4.2 Giải pháp: Feature Mapping $\Phi$

**Ý tưởng cốt lõi:** Thay vì áp mô hình tuyến tính trực tiếp lên $\mathbf{x}$, ta ánh xạ $\mathbf{x}$ vào một không gian mới trước:

$$\Phi : \mathbb{R}^D \rightarrow \mathbb{R}^M$$

Rồi áp mô hình tuyến tính trong không gian mới:

$$\hat{y} = \mathbf{w}^\top \Phi(\mathbf{x}) + b$$

**Tại sao điều này hoạt động?**
- Mô hình vẫn **tuyến tính theo tham số** $\mathbf{w}$ → gradient descent vẫn hoạt động tốt.
- Nhưng mô hình có thể **phi tuyến theo đầu vào** $\mathbf{x}$ → có thể học các quan hệ phức tạp.

> **Ví dụ đơn giản:** $\mathbf{x} = [x_1]$, $\Phi(\mathbf{x}) = [x_1, x_1^2, x_1^3]$. Mô hình $\hat{y} = w_1 x_1 + w_2 x_1^2 + w_3 x_1^3 + b$ vẫn tuyến tính theo $(w_1, w_2, w_3, b)$ nhưng là đa thức bậc 3 theo $x_1$.

> **Phép ẩn dụ:** Hãy tưởng tượng dữ liệu là các hạt đậu rải trên mặt bàn — đỏ và xanh xen lẫn nhau theo vòng tròn, không thể tách bằng đường thẳng. Nếu bạn **nhấc một nhóm hạt lên (transform sang không gian 3D)**, thì một mặt phẳng có thể phân chia chúng hoàn toàn. Feature transformation chính là "nhấc hạt lên" theo nghĩa toán học.

### 4.3 Pipeline tổng quát

```
X (raw input) → Φ(X) (feature space H) → Linear Model → Ŷ
```

$$X \xrightarrow{\Phi} H = \Phi(X) \in \mathbb{R}^{N \times M} \xrightarrow{\text{Linear}} \hat{Y}$$

Trong đó:
- **$X$**: dữ liệu thô (ảnh pixel, text, số đo...)
- **$\Phi$**: bộ biến đổi đặc trưng (feature transformer)
- **$H$**: biểu diễn mới (representation / embedding)
- **Linear head**: Linear/Logistic/Softmax Regression

### 4.4 Ba gia đình Feature Transformation

#### 1. Manual Feature Transform (Thủ công)

Con người thiết kế $\Phi$ dựa trên kiến thức miền:

- **Polynomial features:** $[x, x^2, x^3, x_1 x_2, ...]$ — mở rộng không gian đa thức
- **Fourier basis:** $[\sin(x), \cos(x), \sin(2x), ...]$ — phân tích tần số
- **RBF (Radial Basis Functions):** $[\exp(-\|x - c_1\|^2), ...]$ — đặc trưng dựa trên khoảng cách
- **Kernel methods:** $\Phi$ ngầm định thông qua kernel trick (SVM)

**Hạn chế:** Cần kiến thức chuyên sâu về bài toán. Không tổng quát hóa tốt sang bài toán mới.

#### 2. Shallow Learned Features (Học nông)

**MLP (Multi-Layer Perceptron):** Dùng mạng neural nông (vài lớp Fully Connected + hàm kích hoạt phi tuyến) làm $\Phi$.

$$H = \text{Nonlinearity}(\text{FC}(\text{Nonlinearity}(\text{FC}(X))))$$

**Ưu điểm:** Tự học $\Phi$ từ dữ liệu mà không cần thiết kế tay.

#### 3. Deep Learned Features (Học sâu — trọng tâm môn học này!)

Mạng neural sâu với các kiến trúc chuyên biệt cho từng loại dữ liệu:

| Kiến trúc | Năm | Loại dữ liệu chuyên biệt |
|-----------|-----|--------------------------|
| **CNN** | 2012 | Ảnh (spatial structure) |
| **RNN/LSTM** | ~2015 | Chuỗi thời gian, text |
| **GNN** | ~2017 | Đồ thị (graph) |
| **Transformer** | 2017 | Text, ảnh, đa phương thức |
| **Mamba/SSM** | 2023– | Chuỗi dài, thay thế Transformer |

> **Key Insight:** Sự khác biệt giữa các mô hình hiện đại (ResNet, BERT, GPT, ViT...) chủ yếu nằm ở **cách thiết kế $\Phi$**. Phần "đầu" (head) của chúng thường vẫn chỉ là một linear classifier/regressor đơn giản!

---

## 5. Representation Learning

### 5.1 Ý tưởng cốt lõi

**Representation Learning** là hướng tiếp cận trong đó mô hình **tự học** cách biến đổi dữ liệu thô thành biểu diễn hữu ích, thay vì được con người thiết kế thủ công.

Pipeline thống nhất:

$$\mathbf{x} \xrightarrow{\Phi} \text{representation} \xrightarrow{g(\cdot)} \hat{y}$$

- **$\Phi$:** Bộ học biểu diễn (feature / representation transformer) — phức tạp, học được từ dữ liệu
- **$g(\cdot)$:** Bộ dự đoán đơn giản (thường là linear)

### 5.2 Trước và Sau Deep Learning

#### Trước Deep Learning: Feature Engineering thủ công

Con người phải thiết kế đặc trưng dựa trên kiến thức chuyên sâu về bài toán:

- **SIFT (Scale-Invariant Feature Transform):** Đặc trưng điểm đặc biệt trong ảnh, bất biến với xoay và scale — cần hàng năm nghiên cứu để phát triển.
- **HOG (Histogram of Oriented Gradients):** Mô tả hướng gradient cục bộ trong ảnh — dùng cho nhận dạng người.
- **Polynomial features, domain-specific descriptors:** Cần chuyên gia từng lĩnh vực (y tế, tài chính, khí tượng...).

**Hạn chế:** Tốn công, không tổng quát, phụ thuộc vào chuyên gia.

#### Với Deep Learning: Học đặc trưng tự động

Mạng neural học $\Phi$ trực tiếp từ dữ liệu thô — pixel thô của ảnh, ký tự thô của text.

> **Đây là cuộc cách mạng của Deep Learning:** Không phải thuật toán tối ưu tốt hơn, không phải mạng to hơn — mà là việc **thay thế feature engineering thủ công bằng representation learning tự động**.

### 5.3 Mạng sâu = Xếp chồng các biến đổi biểu diễn

$$\Phi(\mathbf{x}) = f_L(\cdots f_2(f_1(\mathbf{x})))$$

Mỗi lớp $f_l$ học một biến đổi biểu diễn từ đầu ra của lớp trước:

- **Lớp đầu:** Học các đặc trưng cấp thấp (cạnh, màu sắc, texture trong ảnh)
- **Lớp giữa:** Học các đặc trưng cấp trung (hình dạng, bộ phận)
- **Lớp cuối:** Học các đặc trưng cấp cao (khái niệm, ngữ nghĩa)

**Từng kiến trúc học loại biểu diễn chuyên biệt:**

| Kiến trúc | Loại biểu diễn |
|-----------|----------------|
| **CNN** | Biểu diễn không gian (spatial) — tốt cho ảnh |
| **RNN/LSTM** | Biểu diễn thời gian (temporal) — tốt cho chuỗi |
| **Transformer** | Biểu diễn ngữ cảnh (contextual) — tốt cho text, ảnh |
| **Mamba/SSM** | Biểu diễn trạng thái chuỗi (sequence state) |

> **Kết luận sâu sắc:**
>
> $$\boxed{\text{Deep Learning} = \text{Deep Representation Learning}}$$
>
> Học sâu không phải là "mạng to hơn" hay "nhiều dữ liệu hơn". Bản chất của nó là **tự động học cách biến đổi dữ liệu thô thành biểu diễn ngày càng trừu tượng và hữu ích hơn** qua từng lớp — để bộ phân loại/hồi quy tuyến tính đơn giản phía sau có thể làm tốt công việc của mình.

> 📸 **[Cần ảnh]:** Minh họa feature hierarchy trong CNN — lớp 1 học cạnh, lớp 2 học texture, lớp 3 học bộ phận, lớp 4 học vật thể. *(Visualizations nổi tiếng của Zeiler & Fergus 2013; hoặc trang 24–26 slide)*

---

## 6. Tổng kết và Bức tranh Toàn cảnh

### Những hạn chế bẩm sinh của mô hình tuyến tính

| Mô hình | Giới hạn |
|---------|---------|
| **Linear Regression** | Hàm dự đoán luôn là hyperplane trong input space |
| **Logistic Regression** | Decision boundary luôn là hyperplane (binary) |
| **Softmax Regression** | Class regions luôn là convex polytopes |

**Nguyên nhân chung:** Hypothesis class quá hẹp — chỉ chứa các hàm tuyến tính.

### Chuỗi logic của bài học

```
Mô hình tuyến tính bị giới hạn
    → Cần mở rộng hypothesis class
    → Giải pháp: Feature Transformation Φ
    → Φ thủ công (SIFT, HOG) → tốn công, không tổng quát
    → Φ học được (Neural Network) → Deep Learning
    → Deep Learning = học Φ tốt + linear head đơn giản
```

### Câu hỏi mở cho các bài tiếp theo

- **MLP** học $\Phi$ như thế nào? (Bài tiếp theo)
- Làm sao chống lại **Overfitting** khi $\Phi$ quá phức tạp? (Regularization)
- CNN/Transformer thiết kế $\Phi$ theo nguyên lý gì? (Sau này)

---

## Ghi chú về ảnh cần đính kèm

| # | Mô tả ảnh | Trang slide | Gợi ý nguồn |
|---|-----------|-------------|-------------|
| 1 | Linear Regression thất bại với dữ liệu phi tuyến (đường thẳng vs đường cong) | Trang 4–5 | Vẽ bằng `matplotlib` với dữ liệu `sin(x) + noise` |
| 2 | Logistic Regression thất bại với bài toán XOR / vòng tròn | Trang 9–10 | `sklearn` + `matplotlib`, dataset `make_circles` |
| 3 | Ba trường hợp: underfitting / just-right / overfitting | Trang 15 | Cổ điển — có sẵn trong docs của `sklearn` |
| 4 | Bias-Variance "bắn bia" 2×2 | Không có trong slide | Tìm "bias variance bullseye" |
| 5 | Đồ thị U-shape: training error vs validation error theo model complexity | Trang 15–16 | Vẽ tay hoặc `matplotlib` |
| 6 | Feature transformation: dữ liệu từ 2D không tách được → 3D tách được bằng mặt phẳng | Trang 18–20 | Tìm "kernel SVM feature map visualization" |
| 7 | Feature hierarchy của CNN (cạnh → texture → bộ phận → vật thể) | Trang 24–26 | Zeiler & Fergus (2013), hoặc bất kỳ DL textbook nào |
