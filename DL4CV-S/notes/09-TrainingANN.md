# Bài 09 — Thuật toán Huấn luyện Mạng Neural (Training Algorithms)

> **Nguồn slide:** `slides-v1/foundation/09-TrainingANN.pdf` — Thanh-Sach LE, HCMUT, VNU-HCM (05/02/2026)

---

## Mục lục

1. [Problem Setup — Hàm Mất Mát](#1-problem-setup--hàm-mất-mát)
2. [Training Process — Vòng lặp Huấn luyện](#2-training-process--vòng-lặp-huấn-luyện)
3. [Optimization — Các Thuật toán Tối ưu](#3-optimization--các-thuật-toán-tối-ưu)
4. [Training Techniques — Kỹ thuật Huấn luyện](#4-training-techniques--kỹ-thuật-huấn-luyện)
5. [Practical Considerations — Lưu ý Thực tế](#5-practical-considerations--lưu-ý-thực-tế)
6. [Tổng kết](#6-tổng-kết)
7. [Bài Tập Tính Toán](#7-bài-tập-tính-toán)

---

## Giới thiệu

Khi đã có kiến trúc MLP (hay bất kỳ mạng neural nào), câu hỏi tiếp theo là: **làm thế nào để tìm được bộ tham số $\theta = \{W, b\}$ tốt?**

Đây là bài toán tối ưu hóa — và nó không hề đơn giản. Không gian tham số của một mạng neural hiện đại có thể lên đến hàng triệu, thậm chí hàng tỷ chiều. Không tồn tại nghiệm dạng đóng (closed-form solution) như Linear Regression. Ta phải dùng **phương pháp lặp dựa trên gradient**.

Bài học này trình bày hệ thống đầy đủ từ đặt vấn đề → vòng lặp huấn luyện → các thuật toán tối ưu → kỹ thuật hỗ trợ → lưu ý thực tế.

---

## 1. Problem Setup — Hàm Mất Mát

**Mục tiêu:** Tối thiểu hóa hàm mất mát $L(\theta)$ trên tập huấn luyện $\{(x_i, y_i)\}_{i=1}^n$.

### 1.1 Hàm mất mát cho Hồi quy

#### MSE — Mean Squared Error

$$L_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- **Ý nghĩa:** Phạt bình phương sai số — lỗi lớn bị khuếch đại
- **Lý tưởng khi** nhiễu có phân phối Gaussian $\mathcal{N}(0, \sigma^2)$
- **Nhạy cảm với outlier** do bình phương khuếch đại sai lệch lớn

#### MAE — Mean Absolute Error

$$L_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- **Ý nghĩa:** Phạt tuyến tính — bền vững hơn với outlier
- **Hạn chế:** Không khả vi tại $e = 0$ → gradient bị gián đoạn, hội tụ chậm hơn MSE

#### Huber Loss — "Lai giữa MSE và MAE"

$$L_\delta(e_i) = \begin{cases} \frac{1}{2}e_i^2 & \text{nếu } |e_i| \leq \delta \\ \delta(|e_i| - \frac{1}{2}\delta) & \text{nếu } |e_i| > \delta \end{cases}$$

với $e_i = y_i - \hat{y}_i$.

- **Vùng nhỏ** ($|e| \leq \delta$): hành xử như MSE — mượt, đạo hàm liên tục
- **Vùng lớn** ($|e| > \delta$): hành xử như MAE — ít nhạy cảm với outlier
- **Hyperparameter $\delta$** điều chỉnh điểm chuyển tiếp: $\delta$ nhỏ → gần MAE, $\delta$ lớn → gần MSE

> **Phép ẩn dụ:** Huber Loss giống một bộ giảm xóc thông minh — với va chạm nhỏ thì phản ứng mạnh (như lò xo MSE), với va chạm lớn thì hấp thụ êm hơn (như cao su MAE).

---

### 1.2 Hàm mất mát cho Phân loại Nhị phân

#### Binary Cross-Entropy (BCE)

$$L_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log p_i + (1 - y_i)\log(1 - p_i)\right]$$

với $p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$, $z_i$ là logit.

- **Ý nghĩa:** Tối đa hóa Bernoulli likelihood; phạt nặng dự đoán tự tin mà sai

**Vấn đề số học (numerical instability):** Khi $z_i$ rất lớn hoặc rất âm, $\sigma(z_i)$ có thể tràn số (overflow/underflow). Dùng **logits version** ổn định hơn:

$$L_{\text{BCE-Logits}} = \frac{1}{n}\sum_{i=1}^{n}\left[\max(0, z_i) - y_i z_i + \log(1 + e^{-|z_i|})\right]$$

> **Lưu ý thực hành:** Trong PyTorch, dùng `nn.BCEWithLogitsLoss` thay vì `nn.BCELoss` — hàm trước nhận logits và áp công thức ổn định bên trong.

**Các lưu ý thực tế với BCE:**
- **Mất cân bằng lớp (class imbalance):** Áp `pos_weight` để bù trừ lớp thiểu số
- **Ngưỡng $\tau$:** Mặc định 0.5, nhưng có thể tinh chỉnh qua ROC/PR curve
- **Regularization** (L2, Dropout) giúp ngăn overfitting

---

### 1.3 Hàm mất mát cho Phân loại Đa lớp

#### Categorical Cross-Entropy (CE)

$$L_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik}\log p_{ik} = -\frac{1}{n}\sum_{i=1}^{n}\log p_{i, y_i}$$

với $p_{ik} = \text{softmax}(z_i)_k$.

- **Ý nghĩa:** Tối đa hóa Multinoulli likelihood — chỉ xét xác suất của lớp đúng
- **Thực hành:** Luôn dùng **logits version** (Softmax + CE kết hợp) để ổn định số học

#### Label Smoothing

Thay vì one-hot $y_{i,y_i} = 1$, làm mềm nhãn:

$$y_{i, y_i} = 1 - \varepsilon, \quad y_{ik} = \frac{\varepsilon}{K-1} \text{ (các lớp khác)}$$

- **Mục đích:** Ngăn mô hình quá tự tin (overconfident) → cải thiện generalization
- **Điển hình:** $\varepsilon = 0.1$ — phân phối nhỏ sang các lớp sai

#### Focal Loss

$$L_{\text{Focal}} = -\frac{1}{n}\sum_{i=1}^{n}\alpha_{y_i}(1 - p_{i, y_i})^\gamma \log p_{i, y_i}$$

- **$(1 - p_{i,y_i})^\gamma$:** Hệ số giảm — khi mô hình **đã tự tin đúng** ($p$ cao), hệ số này nhỏ → loss nhỏ. Khi **mô hình không chắc** ($p$ thấp → mẫu khó), hệ số lớn → tập trung học mẫu khó.
- **$\alpha_k$:** Trọng số lớp — cân bằng mất cân bằng lớp
- **Ứng dụng nổi tiếng:** RetinaNet (object detection) — giải quyết mất cân bằng giữa background (dễ) và vật thể (khó, ít)

---

## 2. Training Process — Vòng lặp Huấn luyện

### 2.1 Tại sao không tối thiểu hóa trực tiếp?

- Hàm mất mát của mạng neural là **phi lồi, không gian chiều cao** → không có nghiệm dạng đóng
- **Giải pháp:** Phương pháp lặp dựa trên gradient — tiến dần đến cực tiểu bằng từng bước nhỏ

### 2.2 Ba bước cốt lõi

Mỗi iteration (bước) của vòng lặp huấn luyện gồm ba bước:

```
[1] Forward Pass → [2] Backward Pass → [3] Update Step
```

---

#### Bước 1: Forward Pass (Lan truyền thuận)

$$\hat{y}_i = f_\theta(x_i), \qquad L = \frac{1}{n}\sum_{i=1}^{n}\ell(y_i, \hat{y}_i)$$

- Đưa input qua mạng theo chiều thuận
- Tính loss để **đo khoảng cách** giữa dự đoán và nhãn thực
- **Lưu lại các activation trung gian** để dùng trong backward pass

#### Bước 2: Backward Pass — Chain Rule và Backpropagation

Mục tiêu: tính $\nabla_\theta L$ — gradient của loss theo **tất cả** tham số.

**Chain Rule (Quy tắc dây chuyền):** Với mỗi tham số $w$ trên đường từ input đến loss:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_1} \cdot \frac{\partial z_1}{\partial z_2} \cdots \frac{\partial z_k}{\partial w}$$

Mỗi nhân tử là **đạo hàm cục bộ** — gradient lan truyền ngược qua từng nút trong computational graph, nhân với đạo hàm cục bộ tại mỗi nút.

> **Phép ẩn dụ:** Hãy tưởng tượng một nhà máy dây chuyền — lỗi ở sản phẩm cuối (loss) được truy ngược về từng công đoạn (layer). Mỗi công đoạn biết mình đã "góp phần" bao nhiêu vào lỗi đó — đó chính là gradient.

**Frameworks làm điều này tự động qua autograd** (PyTorch, TensorFlow). Tuy nhiên hiểu chain rule giúp:
- Debug khi gradient bằng 0 hoặc vô cực
- Thiết kế custom layer với backward thủ công

> **Ví dụ minh họa từ slide:**
>
> Computational graph: $t = x_1 + x_2$, $u = F(t)$, $v = G(t)$, $m = w \cdot u$, $n = v + b$, $y = m + n$
>
> **Forward:** Tính lần lượt từ input → output, lưu cache cần thiết.
>
> **Backward đến $w$ và $b$:**
> $$\frac{\partial y}{\partial w} = \frac{\partial y}{\partial m} \cdot \frac{\partial m}{\partial w} = 1 \cdot u = u$$
> $$\frac{\partial y}{\partial b} = \frac{\partial y}{\partial n} \cdot \frac{\partial n}{\partial b} = 1 \cdot 1 = 1$$
>
> **Backward đến $x_1$** (đi qua hai đường: qua $F$ và qua $G$):
> $$\frac{\partial y}{\partial x_1} = w \cdot F'(t) + G'(t)$$

#### Bước 3: Update Step

Dùng gradient để cập nhật tham số. Ví dụ với SGD đơn giản:

$$\theta \leftarrow \theta - \eta \nabla_\theta L$$

Với $\eta$ là **learning rate** — kiểm soát độ lớn của bước cập nhật.

---

### 2.3 Pseudocode: Thuật toán SGD Huấn luyện

```
Input: Dataset {(xᵢ, yᵢ)}, model fθ, loss ℓ, learning rate η, epochs E, batch size B

1. Khởi tạo θ (Xavier/He initialization)
2. for epoch = 1 to E:
   a. Xáo trộn dataset, chia thành mini-batches kích thước B
   b. for mỗi mini-batch (X, y):
      i.  Forward:   Ŷ ← fθ(X)
      ii. Loss:      L ← ℓ(y, Ŷ)
      iii.Backward:  tính ∇θL
      iv. Update:    θ ← θ - η·∇θL
```

---

### 2.4 Thuật ngữ quan trọng

| Thuật ngữ | Định nghĩa |
|-----------|-----------|
| **Mini-batch** | Tập con nhỏ của training set dùng trong một forward–backward–update cycle. Cân bằng hiệu quả tính toán và độ ổn định gradient |
| **Epoch** | Một lượt đi qua toàn bộ training dataset (tất cả mini-batches) |
| **Iteration (step)** | Một lần cập nhật tham số dùng một mini-batch |
| **Learning rate $\eta$** | Kích thước bước cập nhật: quá lớn → phân kỳ, quá nhỏ → hội tụ chậm |
| **Momentum** | Kỹ thuật làm mượt update bằng cách tích lũy gradient quá khứ, giảm dao động |
| **LR Schedule** | Chiến lược điều chỉnh $\eta$ trong quá trình training (decay, cosine annealing...) |

---

## 3. Optimization — Các Thuật toán Tối ưu

### 3.1 Bảng tổng quan

| Optimizer | Hyperparams chính | Phù hợp | Hạn chế |
|-----------|-------------------|---------|---------|
| **SGD + Momentum** | $\eta$, $\mu$ | CNN, generalization tốt | Nhạy cảm với LR |
| **NAG** | $\eta$, $\mu$ | Convex, smooth | Cần tinh chỉnh thêm |
| **AdaGrad** | $\eta$, $\varepsilon$ | Sparse features, NLP | LR giảm đơn điệu |
| **RMSProp** | $\eta$, $\rho$, $\varepsilon$ | RNN, non-stationary | Cần schedule |
| **Adam** | $\eta$, $\beta_1$, $\beta_2$, $\varepsilon$ | Default cho nhiều task | Có thể overfit |
| **AdamW** | Adam + $\lambda$ | Transformer, vision hiện đại | Thêm $\lambda$ |
| **Adafactor** | schedule | Mô hình rất lớn (T5) | Ít đảm bảo lý thuyết |
| **LAMB** | AdamW + trust ratio | Large-batch pretraining | Nhiều hyperparams |
| **SAM** | Base opt + $\rho$ | Flat minima, generalization | 2× compute |
| **Lion** | $\eta$, $\beta_1$, $\beta_2$ | Vision, memory-light | Mới, chưa phổ biến |

---

### 3.2 SGD + Momentum

**Công thức:**

$$g_t = \nabla_\theta L_t(\theta), \qquad v_t = \mu v_{t-1} + g_t, \qquad \theta \leftarrow \theta - \eta v_t$$

- $v_t$: vector **momentum (vận tốc)** — tích lũy gradient theo thời gian
- $\mu$: hệ số momentum, thường $0.9$; khởi tạo $v_0 = 0$

**Tại sao cần Momentum?**

Hãy tưởng tượng tối ưu hóa trên một bề mặt địa hình: vanilla SGD lăn xuống theo gradient hiện tại — dễ bị mắc kẹt trong "khe hẹp" (dải hẹp với gradient mạnh theo một chiều, yếu theo chiều khác) và dao động qua lại.

Momentum **tích lũy quán tính**:
- **Giảm dao động** theo hướng dốc cao (gradient dao động → bù trừ nhau trong $v_t$)
- **Gia tốc** theo hướng thung lũng nông (gradient ổn định → tích lũy trong $v_t$)
- **Ổn định hơn** vanilla SGD trên bề mặt ill-conditioned

**Hyperparameters:**
- $\mu \in [0.8, 0.99]$, phổ biến $0.9$
- $\eta$: thử $10^{-3}$–$10^{-1}$, dùng cosine decay + warmup
- $\lambda$ (weight decay): thường $10^{-4}$ khi train CNN from scratch

**Lưu ý thực hành:** SGD + Momentum vẫn là **baseline mạnh cho CNN** — thường generalize tốt hơn Adam. Scale $\eta$ theo batch size (linear scaling rule khi tăng batch).

---

### 3.3 Nesterov Accelerated Gradient (NAG)

**Công thức:**

$$g_t = \nabla_\theta L\!\left(\theta - \eta\mu v_{t-1}\right), \qquad v_t = \mu v_{t-1} + g_t, \qquad \theta \leftarrow \theta - \eta v_t$$

**Sự khác biệt then chốt:** NAG tính gradient **tại vị trí "nhìn trước"** ($\theta - \eta\mu v_{t-1}$), không phải tại vị trí hiện tại.

**Tại sao Nesterov tốt hơn Momentum?**

Momentum "nhảy" theo quán tính trước, rồi mới hiệu chỉnh. NAG **nhìn trước rồi mới hiệu chỉnh** — giống như người chạy cẩn thận: nhìn xuống phía trước một bước trước khi đặt chân.

- Giảm overshoot so với classical momentum
- Hội tụ nhanh hơn trong bài toán lồi và mượt

**Lưu ý:** Nhạy cảm hơn với $\eta$; dùng trong vision tasks, đôi khi RNN training. Default: $\mu = 0.9$, cosine LR decay với warmup.

---

### 3.4 AdaGrad — Adaptive Gradient

**Công thức:**

$$G_t = G_{t-1} + g_t^2, \qquad \theta \leftarrow \theta - \frac{\eta}{\sqrt{G_t + \varepsilon}} g_t$$

- $G_t$: tổng tích lũy bình phương gradient (per-parameter), khởi tạo $G_0 = 0$
- Tham số nào nhận gradient lớn thường → bước update nhỏ đi; tham số ít update → bước update lớn hơn

**Tại sao AdaGrad ra đời?**

Trong NLP với embedding từ (sparse features), phần lớn từ xuất hiện rất ít trong từng batch — gradient chúng nhận được rất thưa. SGD dùng cùng một $\eta$ cho tất cả → từ hiếm gần như không học được. AdaGrad **tự động tăng learning rate cho tham số cập nhật ít**.

**Hạn chế nghiêm trọng:** $G_t$ chỉ tăng không bao giờ giảm → effective learning rate $\frac{\eta}{\sqrt{G_t}}$ **giảm đơn điệu về 0** → mạng ngừng học sau đủ nhiều bước. Thực tế hiếm dùng cho deep networks; bị thay thế bởi RMSProp/Adam.

---

### 3.5 RMSProp — Fixing AdaGrad

**Công thức:**

$$s_t = \rho s_{t-1} + (1 - \rho)g_t^2, \qquad \theta \leftarrow \theta - \frac{\eta}{\sqrt{s_t + \varepsilon}} g_t$$

- $s_t$: **trung bình động có trọng số hàm mũ** (exponentially decayed moving average) của bình phương gradient — khác AdaGrad ở chỗ "quên" gradient cũ dần dần
- $\rho$: decay rate (forgetting factor), thường $0.9$

**Tại sao RMSProp tốt hơn AdaGrad?**

Thay vì tích lũy mãi mãi, $s_t$ chỉ nhớ gradient gần đây (thông qua $\rho$) → effective learning rate không về 0. Phù hợp cho bài toán **non-stationary** (phân phối dữ liệu thay đổi theo thời gian) như RNN training.

**Hyperparameters:** $\eta = 10^{-3}$, $\rho = 0.9$, $\varepsilon = 10^{-8}$.

---

### 3.6 Adam — Adaptive Moment Estimation

Adam kết hợp **Momentum** (first moment) và **RMSProp** (second moment):

**Công thức đầy đủ:**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t \qquad \text{(first moment — mean)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2 \qquad \text{(second moment — variance)}$$

**Bias correction** (quan trọng ở đầu training khi $m_0 = v_0 = 0$):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update:**

$$\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

**Tại sao cần bias correction?** Ở bước đầu ($t=1$), $m_1 = (1-\beta_1)g_1$ — rất nhỏ vì $\beta_1 = 0.9$ → chia cho $(1-0.9^1) = 0.1$ → scale lên đúng giá trị thực. Không có bias correction, những bước đầu sẽ quá nhỏ.

**Hyperparameters mặc định:**
- $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$

**Adam = Momentum + RMSProp + Bias Correction** → hội tụ nhanh, ổn định, ít tinh chỉnh.

**Hạn chế:** Có thể overfit hoặc generalize kém hơn SGD nếu không có weight decay.

---

### 3.7 AdamW — Adam với Decoupled Weight Decay ⭐

**Vấn đề của Adam + L2 regularization:**

Trong Adam, nếu thêm L2 vào loss ($L_{\text{total}} = L + \frac{\lambda}{2}\|\theta\|^2$), gradient trở thành $g_t + \lambda\theta$. Khi chia cho $\sqrt{\hat{v}_t}$ (adaptive scale), weight decay cũng bị scale → **weight decay không hoạt động đúng như kỳ vọng**.

**Giải pháp — Tách riêng weight decay:**

$$\theta \leftarrow \underbrace{(1 - \eta\lambda)\theta}_{\text{weight decay}} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

Weight decay $\lambda$ được áp **trực tiếp lên $\theta$**, không qua adaptive scaling → regularization hoạt động đúng.

**Hyperparameters:**
- $\eta$: $10^{-3}$ (general), $2 \times 10^{-4}$–$5 \times 10^{-5}$ (transformers)
- $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$
- $\lambda = 0.01$ cho mô hình lớn

**AdamW là optimizer mặc định cho Transformer (BERT, GPT, ViT) và backbone vision hiện đại.** Luôn kết hợp với cosine LR schedule + warmup.

---

### 3.8 Các Optimizer đặc biệt khác

#### Nadam
- Áp Nesterov lookahead lên Adam's momentum
- Đôi khi nhanh hơn Adam một chút; dùng hyperparams của Adam

#### Adafactor (Memory-Efficient)
- Xấp xỉ second moment bằng cách factor hóa (row/col statistics) → tiết kiệm bộ nhớ đáng kể
- Dùng trong training mô hình cực lớn (T5, Flan-T5)
- Thường dùng với LR schedule (không có fixed $\eta$)

#### LAMB (Large-Batch Training)
- AdamW + trust ratio (tỷ lệ tin tưởng theo từng layer)
- Ổn định training với batch size 1k–64k
- Dùng trong pretraining quy mô lớn (BERT from scratch)

#### SAM — Sharpness-Aware Minimization
- **Ý tưởng:** Tìm cực tiểu **bằng phẳng (flat minima)** thay vì cực tiểu nhọn → flat minima generalize tốt hơn (Bias-Variance: Variance thấp hơn)
- **Cơ chế:** Mỗi step có 2 pha: (1) perturbation lên worst-case trong neighborhood, (2) descent từ worst-case đó
- **Chi phí:** ~2× compute per step; bù lại bằng generalization tốt hơn
- Biến thể: ASAM, GSAM; có thể wrap around SGD hoặc AdamW

#### Lion (Sign-Based Momentum)
- Update dùng `sign(mt)` — chỉ lấy dấu của momentum
- Tiết kiệm bộ nhớ (không lưu second moment)
- Kết quả tốt trên vision tasks; còn mới

---

### 3.9 Hướng dẫn Chọn Optimizer

```
Bài toán là gì?
├─ CNN training from scratch → SGD + Momentum (generalization tốt nhất)
├─ Transformer / modern vision → AdamW + cosine schedule + warmup
├─ NLP với sparse features → AdaGrad hoặc RMSProp
├─ Large-batch pretraining (batch > 1k) → LAMB
├─ Muốn flat minima / robust → SAM (wrap base optimizer)
├─ Mô hình cực lớn, bộ nhớ hạn chế → Adafactor
└─ Thử nghiệm mới / vision → Lion
```

**Practical tips:**
1. Tune $\eta$ trước, sau đó điều chỉnh decay/momentum
2. **Schedule luôn quan trọng:** cosine annealing, step decay, one-cycle; luôn thử warmup
3. Log training curves và validate thường xuyên

---

## 4. Training Techniques — Kỹ thuật Huấn luyện

### 4.1 Learning Rate Scheduling

**Ý tưởng:** Điều chỉnh $\eta$ động trong quá trình training để cân bằng **khám phá (exploration)** vs **hội tụ (convergence)**.

#### Step Decay
Giảm $\eta$ theo hệ số cố định sau mỗi $k$ epochs:

$$\eta \leftarrow \eta \times \text{factor} \quad \text{sau mỗi } k \text{ epochs}$$

Phổ biến trong CNN training truyền thống. Đơn giản nhưng thô.

#### Cosine Annealing
$$\eta_t = \frac{1}{2}\eta_0\left(1 + \cos\!\left(\frac{t}{T}\pi\right)\right)$$

- $T$: tổng số steps, $\eta_0$: learning rate ban đầu
- Giảm mượt từ $\eta_0$ về 0 theo đường cong cosine
- **Kết hợp warmup:** Vài epoch đầu tăng $\eta$ từ 0 lên $\eta_0$, sau đó mới cosine decay

#### Warm Restarts (SGDR)
Định kỳ reset $\eta$ lên cao → giúp **thoát khỏi cực tiểu nhọn** và tìm cực tiểu bằng phẳng hơn.

> **Tại sao warmup quan trọng?** Ở những bước đầu, tham số chưa ổn định — một $\eta$ lớn ngay từ đầu có thể gây phân kỳ. Warmup cho phép mạng "định hướng" từ từ trước khi chạy với tốc độ đầy đủ.

> 📸 **[Cần ảnh]:** Đồ thị LR theo thời gian: (1) step decay — bậc thang, (2) cosine annealing — đường cong mượt, (3) cosine + warmup — tăng rồi cong xuống. *(Vẽ bằng matplotlib)*

---

### 4.2 Regularization: L2 (Weight Decay)

$$L_{\text{total}} = L + \frac{\lambda}{2}\|\theta\|_2^2 = L + \frac{\lambda}{2}\sum_j \theta_j^2$$

- **Cơ chế:** Phạt trọng số lớn → đẩy về 0
- **Tương đương với SGD:** $\theta \leftarrow (1 - \eta\lambda)\theta - \eta\nabla L$
- **Trong AdamW:** Tách weight decay khỏi gradient update (xem phần 3.7)
- **Điển hình:** $\lambda \in [10^{-4}, 10^{-2}]$

**Ý nghĩa thống kê:** L2 regularization tương đương với prior Gaussian trên trọng số trong framework Bayesian — tham số lớn có prior thấp, nên cần được "chứng minh" bởi dữ liệu.

---

### 4.3 Regularization: L1 (Lasso)

$$L_{\text{total}} = L + \lambda\|\theta\|_1 = L + \lambda\sum_j |\theta_j|$$

- **Đặc điểm nổi bật:** Thúc đẩy **sparsity** — đẩy nhiều trọng số về **đúng 0** (không chỉ gần 0 như L2)
- **Ứng dụng:** Feature selection — những $\theta_j = 0$ nghĩa là feature $j$ không có giá trị
- **Trong deep learning:** Ít phổ biến hơn L2; kết hợp L2 + Dropout thường được ưu tiên cho MLP/CNN

---

### 4.4 Regularization: Dropout

**Cơ chế:**
- **Training:** Tắt ngẫu nhiên mỗi neuron với xác suất $p$ (thường $p = 0.1$–$0.5$) → output neuron đó = 0
- **Inference:** Dùng tất cả neuron, nhân output với $(1-p)$ để cân bằng kỳ vọng

**Tại sao Dropout hiệu quả?**

1. **Chống co-adaptation:** Mỗi neuron không thể "dựa dẫm" vào những neuron cụ thể khác → buộc học đặc trưng độc lập, bền vững hơn
2. **Ensemble ngầm định:** Mỗi forward pass là một sub-network khác nhau (do mask ngẫu nhiên) → inference bằng toàn mạng ≈ lấy trung bình hàng ngàn sub-networks

> **Phép ẩn dụ:** Dropout giống như huấn luyện đội bóng bằng cách ngẫu nhiên cho nghỉ một số cầu thủ mỗi buổi — từng người phải tự lực, không ỷ lại vào đồng đội. Kết quả: cả đội mạnh hơn.

**Lưu ý hiện đại:** Transformer hiện đại ít dùng Dropout trong hidden layers, thay bằng LayerNorm + data augmentation mạnh.

---

### 4.5 Early Stopping

**Ý tưởng:** Dừng training khi **validation error không còn giảm** thay vì chạy đủ số epoch.

**Quy trình:**
1. Sau mỗi epoch, kiểm tra validation loss/accuracy
2. Nếu không cải thiện sau $p$ epochs (patience), dừng
3. Lưu **checkpoint** (best model weights) khi validation loss thấp nhất

**Tại sao hiệu quả?**
- Là dạng **implicit regularization** — ngăn mạng "học thuộc" quá mức training data
- Đơn giản, không cần hyperparameter phức tạp (chỉ cần patience)

> **Lưu ý:** Cần validation set thực sự độc lập. Không nên quyết định dừng dựa trên test set (data leakage).

---

## 5. Practical Considerations — Lưu ý Thực tế

### 5.1 Khởi tạo Trọng số (Weight Initialization)

Khởi tạo ngẫu nhiên thuần túy (e.g., $\mathcal{N}(0, 1)$) có thể gây ra vấn đề nghiêm trọng: activation của các lớp sâu bị **saturate** (sigmoid/tanh) hoặc **khuếch đại vô cực** (ReLU).

**Mục tiêu của khởi tạo tốt:** Giữ variance của activation **ổn định** qua các lớp — không bùng nổ, không biến mất.

#### Xavier / Glorot Initialization
$$W \sim \mathcal{U}\!\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\ \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]$$

- Dùng cho **Sigmoid** và **Tanh**
- Điều chỉnh variance theo cả số neuron vào ($n_{\text{in}}$) và ra ($n_{\text{out}}$)

#### He / Kaiming Initialization
$$W \sim \mathcal{N}\!\left(0,\ \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

- Dùng cho **ReLU** và biến thể
- Nhân thêm $\sqrt{2}$ vì ReLU "giết" một nửa activation (âm → 0), cần bù lại variance

> **Quy tắc:** Dùng PyTorch mặc định thường OK. Nhưng khi thiết kế custom architecture, hãy chú ý: ReLU → He init; Sigmoid/Tanh → Xavier init.

---

### 5.2 Vanishing và Exploding Gradients

Hai vấn đề cổ điển khi train mạng sâu:

#### Vanishing Gradients (Gradient biến mất)

Khi backprop qua nhiều lớp, gradient nhân liên tiếp với các Jacobian nhỏ hơn 1 (ví dụ đạo hàm của Sigmoid tối đa là 0.25) → gradient **teo dần về 0** → các lớp đầu không học được.

**Giải pháp:**
- Dùng **ReLU** thay Sigmoid (gradient = 1 trong vùng $z > 0$)
- **Residual connections** (skip connections) — gradient có đường tắt đi thẳng về lớp trước
- **Normalization layers** (BatchNorm, LayerNorm) — ổn định scale của activation

#### Exploding Gradients (Gradient bùng nổ)

Gradient nhân liên tiếp với các Jacobian lớn hơn 1 → gradient **tăng vô cực** → update step khổng lồ → divergence.

**Giải pháp:**
- **Gradient clipping:** Cap gradient nếu norm vượt ngưỡng:
  $$\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{\tau}{\max(\|\mathbf{g}\|, \tau)}$$
  Phổ biến trong RNN training (`nn.utils.clip_grad_norm_` trong PyTorch)
- **Careful initialization** (Xavier/He)

> 📸 **[Cần ảnh]:** Đồ thị gradient norm theo lớp: (1) vanishing — giảm về 0 ở lớp đầu, (2) exploding — tăng vô cực. *(Vẽ bằng matplotlib sau khi train một mạng)*

---

### 5.3 Batch Size Trade-off

| Batch Size | Ưu điểm | Nhược điểm |
|-----------|---------|-----------|
| **Nhỏ** (8–64) | Gradient "nhiễu" hơn → có thể escape local minima, generalize tốt hơn | Throughput thấp, training chậm |
| **Lớn** (1k+) | Throughput cao, gradient mượt hơn | Nguy cơ converge đến sharp minima → generalize kém hơn; cần scale LR |

**Linear Scaling Rule:** Khi tăng batch size $k$ lần, tăng $\eta$ $k$ lần — giữ nguyên "tổng lượng học" mỗi epoch. Tuy nhiên rule này chỉ là heuristic, cần kết hợp warmup để tránh instability đầu training.

---

## 6. Tổng kết

### Pipeline huấn luyện hoàn chỉnh

```
Data → [Forward Pass] → [Loss] → [Backward Pass] → [Update] → Repeat
         fθ(x) = ŷ      ℓ(y,ŷ)    ∇θL              θ ← θ - η·opt(∇θL)
```

### Checklist cho mỗi dự án

- [ ] **Loss phù hợp:** MSE/Huber (regression), BCE/CE (classification), Focal (imbalanced)
- [ ] **Optimizer:** AdamW là default tốt; SGD+Momentum nếu train CNN from scratch
- [ ] **LR Schedule:** Cosine annealing + warmup gần như luôn nên dùng
- [ ] **Regularization:** Weight decay ($\lambda$) + Dropout; Early stopping
- [ ] **Initialization:** He cho ReLU; Xavier cho Sigmoid/Tanh
- [ ] **Gradient health:** Monitor gradient norms; clip nếu cần
- [ ] **Batch size:** Tune LR theo batch size; nhỏ generalize tốt hơn

### Key Insights

> 1. **Ba bước cốt lõi không thể tách rời:** Forward → Backward → Update. Hiểu rõ mỗi bước giúp debug hiệu quả.
>
> 2. **Optimizer ≠ giải pháp tất cả:** Optimizer tốt chỉ giúp tìm cực tiểu nhanh hơn — nhưng nếu kiến trúc sai hoặc dữ liệu xấu, không optimizer nào cứu được.
>
> 3. **LR là hyperparameter quan trọng nhất.** Tune LR trước tất cả. Một LR schedule tốt thường quan trọng hơn việc chọn optimizer "fancy".
>
> 4. **Generalization ≠ Convergence.** SGD + Momentum thường converge chậm hơn Adam nhưng generalize tốt hơn — vì Adam hội tụ đến sharp minima, SGD hội tụ đến flat minima.

---

## 7. Bài Tập Tính Toán

> Tự làm trước khi mở đáp án.

---

### Bài 1 — Huber Loss

Cho $\delta = 1.0$. Tính Huber loss cho từng mẫu sau và so sánh với MSE, MAE:

| Mẫu | $y_i$ | $\hat{y}_i$ | $e_i$ | Huber | MSE term | MAE term |
|-----|--------|-------------|-------|-------|----------|----------|
| 1 | 3.0 | 3.5 | ? | ? | ? | ? |
| 2 | 2.0 | 4.0 | ? | ? | ? | ? |
| 3 | 5.0 | 8.5 | ? | ? | ? | ? |

**(a)** Điền vào bảng.

**(b)** Tính $L_{\text{Huber}}$, $L_{\text{MSE}}$, $L_{\text{MAE}}$ trung bình trên 3 mẫu.

**(c)** Mẫu 3 là outlier (lệch 3.5 đơn vị). So sánh đóng góp của nó vào ba loại loss. Rút ra nhận xét về ưu điểm Huber.

<details>
<summary>📋 Đáp án Bài 1</summary>

Công thức Huber ($\delta = 1.0$):
$$L_\delta(e) = \begin{cases} \frac{1}{2}e^2 & |e| \leq 1 \\ |e| - \frac{1}{2} & |e| > 1 \end{cases}$$

**(a) Bảng:**

| Mẫu | $e_i$ | Huber | MSE: $e^2$ | MAE: $|e|$ |
|-----|-------|-------|------------|------------|
| 1 | $-0.5$ | $\frac{1}{2}(0.5)^2 = \mathbf{0.125}$ | $0.25$ | $0.5$ |
| 2 | $-2.0$ | $2.0 - 0.5 = \mathbf{1.500}$ | $4.00$ | $2.0$ |
| 3 | $-3.5$ | $3.5 - 0.5 = \mathbf{3.000}$ | $12.25$ | $3.5$ |

**(b) Trung bình:**

$$L_{\text{Huber}} = \frac{0.125 + 1.5 + 3.0}{3} \approx \mathbf{1.542}$$
$$L_{\text{MSE}} = \frac{0.25 + 4.0 + 12.25}{3} = \frac{16.5}{3} \approx \mathbf{5.500}$$
$$L_{\text{MAE}} = \frac{0.5 + 2.0 + 3.5}{3} = \frac{6.0}{3} = \mathbf{2.000}$$

**(c) Đóng góp của outlier (mẫu 3):**

| Loss | Mẫu 3 đóng góp | Phần trăm tổng |
|------|----------------|----------------|
| MSE | $12.25/16.5$ | **74.2%** |
| MAE | $3.5/6.0$ | 58.3% |
| Huber | $3.0/4.625$ | 64.9% |

**Nhận xét:** MSE bị mẫu outlier (lệch 3.5) chiếm tới **74%** loss — do $3.5^2 = 12.25$ khuếch đại mạnh. Huber xử lý vùng $|e| > 1$ bằng phạt tuyến tính ($3.5 - 0.5 = 3.0$) thay vì $3.5^2 = 12.25$ → giảm ảnh hưởng outlier trong khi vẫn mượt hơn MAE ở vùng nhỏ (nhờ quadratic term).

</details>

---

### Bài 2 — SGD + Momentum: tính tay từng bước

Một tham số vô hướng $\theta$ với:
- Gradient tại 3 steps liên tiếp: $g_1 = 0.8$, $g_2 = 0.6$, $g_3 = -0.2$
- $\eta = 0.1$, $\mu = 0.9$, khởi tạo $\theta_0 = 1.0$, $v_0 = 0$

**(a)** Tính $v_1, v_2, v_3$ và $\theta_1, \theta_2, \theta_3$ theo SGD + Momentum.

**(b)** Tính lại bằng Vanilla SGD (không momentum). So sánh bước update tại step 2.

**(c)** Giả sử gradient dao động: $g_1 = 1.0$, $g_2 = -1.0$, $g_3 = 1.0$. Với $\mu = 0.9$, tính $v_1, v_2, v_3$. Tổng update $|\Delta\theta|$ sau 3 steps với momentum vs vanilla SGD là bao nhiêu?

<details>
<summary>📋 Đáp án Bài 2</summary>

Công thức: $v_t = \mu v_{t-1} + g_t$, $\theta_t = \theta_{t-1} - \eta v_t$

**(a) SGD + Momentum:**

| Step | $g_t$ | $v_t = 0.9 v_{t-1} + g_t$ | $\Delta\theta = -\eta v_t$ | $\theta_t$ |
|------|--------|--------------------------|--------------------------|-----------|
| 1 | 0.8 | $0.9(0) + 0.8 = 0.800$ | $-0.1 \times 0.800 = -0.080$ | $1.0 - 0.080 = \mathbf{0.920}$ |
| 2 | 0.6 | $0.9(0.8) + 0.6 = 1.320$ | $-0.1 \times 1.320 = -0.132$ | $0.920 - 0.132 = \mathbf{0.788}$ |
| 3 | -0.2 | $0.9(1.32) - 0.2 = 0.988$ | $-0.1 \times 0.988 = -0.099$ | $0.788 - 0.099 = \mathbf{0.689}$ |

**(b) Vanilla SGD** ($v_t = g_t$):

| Step | $g_t$ | $\Delta\theta = -\eta g_t$ | $\theta_t$ |
|------|--------|---------------------------|-----------|
| 1 | 0.8 | $-0.080$ | $0.920$ |
| 2 | 0.6 | $-0.060$ | $0.860$ |
| 3 | -0.2 | $+0.020$ | $0.880$ |

**So sánh tại step 2:**
- Vanilla SGD: update = $-0.060$
- Momentum: update = $-0.132$ (**2.2× lớn hơn**)

Momentum "nhớ" gradient step 1 ($g_1 = 0.8$) → tích lũy thêm vận tốc → bước lớn hơn khi gradient liên tục cùng chiều. Đây là "gia tốc theo thung lũng".

**(c) Gradient dao động** ($g = 1, -1, 1, ...$):

| Step | $g_t$ | $v_t^{mom}$ | Update momentum | Update vanilla |
|------|--------|-------------|-----------------|----------------|
| 1 | +1.0 | $0 + 1.0 = 1.000$ | $-0.100$ | $-0.100$ |
| 2 | -1.0 | $0.9(1.0) - 1.0 = -0.100$ | $+0.010$ | $+0.100$ |
| 3 | +1.0 | $0.9(-0.1) + 1.0 = 0.910$ | $-0.091$ | $-0.100$ |

Tổng $|\Delta\theta|$ sau 3 steps:
- **Vanilla SGD:** $0.1 + 0.1 + 0.1 = \mathbf{0.300}$
- **Momentum:** $0.1 + 0.01 + 0.091 = \mathbf{0.201}$

**Nhận xét:** Khi gradient dao động đổi dấu, momentum **giảm 33%** tổng update so với vanilla SGD — do $v_t$ tích lũy gradient đổi dấu nên bù trừ nhau, làm mượt dao động. Đây là lý do momentum ổn định hơn trên bề mặt hẹp (ill-conditioned).

</details>

---

### Bài 3 — Adam: tính bias correction

Một tham số, 2 bước đầu của Adam với:
- $\eta = 0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$
- $m_0 = 0$, $v_0 = 0$, $\theta_0 = 0.5$
- $g_1 = 0.3$, $g_2 = 0.5$

**(a)** Tính $m_1, v_1$, bias-corrected $\hat{m}_1, \hat{v}_1$, và $\theta_1$.

**(b)** Tính $m_2, v_2$, bias-corrected $\hat{m}_2, \hat{v}_2$, và $\theta_2$.

**(c)** Tại sao bias correction quan trọng? Nếu không có bias correction, $\theta_1$ sẽ là bao nhiêu? So sánh với kết quả đúng.

*(Cho: $\sqrt{9 \times 10^{-5}} \approx 0.00949$, $\sqrt{2.49 \times 10^{-4}} \approx 0.01578$)*

<details>
<summary>📋 Đáp án Bài 3</summary>

**(a) Bước 1** ($g_1 = 0.3$, $t = 1$):

$$m_1 = 0.9 \times 0 + 0.1 \times 0.3 = \mathbf{0.030}$$
$$v_1 = 0.999 \times 0 + 0.001 \times (0.3)^2 = 0.001 \times 0.09 = \mathbf{9 \times 10^{-5}}$$

**Bias correction:**
$$\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{0.030}{1 - 0.9} = \frac{0.030}{0.1} = \mathbf{0.300}$$
$$\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{9 \times 10^{-5}}{1 - 0.999} = \frac{9 \times 10^{-5}}{0.001} = \mathbf{0.090}$$

**Update:**
$$\theta_1 = \theta_0 - \eta \frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \varepsilon} = 0.5 - 0.01 \times \frac{0.300}{\sqrt{0.090}} = 0.5 - 0.01 \times \frac{0.300}{0.300} = 0.5 - 0.01 \times 1.0 = \mathbf{0.490}$$

**(b) Bước 2** ($g_2 = 0.5$, $t = 2$):

$$m_2 = 0.9 \times 0.030 + 0.1 \times 0.5 = 0.027 + 0.05 = \mathbf{0.077}$$
$$v_2 = 0.999 \times 9 \times 10^{-5} + 0.001 \times 0.25 = 8.991 \times 10^{-5} + 2.5 \times 10^{-4} \approx \mathbf{3.399 \times 10^{-4}}$$

**Bias correction ($t=2$):**
$$\hat{m}_2 = \frac{0.077}{1 - 0.9^2} = \frac{0.077}{0.19} \approx \mathbf{0.405}$$
$$\hat{v}_2 = \frac{3.399 \times 10^{-4}}{1 - 0.999^2} = \frac{3.399 \times 10^{-4}}{0.001999} \approx \mathbf{0.170}$$

**Update:**
$$\theta_2 = 0.490 - 0.01 \times \frac{0.405}{\sqrt{0.170}} = 0.490 - 0.01 \times \frac{0.405}{0.412} \approx 0.490 - 0.0098 \approx \mathbf{0.480}$$

**(c) Nếu không có bias correction tại bước 1:**

$$\theta_1^{no-bc} = 0.5 - 0.01 \times \frac{m_1}{\sqrt{v_1} + \varepsilon} = 0.5 - 0.01 \times \frac{0.030}{\sqrt{9 \times 10^{-5}}} = 0.5 - 0.01 \times \frac{0.030}{0.00949}$$
$$= 0.5 - 0.01 \times 3.16 = 0.5 - 0.0316 \approx \mathbf{0.468}$$

**So sánh:**
- Có bias correction: $\theta_1 = 0.490$ (update $-0.010$)
- Không bias correction: $\theta_1 = 0.468$ (update $-0.032$, lớn hơn **3.2×**)

Không có bias correction, bước đầu quá lớn vì $m_1 = 0.030$ nhỏ nhưng $v_1 = 9 \times 10^{-5}$ còn nhỏ hơn nhiều → tỉ lệ $m/\sqrt{v}$ bị thổi phồng. Bias correction giúp cả hai moment về đúng tầm — update step ổn định từ đầu.

</details>

---

### Bài 4 — L2 Regularization và AdamW Weight Decay

Một tham số $\theta = 2.0$, gradient $g = 0.4$, $\eta = 0.1$, $\lambda = 0.1$.

**(a)** Tính $\theta$ sau một bước **Vanilla SGD** (không regularization).

**(b)** Tính $\theta$ sau một bước **SGD + L2 regularization** (thêm $\lambda\theta$ vào gradient).

**(c)** Tính $\theta$ sau một bước **AdamW** (weight decay tách riêng):
$$\theta_{\text{new}} = (1 - \eta\lambda)\theta - \eta g$$

**(d)** Với Adam thường + L2 (gradient = $g + \lambda\theta$), giả sử adaptive scale $\frac{1}{\sqrt{\hat{v}}+\varepsilon} = 2.0$. Tính effective weight decay. So sánh với AdamW. Tại sao AdamW tốt hơn?

<details>
<summary>📋 Đáp án Bài 4</summary>

**(a) Vanilla SGD:**
$$\theta_1 = 2.0 - 0.1 \times 0.4 = 2.0 - 0.04 = \mathbf{1.960}$$

**(b) SGD + L2:**

Gradient hiệu dụng: $g' = g + \lambda\theta = 0.4 + 0.1 \times 2.0 = 0.6$

$$\theta_1 = 2.0 - 0.1 \times 0.6 = 2.0 - 0.06 = \mathbf{1.940}$$

Tương đương: $\theta_1 = (1 - \eta\lambda)\theta - \eta g = (1 - 0.01)(2.0) - 0.04 = 1.98 - 0.04 = 1.940$ ✓

**(c) AdamW:**
$$\theta_1 = (1 - \eta\lambda)\theta - \eta g = (1 - 0.1 \times 0.1)(2.0) - 0.1 \times 0.4$$
$$= (1 - 0.01)(2.0) - 0.04 = 0.99 \times 2.0 - 0.04 = 1.98 - 0.04 = \mathbf{1.940}$$

*(AdamW và SGD+L2 cho cùng kết quả vì đây là SGD — khác biệt xuất hiện khi dùng Adam)*

**(d) Adam thường + L2 (adaptive scale = 2.0):**

Gradient hiệu dụng = $g + \lambda\theta = 0.4 + 0.2 = 0.6$

Adam update: $\Delta\theta = -\eta \times \text{scale} \times (g + \lambda\theta) = -0.1 \times 2.0 \times 0.6 = -0.12$

→ Effective weight decay contribution: $-0.1 \times 2.0 \times \lambda\theta = -0.1 \times 2.0 \times 0.2 = -0.04$

**AdamW** weight decay: $-\eta\lambda\theta = -0.1 \times 0.1 \times 2.0 = -0.02$

**So sánh:**
- Adam + L2: weight decay = $-0.04$ (bị khuếch đại bởi adaptive scale $\times 2$)
- AdamW: weight decay = $-0.02$ (cố định, không phụ thuộc adaptive scale)

**Kết luận:** Trong Adam thường, weight decay bị scale adaptive làm cho **không nhất quán** — tham số nào có gradient nhỏ (scale lớn) sẽ bị regularize mạnh hơn, tham số có gradient lớn (scale nhỏ) bị regularize yếu hơn. AdamW tách weight decay ra ngoài, áp **đồng đều** lên tất cả tham số → L2 regularization hoạt động đúng kỳ vọng.

</details>

---

### Bài 5 — Cosine Annealing và Warmup

Training schedule:
- **Warmup:** 5 epochs đầu tăng tuyến tính từ $\eta = 0$ đến $\eta_0 = 0.01$
- **Cosine decay:** từ epoch 5 đến $T = 100$:

$$\eta_t = \frac{1}{2}\eta_0\left(1 + \cos\!\left(\frac{t - t_{warmup}}{T - t_{warmup}}\pi\right)\right)$$

**(a)** Tính $\eta$ tại các epoch sau warmup: $t = 5$ (bắt đầu decay), $t = 27.5$, $t = 52.5$, $t = 100$.

**(b)** Nếu thay cosine bằng step decay: giảm $\times 0.1$ sau mỗi 30 epochs, tính $\eta$ tại epoch 31, 61, 91.

**(c)** Batch size tăng từ 32 lên 256 (gấp 8 lần). Theo Linear Scaling Rule, $\eta_0$ nên điều chỉnh thành bao nhiêu? Tại sao cần warmup khi batch size lớn?

<details>
<summary>📋 Đáp án Bài 5</summary>

**(a) Cosine annealing** (từ $t_{warmup} = 5$ đến $T = 100$, $\eta_0 = 0.01$):

Sau warmup: $\eta_t = \frac{0.01}{2}\left(1 + \cos\!\left(\frac{t-5}{95}\pi\right)\right)$

| $t$ | $(t-5)/95$ | $\cos(\cdot \pi)$ | $\eta_t$ |
|-----|-----------|-----------------|---------|
| 5 | $0/95 = 0$ | $\cos(0) = 1$ | $\frac{0.01}{2}(1+1) = \mathbf{0.0100}$ |
| 27.5 | $22.5/95 \approx 0.237$ | $\cos(0.237\pi) \approx 0.707$ | $\frac{0.01}{2}(1.707) \approx \mathbf{0.00854}$ |
| 52.5 | $47.5/95 = 0.5$ | $\cos(0.5\pi) = 0$ | $\frac{0.01}{2}(1+0) = \mathbf{0.00500}$ |
| 100 | $95/95 = 1$ | $\cos(\pi) = -1$ | $\frac{0.01}{2}(1-1) = \mathbf{0.00000}$ |

**Nhận xét:** $t = 27.5$ (25% tiến trình) → LR giảm ~15%; $t = 52.5$ (50% tiến trình) → LR giảm 50%; cuối cùng về 0. Đường cong mượt và chậm lúc đầu, nhanh hơn ở giữa.

**(b) Step decay** ($\eta_0 = 0.01$, ×0.1 mỗi 30 epoch):

| Epoch | Số lần giảm | $\eta$ |
|-------|------------|--------|
| 31 | 1 lần (sau epoch 30) | $0.01 \times 0.1 = \mathbf{0.001}$ |
| 61 | 2 lần | $0.01 \times 0.01 = \mathbf{0.0001}$ |
| 91 | 3 lần | $0.01 \times 0.001 = \mathbf{0.00001}$ |

Step decay giảm **đột ngột** 10× mỗi milestone — có thể gây bất ổn nếu mô hình chưa hội tụ tốt trước điểm giảm.

**(c) Linear Scaling Rule:**

Batch size tăng $8\times$ (32 → 256):

$$\eta_0^{new} = \eta_0 \times 8 = 0.01 \times 8 = \mathbf{0.08}$$

**Lý do cần warmup với batch size lớn:**

Gradient với batch lớn có variance **nhỏ hơn** (ít nhiễu hơn) → mỗi step tin cậy hơn → có thể dùng LR lớn hơn. Nhưng ở những bước **đầu tiên**, tham số chưa ổn định — $\eta = 0.08$ ngay lập tức có thể gây phân kỳ (divergence). Warmup tăng dần từ 0 → 0.08 qua 5–10 epochs, cho phép mạng "định hướng" trước. Đây là lý do tại sao **batch size lớn cần warmup dài hơn** batch size nhỏ.

</details>

---

### Bài 6 — Gradient Clipping

Một mạng RNN tại một bước training, vector gradient có norm:

$$\|\mathbf{g}\| = 45.0$$

Ngưỡng clipping: $\tau = 5.0$.

**(a)** Sau clipping, norm của gradient mới là bao nhiêu? Viết công thức và tính hệ số scale.

**(b)** Nếu gradient gốc là $\mathbf{g} = [36, -27]$ (kiểm tra: $\|\mathbf{g}\| = \sqrt{36^2 + 27^2} = 45$), tính vector gradient sau clipping $\mathbf{g}_{clip}$.

**(c)** Gradient clipping thay đổi **hướng** của gradient không? Thay đổi **độ lớn** không? Điều này ảnh hưởng thế nào đến quá trình học?

**(d)** Nếu dùng clipping by value (clip từng chiều riêng: $g_i \leftarrow \text{clip}(g_i, -\tau, \tau)$ với $\tau = 5$), $\mathbf{g}_{clip}^{val}$ là bao nhiêu? So sánh với clipping by norm.

<details>
<summary>📋 Đáp án Bài 6</summary>

**(a) Norm sau clipping:**

$$\mathbf{g}_{clip} = \mathbf{g} \cdot \frac{\tau}{\max(\|\mathbf{g}\|, \tau)} = \mathbf{g} \cdot \frac{5.0}{45.0} = \mathbf{g} \cdot \frac{1}{9}$$

Norm mới: $\|\mathbf{g}_{clip}\| = 45.0 \times \frac{5}{45} = \mathbf{5.0}$ — đúng bằng ngưỡng $\tau$.

Hệ số scale: $\frac{1}{9} \approx 0.111$ — gradient được thu nhỏ **9 lần**.

**(b) Vector sau clipping** ($\mathbf{g} = [36, -27]$):

$$\mathbf{g}_{clip} = [36, -27] \times \frac{1}{9} = \mathbf{[4.0,\ -3.0]}$$

Kiểm tra: $\|(4, -3)\| = \sqrt{16+9} = \sqrt{25} = 5.0$ ✓

**(c) Hướng vs độ lớn:**

- **Hướng:** **KHÔNG thay đổi** — nhân cả vector với scalar dương $\frac{\tau}{\|\mathbf{g}\|}$ giữ nguyên hướng.
- **Độ lớn:** Có thay đổi — thu nhỏ về đúng $\tau$ nếu $\|\mathbf{g}\| > \tau$; không đổi nếu $\|\mathbf{g}\| \leq \tau$.

**Ảnh hưởng đến học:** Mô hình vẫn biết **đi theo hướng nào** (gradient direction đúng), chỉ là **bước đi nhỏ hơn** → tránh update quá lớn gây phân kỳ. Đây khác với zero-ing gradient (mất hoàn toàn tín hiệu).

**(d) Clipping by value** ($\tau = 5$):

$$g_1 = 36 \xrightarrow{clip} \min(36, 5) = \mathbf{5}$$
$$g_2 = -27 \xrightarrow{clip} \max(-27, -5) = \mathbf{-5}$$

$$\mathbf{g}_{clip}^{val} = [5,\ -5]$$

**So sánh:**

| | Clipping by Norm | Clipping by Value |
|--|-----------------|------------------|
| $\mathbf{g}_{clip}$ | $[4.0, -3.0]$ | $[5.0, -5.0]$ |
| Norm | $5.0$ | $\sqrt{50} \approx 7.07$ |
| Hướng | Giữ nguyên $(36:-27 = 4:-3)$ | **Thay đổi** — giờ là $(5:-5 = 1:-1)$ |

**Clipping by value thay đổi hướng gradient** — tỉ lệ giữa các chiều bị bóp méo. Clipping by norm an toàn hơn vì giữ hướng. Đây là lý do `clip_grad_norm_` trong PyTorch được dùng phổ biến hơn `clip_grad_value_`.

</details>

---

### Bài 7 — Lý luận chọn Optimizer và Hyperparameters

**(a)** Bạn đang train một ViT (Vision Transformer) từ đầu. Chọn optimizer và LR schedule phù hợp nhất, giải thích lý do.

**(b)** Mô hình huấn luyện 10 epochs đầu loss giảm tốt, sau đó chững lại hoàn toàn trong 20 epochs tiếp theo. Liệt kê ít nhất 3 nguyên nhân có thể và cách kiểm tra từng nguyên nhân.

**(c)** Validation loss sau mỗi epoch: `[2.1, 1.8, 1.5, 1.3, 1.2, 1.2, 1.25, 1.35, 1.4, ...]`. Với patience = 3, Early Stopping dừng ở epoch nào? Nên load checkpoint epoch nào?

**(d)** Hai kỹ thuật nào sau đây có thể **mâu thuẫn** nhau nếu dùng cùng lúc, và tại sao?
- (i) BatchNorm
- (ii) Dropout
- (iii) L2 Weight Decay
- (iv) Data Augmentation

<details>
<summary>📋 Đáp án Bài 7</summary>

**(a) ViT training từ đầu:**

**Chọn: AdamW + Cosine Annealing + Warmup**

- **AdamW:** Transformer được thiết kế với AdamW — attention weight cần adaptive learning rate, và weight decay cần tách riêng (không dùng L2 trong loss).
- **Cosine annealing:** LR giảm mượt, tốt hơn step decay cho training dài.
- **Warmup (5–10% tổng steps):** ViT không có inductive bias như CNN → khởi đầu gradient không ổn định → warmup giúp tránh phân kỳ sớm.
- **Điển hình:** $\eta_0 = 10^{-3}$ đến $10^{-4}$, $\lambda = 0.05$, warmup 10k steps.

**(b) Loss chững lại:**

| Nguyên nhân | Cách kiểm tra |
|------------|---------------|
| **Learning rate quá nhỏ** | Vẽ LR theo epoch; thử tăng LR hoặc dùng LR finder | 
| **Gradient vanishing** | Vẽ gradient norm theo layer; nếu layer đầu ≈ 0 → thêm skip connection hoặc đổi activation |
| **Dead neurons (ReLU)** | Kiểm tra % neurons có activation = 0 > 90% → đổi Leaky ReLU hoặc giảm LR |
| **Model đã fit tốt** (training loss = 0?) | So sánh training loss vs val loss — nếu training loss vẫn cao → underfitting, tăng capacity |
| **BatchNorm sai mode** | Kiểm tra `model.train()` vs `model.eval()` — BN dùng running stats khác batch stats |

**(c) Early Stopping với patience = 3:**

| Epoch | Val Loss | Best? | Không cải thiện |
|-------|----------|-------|----------------|
| 1 | 2.1 | ✓ best | 0 |
| 2 | 1.8 | ✓ best | 0 |
| 3 | 1.5 | ✓ best | 0 |
| 4 | 1.3 | ✓ best | 0 |
| 5 | 1.2 | ✓ best | 0 |
| 6 | 1.2 | = best (không giảm) | 1 |
| 7 | 1.25 | ✗ | 2 |
| 8 | 1.35 | ✗ | 3 ← **dừng!** |

**Dừng ở epoch 8** (sau 3 epoch không cải thiện từ epoch 5).

**Nên load checkpoint epoch 5** (val loss = 1.2, thấp nhất) — không phải epoch 8.

**(d) Mâu thuẫn tiềm tàng:**

**BatchNorm (i) và Dropout (ii) có thể mâu thuẫn:**

BatchNorm tính $\mu_B$ và $\sigma_B$ trên batch. Dropout ngẫu nhiên tắt một số neuron → output distribution của layer trước BN thay đổi ngẫu nhiên → **batch statistics không ổn định** (mean/variance tính trên subset neurons không tắt). Điều này làm BN hoạt động không đúng như thiết kế.

**Giải pháp thực hành:**
- Nếu dùng BN: đặt Dropout **sau** BN, hoặc dùng **rất ít** Dropout.
- Transformer hiện đại: dùng LayerNorm (không phụ thuộc batch) + Dropout nhẹ → ít vấn đề hơn.

*(L2 weight decay + Data Augmentation: không mâu thuẫn — bổ sung nhau tốt)*

</details>

---

### Tổng hợp công thức và mẹo thi

| Chủ đề | Công thức | Mẹo nhớ |
|--------|----------|---------|
| **Huber Loss** | $\frac{1}{2}e^2$ nếu $\|e\|\leq\delta$; $\delta(\|e\|-\frac{\delta}{2})$ nếu $\|e\|>\delta$ | MSE gần 0, MAE xa 0 |
| **SGD + Momentum** | $v_t = \mu v_{t-1} + g_t$; $\theta \leftarrow \theta - \eta v_t$ | $\mu = 0.9$; $v_0 = 0$ |
| **Adam m₁** | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ | First moment (mean) |
| **Adam v₁** | $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ | Second moment (var) |
| **Bias correction** | $\hat{m}_t = m_t/(1-\beta_1^t)$; $\hat{v}_t = v_t/(1-\beta_2^t)$ | Quan trọng ở bước đầu |
| **AdamW** | $(1-\eta\lambda)\theta - \eta\hat{m}/(\sqrt{\hat{v}}+\varepsilon)$ | WD tách khỏi adaptive |
| **Cosine anneal** | $\eta_t = \frac{\eta_0}{2}(1+\cos(t\pi/T))$ | Đầu = $\eta_0$, cuối = 0 |
| **Linear scaling** | $\eta \propto \text{batch size}$ | Double batch → double LR |
| **Grad clipping** | $\mathbf{g} \leftarrow \mathbf{g} \cdot \tau/\max(\|\mathbf{g}\|, \tau)$ | Giữ hướng, giảm độ lớn |
| **L2 effect** | $\theta \leftarrow (1-\eta\lambda)\theta - \eta g$ | "Shrinkage" mỗi bước |
| **Init loss** | $-\log(1/K) = \log K$ | CIFAR-10: 2.303; ImageNet: 6.908 |

**Quy trình debug training:**
```
Loss không giảm?
├── LR quá nhỏ? → Tăng LR hoặc dùng LR finder
├── LR quá lớn? → Loss dao động / NaN → Giảm LR, thêm warmup
├── Gradient vanish? → Kiểm tra grad norm per layer
├── Dead neurons? → Kiểm tra % ReLU output = 0
└── Bug trong code? → Overfit trên 1 batch trước (loss phải về 0)
```

## Ghi chú về ảnh cần đính kèm

| # | Mô tả ảnh | Trang slide | Gợi ý nguồn |
|---|-----------|-------------|-------------|
| 1 | Loss surface 3D — bề mặt không lồi với saddle points, local minima | Trang 5 | `matplotlib` 3D plot, tìm "loss landscape visualization" |
| 2 | Sơ đồ computational graph: nút, cạnh, chiều forward/backward | Trang 20–23 | Trang 20 slide; hoặc draw.io |
| 3 | Momentum vs SGD trên bề mặt "banana" — SGD dao động, Momentum đi thẳng | Trang 28–30 | Tìm "momentum optimization visualization" |
| 4 | Đồ thị LR theo thời gian: step decay, cosine, cosine+warmup | Trang 60 | `matplotlib` |
| 5 | Dropout mask: một số neuron bị tắt (xám) trong training | Trang 63 | Mọi DL textbook |
| 6 | Gradient norm theo lớp: vanishing (giảm) và exploding (tăng) | Trang 68 | Vẽ sau khi train thực nghiệm |
| 7 | Đồ thị training/validation loss với early stopping: điểm dừng tối ưu | Trang 65 | `matplotlib` với hai đường loss |
