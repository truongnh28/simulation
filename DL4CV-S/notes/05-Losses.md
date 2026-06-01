# Bài 05 — Hàm Mất Mát (Loss Functions)

> **Nguồn slide:** `slides-v1/foundation/05-Losses.pdf` — Thanh-Sach LE, HCMUT, VNU-HCM (05/02/2026)

---

## Mục lục

1. [Regression Loss](#1-regression-loss)
2. [Label Representation for Classification](#2-label-representation-for-classification)
3. [Classification Loss](#3-classification-loss)
4. [Tổng kết](#4-tổng-kết)
5. [Bài Tập Tính Toán](#5-bài-tập-tính-toán)

---

## Giới thiệu

Khi huấn luyện một mô hình học máy, ta cần một thước đo cụ thể để biết mô hình đang "sai bao nhiêu". Thước đo đó chính là **hàm mất mát (loss function)**. Hàm mất mát đóng vai trò như la bàn trong quá trình tối ưu hóa: nó chỉ hướng cho thuật toán gradient descent biết cần điều chỉnh tham số theo chiều nào để mô hình tiến dần đến dự đoán chính xác hơn.

Trong bài này, chúng ta sẽ khảo sát hai nhóm hàm mất mát chính:
- Nhóm dành cho **bài toán hồi quy (regression)**
- Nhóm dành cho **bài toán phân loại (classification)**

---

## 1. Regression Loss

### 1.1 Đặt vấn đề

Trong bài toán hồi quy, đầu ra của mô hình là một giá trị thực liên tục:

$$y \in \mathbb{R}, \quad \hat{y} = f(x)$$

Trong đó $y$ là **giá trị thực (ground truth)** và $\hat{y}$ là **giá trị dự đoán** của mô hình.

**Phần dư (residual)** được định nghĩa là:

$$e = y - \hat{y}$$

Mục tiêu của hàm mất mát hồi quy là **đo lường khoảng cách** giữa dự đoán $\hat{y}$ và giá trị thực $y$. Khoảng cách này càng nhỏ, mô hình càng tốt.

> **Phép ẩn dụ:** Hãy tưởng tượng bạn đang bắn cung. $y$ là tâm bia, $\hat{y}$ là điểm mũi tên chạm vào. Hàm mất mát chính là thước đo bạn trượt xa tâm bao nhiêu centimeter.

---

### 1.2 Mean Squared Error (MSE)

**Công thức:**

$$L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Tại sao bình phương?** Có hai lý do:

1. **Làm cho mọi sai số đều dương:** Phần dư $e = y - \hat{y}$ có thể âm hoặc dương. Nếu ta chỉ lấy trung bình các $e_i$ thẳng, chúng có thể triệt tiêu nhau và che giấu sai lầm thực sự. Bình phương đảm bảo mọi sai lệch đều được tính.

2. **Phạt nặng hơn các sai số lớn:** Vì hàm bình phương là hàm lồi (convex) và tăng nhanh, một sai số gấp đôi sẽ bị phạt gấp **bốn** lần. Điều này khiến MSE rất nhạy cảm với **outlier** (điểm dữ liệu bất thường).

> **Ví dụ minh họa (từ slide):**
>
> | $i$ | $y_i$ | $\hat{y}_i$ | $(y_i - \hat{y}_i)^2$ |
> |-----|--------|-------------|----------------------|
> | 1   | 3      | 2.5         | 0.25                 |
> | 2   | -0.5   | 0           | 0.25                 |
> | 3   | 2      | 2           | 0                    |
> | 4   | 7      | 8           | 1                    |
>
> $$L_{\text{MSE}} = \frac{0.25 + 0.25 + 0 + 1}{4} = 0.375$$

> 📸 **[Cần ảnh]:** Đồ thị hàm $L = e^2$ — đường parabol cho thấy sai số lớn bị phạt phi tuyến tính. *(Tham khảo: bất kỳ textbook ML nào, hoặc vẽ bằng matplotlib)*

---

### 1.3 Mean Absolute Error (MAE)

**Công thức:**

$$L_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Thay vì bình phương, MAE lấy **giá trị tuyệt đối** của phần dư.

> **Ví dụ (dùng lại dữ liệu trên):**
>
> $$|3 - 2.5| + |-0.5 - 0| + |2 - 2| + |7 - 8| = 0.5 + 0.5 + 0 + 1 = 2$$
>
> $$L_{\text{MAE}} = \frac{2}{4} = 0.5$$

---

### 1.4 So sánh MSE và MAE

| Tiêu chí | MSE | MAE |
|----------|-----|-----|
| Loại penalty | **Bậc hai (quadratic)** | **Tuyến tính (linear)** |
| Nhạy cảm với outlier | **Cao** (outlier bị khuếch đại) | **Thấp hơn** (sai số lớn không bị phạt quá mức) |
| Đạo hàm tại 0 | Liên tục | **Không xác định** (bị gãy tại $e=0$) |
| Ứng dụng phổ biến | Khi dữ liệu sạch | Khi dữ liệu có nhiều outlier |

> **Khi nào dùng cái nào?** Nếu bạn biết dữ liệu của mình có outlier (ví dụ: giá nhà bất thường, cảm biến nhiễu), hãy ưu tiên MAE. Nếu muốn mô hình "sợ" các sai lầm lớn và cố gắng tránh chúng, dùng MSE.

> 📸 **[Cần ảnh]:** Đồ thị so sánh hai đường $L = e^2$ và $L = |e|$ trên cùng một trục — minh họa rõ MAE tuyến tính vs MSE bậc hai. *(Tham khảo: slide gốc hoặc vẽ bằng numpy/matplotlib)*

---

## 2. Label Representation for Classification

Trước khi nói đến hàm mất mát phân loại, ta cần hiểu rõ **nhãn (label)** trong bài toán phân loại được biểu diễn như thế nào về mặt toán học.

### 2.1 Nhãn là gì?

Trong phân loại, mỗi mẫu dữ liệu có một **nhãn lớp**:

$$y \in \{\text{cat, dog, car, ...}\}$$

Con người nhìn vào thấy ngôn ngữ tự nhiên ("mèo", "chó"), nhưng máy tính chỉ hiểu được số. Vì vậy ta cần **mã hóa nhãn** thành dạng số.

---

### 2.2 Text-to-Index Encoding (Mã hóa chỉ số)

Bước đầu tiên là ánh xạ mỗi lớp sang một số nguyên:

| Tên lớp | Chỉ số |
|---------|--------|
| cat     | 0      |
| dog     | 1      |
| horse   | 2      |

Lúc này: $y \in \{0, 1, 2\}$

> **Lưu ý quan trọng:** Chỉ số là **ký hiệu (symbol)**, không phải giá trị số học có thứ tự lớn-nhỏ. Việc "dog = 1, horse = 2" không có nghĩa là horse > dog hay horse gần dog hơn cat. Đây là một cái bẫy phổ biến mà nhiều người mới học hay nhầm!

---

### 2.3 One-Hot Encoding (Hard Label)

Để tránh vấn đề thứ tự giả tạo ở trên, ta dùng **one-hot encoding**: mỗi nhãn được mã hóa thành một vector nhị phân độ dài $K$ (số lượng lớp):

$$y \rightarrow \mathbf{y} \in \{0, 1\}^K$$

**Quy tắc:** Đúng một phần tử bằng 1 (ứng với lớp đúng), tất cả còn lại bằng 0.

> **Ví dụ:** $K = 3$, nhãn "dog" (index = 1):
>
> $$\mathbf{y} = [0, 1, 0]$$

> **Phép ẩn dụ:** One-hot giống như phiếu bầu — chỉ được chọn một ứng cử viên (lớp đúng), và không thể bầu một nửa phiếu.

---

### 2.4 Nhãn như Phân phối Xác suất

Một quan sát sâu sắc hơn: **one-hot label chính là một phân phối xác suất đặc biệt**:

$$\sum_{k=1}^{K} y_k = 1$$

Cụ thể:

$$[0, 1, 0] \Rightarrow P(\text{dog}) = 1, \quad P(\text{others}) = 0$$

Đây là một phân phối xác suất hoàn toàn hợp lệ — nó chỉ đang nói "tôi chắc 100% đây là dog". Cách nhìn này rất quan trọng vì nó kết nối nhãn phân loại với lý thuyết thông tin.

---

### 2.5 Soft Label (Nhãn mềm)

Đôi khi nhãn không nhất thiết phải hoàn toàn chắc chắn. **Soft label** cho phép biểu diễn sự không chắc chắn:

$$\mathbf{y} = [0.05, 0.90, 0.05]$$

- Tổng vẫn bằng 1 (vẫn là phân phối xác suất)
- Nhưng xác suất được trải ra nhiều lớp

**Nguồn gốc của soft labels:**

- **Label smoothing:** Kỹ thuật regularization — thay vì tin 100% vào nhãn, ta "làm mềm" bằng cách trải một phần nhỏ xác suất sang các lớp khác.
- **Knowledge distillation:** Học từ một mô hình lớn (teacher) — teacher thường cho ra phân phối mềm thay vì one-hot.
- **Human ambiguity:** Đôi khi con người không đồng ý với nhau về nhãn (ví dụ: ảnh vừa có thể là "sói" vừa có thể là "chó").

---

### 2.6 Hard Label vs Soft Label

| Đặc điểm | Hard Label | Soft Label |
|----------|-----------|-----------|
| Dạng | One-hot | Phân phối |
| Cho phép bất định? | Không | Có |
| Ví dụ | `[0, 1, 0]` | `[0.1, 0.8, 0.1]` |

> **Nhận xét quan trọng:** Cả hai đều là **mục tiêu hợp lệ** cho hàm cross-entropy. Cross-entropy hoạt động được với mọi phân phối xác suất, không chỉ riêng one-hot.

---

### 2.7 Từ Nhãn đến Hàm Mất Mát

Bây giờ ta có:
- $\mathbf{y}$ = phân phối thực (true distribution) — là nhãn
- $\hat{\mathbf{p}}$ = phân phối dự đoán (predicted distribution) — đầu ra của mô hình

**Mục tiêu:** Đo lường sự khác biệt giữa hai phân phối này → **Cross-Entropy Loss**.

---

## 3. Classification Loss

### 3.1 Tư tưởng cốt lõi

Mô hình phân loại đầu ra là một xác suất:

$$\hat{p} = P(y = 1 | x)$$

Hàm mất mát lý tưởng phải có tính chất:
- **Nhỏ** khi mô hình gán xác suất cao cho lớp đúng
- **Lớn** khi mô hình tự tin nhưng sai

> **Phép ẩn dụ:** Giống như chấm điểm một thám tử. Nếu thám tử nói "Tôi chắc 90% là kẻ đó" mà đúng → điểm cao. Nếu nói "Tôi chắc 90%" mà sai → phạt rất nặng. Không ai phạt nhẹ thám tử tự tin mà sai.

---

### 3.2 Binary Cross Entropy (BCE)

Dành cho bài toán **phân loại nhị phân** ($y \in \{0, 1\}$):

$$L_{\text{BCE}} = -\left[ y \log(\hat{p}) + (1 - y) \log(1 - \hat{p}) \right]$$

**Phân tích theo từng trường hợp:**

- **Nếu $y = 1$ (nhãn thực là dương):**
  $$L = -\log(\hat{p})$$
  → Muốn loss nhỏ, cần $\hat{p}$ lớn (gần 1). Nếu $\hat{p} = 0.9$: $L = -\log(0.9) \approx 0.105$ (tốt). Nếu $\hat{p} = 0.1$: $L = -\log(0.1) \approx 2.30$ (tệ!).

- **Nếu $y = 0$ (nhãn thực là âm):**
  $$L = -\log(1 - \hat{p})$$
  → Muốn loss nhỏ, cần $\hat{p}$ nhỏ (gần 0).

> **Ví dụ step-by-step:**
>
> *Mẫu 1:* $y = 1$, $\hat{p} = 0.8$
> $$L = -\log(0.8) \approx 0.223$$
>
> *Mẫu 2:* $y = 0$, $\hat{p} = 0.7$ (mô hình nghĩ xác suất dương là 0.7, nhưng thực ra là âm)
> $$L = -\log(1 - 0.7) = -\log(0.3) \approx 1.204$$

> 📸 **[Cần ảnh]:** Đồ thị $L = -\log(p)$ — đường cong giảm nhanh khi $p \to 1$ và tăng vô cực khi $p \to 0$. Minh họa trực quan tại sao "tự tin mà sai" bị phạt nặng. *(Slide gốc trang 19, hoặc vẽ bằng numpy)*

---

### 3.3 Multiclass Cross Entropy (CE)

Dành cho bài toán phân loại **nhiều lớp ($K$ lớp)**:

$$L_{\text{CE}} = -\sum_{k=1}^{K} y_k \log(\hat{p}_k)$$

Với $\mathbf{y}$ là one-hot vector, chỉ một $y_k = 1$, nên:

$$L_{\text{CE}} = -\log(\hat{p}_{\text{class đúng}})$$

**Công thức rút gọn này chính là một nhận xét rất sâu sắc:** Cross-entropy đơn giản chỉ là **âm log-xác suất của lớp đúng**. Mô hình giỏi là mô hình gán xác suất cao nhất cho lớp đúng.

> **Ví dụ step-by-step:**
>
> $K = 4$ lớp. Nhãn thực: lớp 2 (index bắt đầu từ 0).
>
> $$\mathbf{y} = (0, 0, 1, 0), \quad \hat{\mathbf{p}} = (0.1, 0.2, 0.6, 0.1)$$
>
> $$L = -\log(0.6) \approx 0.511 \quad \text{(tốt)}$$
>
> Nhưng nếu mô hình dự đoán kém:
>
> $$\hat{\mathbf{p}} = (0.05, 0.05, 0.05, 0.85)$$
>
> $$L = -\log(0.05) \approx 2.996 \quad \text{(tệ)}$$

---

### 3.4 Softmax + CE = Log Loss

Trong mạng neural, đầu ra cuối cùng thường là **logits** — các giá trị thực chưa chuẩn hóa:

$$z_k = \mathbf{w}_k^\top \mathbf{h} + b_k$$

**Softmax** chuyển logits thành phân phối xác suất hợp lệ:

$$\hat{p}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

Sau đó áp CE Loss:

$$L = -\log(\hat{p}_y) = -\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right)$$

**Ý nghĩa:** Tối ưu hóa loss này tương đương với việc **tối đa hóa xác suất của lớp đúng** — đây là nguyên lý **Maximum Likelihood Estimation (MLE)** trong ngụy trang!

> **Lưu ý thực hành:** Trong PyTorch, `nn.CrossEntropyLoss` đã kết hợp Softmax + NLL Loss vào một bước duy nhất. Không cần (và không nên) gọi Softmax thủ công trước khi truyền vào `CrossEntropyLoss`.

---

### 3.5 Tại sao lại dùng Logarithm?

Đây là câu hỏi nhiều bạn thắc mắc. Có ba lý do chính:

1. **Biến nhân thành cộng:** Trong xác suất, ta thường nhân nhiều xác suất lại với nhau. Logarithm biến phép nhân thành phép cộng:
   $$\log(p_1 \cdot p_2 \cdot \ldots) = \log(p_1) + \log(p_2) + \ldots$$
   Điều này ổn định về mặt số học (tránh underflow khi nhân nhiều số nhỏ).

2. **Phạt nặng các sai lầm tự tin:** Hàm $-\log(p)$ tăng vô cực khi $p \to 0$. Mô hình càng tự tin sai, hình phạt càng khủng khiếp.

3. **Trơn và khả vi (smooth & differentiable):** Hàm log trơn tru ở khắp nơi trong $(0, 1]$, cho phép gradient descent hoạt động tốt.

> 📸 **[Cần ảnh]:** Đồ thị hàm $-\log(p)$ với $p \in (0, 1]$ — minh họa behavior tại các giá trị $p$ khác nhau. *(Tham khảo: slide gốc trang 18-19)*

---

## 4. Tổng kết

### Bảng tóm tắt các hàm mất mát

| Bài toán | Hàm mất mát | Công thức tóm tắt |
|----------|-------------|-------------------|
| Hồi quy | **MSE** | $\frac{1}{n}\sum(y-\hat{y})^2$ — phạt bậc hai |
| Hồi quy | **MAE** | $\frac{1}{n}\sum|y-\hat{y}|$ — phạt tuyến tính |
| Hồi quy | **RMSE** | $\sqrt{\text{MSE}}$ — cùng đơn vị với $y$ |
| Phân loại nhị phân | **BCE** | $-[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ |
| Phân loại đa lớp | **CE** | $-\sum_k y_k \log \hat{p}_k = -\log\hat{p}_{\text{correct}}$ |

### Key Insight

> **Loss = Mức độ ngạc nhiên khi thấy nhãn thực.**
>
> Nếu mô hình tự tin đúng → ít ngạc nhiên → loss nhỏ.
> Nếu mô hình tự tin sai → rất ngạc nhiên → loss lớn.
>
> Đây chính là tinh thần của lý thuyết thông tin: **cross-entropy đo lường sự bất ngờ trung bình** khi dùng phân phối dự đoán $\hat{\mathbf{p}}$ để mô tả phân phối thực $\mathbf{y}$.

---

## Ghi chú về ảnh cần đính kèm

| # | Mô tả ảnh | Vị trí trong slide | Gợi ý nguồn |
|---|-----------|-------------------|-------------|
| 1 | Đồ thị $L = e^2$ (parabol) so với $L = \|e\|$ (V-shape) | Trang 5–7 | Vẽ bằng `matplotlib` hoặc xem [d2l.ai](https://d2l.ai) |
| 2 | Đồ thị BCE loss $L = -\log(p)$ — hình dạng hyperbolic | Trang 18–19 | Slide gốc trang 19; hoặc `plt.plot(p, -np.log(p))` |
| 3 | Minh họa softmax: logits → xác suất | Trang 22–23 | Bất kỳ tutorial PyTorch/CS231n nào |
| 4 | So sánh hard label vs soft label dạng bar chart | Trang 13–14 | Vẽ tay hoặc `matplotlib` bar chart |

---

## 5. Bài Tập Tính Toán

> Tự làm trước khi mở đáp án — đây là dạng bài thi phổ biến nhất trong chương này.

---

### Bài 1 — MSE và MAE

Một mô hình hồi quy dự đoán giá nhà (đơn vị: tỷ đồng) trên 5 mẫu:

| Mẫu | $y_i$ (thực tế) | $\hat{y}_i$ (dự đoán) |
|-----|-----------------|----------------------|
| 1 | 2.0 | 2.5 |
| 2 | 4.0 | 3.5 |
| 3 | 3.0 | 3.0 |
| 4 | 5.0 | 7.0 |
| 5 | 1.0 | 1.2 |

**(a)** Tính $L_{\text{MSE}}$.

**(b)** Tính $L_{\text{MAE}}$.

**(c)** Mẫu số 4 là outlier (sai lệch lớn). Nó đóng góp bao nhiêu phần trăm vào MSE? Và vào MAE? Từ đó rút ra nhận xét.

<details>
<summary>📋 Đáp án Bài 1</summary>

Phần dư $e_i = y_i - \hat{y}_i$:

| Mẫu | $e_i$ | $e_i^2$ | $\|e_i\|$ |
|-----|-------|---------|-----------|
| 1 | $-0.5$ | $0.25$ | $0.5$ |
| 2 | $0.5$ | $0.25$ | $0.5$ |
| 3 | $0$ | $0$ | $0$ |
| 4 | $-2.0$ | $4.00$ | $2.0$ |
| 5 | $-0.2$ | $0.04$ | $0.2$ |
| **Tổng** | | **4.54** | **3.2** |

**(a) MSE:**
$$L_{\text{MSE}} = \frac{4.54}{5} = \mathbf{0.908}$$

**(b) MAE:**
$$L_{\text{MAE}} = \frac{3.2}{5} = \mathbf{0.640}$$

**(c) Đóng góp của mẫu 4 (outlier):**

- Vào MSE: $4.00 / 4.54 \approx \mathbf{88.1\%}$
- Vào MAE: $2.0 / 3.2 = \mathbf{62.5\%}$

**Nhận xét:** Outlier chiếm 88% MSE nhưng chỉ 62.5% MAE — do bình phương khuếch đại sai lệch lớn (sai 2× bị phạt 4×). MSE nhạy cảm với outlier hơn MAE rất nhiều. Khi dữ liệu có outlier, nên cân nhắc dùng MAE hoặc Huber Loss.

</details>

---

### Bài 2 — Binary Cross Entropy (BCE)

Bài toán phân loại ảnh: mèo (y=1) hay không phải mèo (y=0).

**(a)** Tính BCE loss cho từng mẫu sau:

| Mẫu | $y$ | $\hat{p}$ | $L_{\text{BCE}}$ |
|-----|-----|-----------|-----------------|
| 1 | 1 | 0.9 | ? |
| 2 | 1 | 0.3 | ? |
| 3 | 0 | 0.1 | ? |
| 4 | 0 | 0.8 | ? |

**(b)** Tính $L_{\text{BCE}}$ trung bình trên 4 mẫu.

**(c)** Mẫu nào bị phạt nặng nhất? Tại sao?

*(Dùng: $\log(0.9) \approx -0.105$, $\log(0.3) \approx -1.204$, $\log(0.1) \approx -2.303$, $\log(0.2) \approx -1.609$)*

<details>
<summary>📋 Đáp án Bài 2</summary>

Công thức: $L = -[y\log\hat{p} + (1-y)\log(1-\hat{p})]$

| Mẫu | $y$ | $\hat{p}$ | Tính | $L$ |
|-----|-----|-----------|------|-----|
| 1 | 1 | 0.9 | $-[1 \times \log(0.9) + 0] = -(-0.105)$ | **0.105** |
| 2 | 1 | 0.3 | $-[1 \times \log(0.3) + 0] = -(-1.204)$ | **1.204** |
| 3 | 0 | 0.1 | $-[0 + 1 \times \log(1-0.1)] = -\log(0.9) = 0.105$ | **0.105** |
| 4 | 0 | 0.8 | $-[0 + 1 \times \log(1-0.8)] = -\log(0.2) = 1.609$ | **1.609** |

**(b) Trung bình:**
$$L_{\text{avg}} = \frac{0.105 + 1.204 + 0.105 + 1.609}{4} = \frac{3.023}{4} \approx \mathbf{0.756}$$

**(c)** Mẫu 4 bị phạt nặng nhất ($L = 1.609$): mô hình **tự tin sai** — gán $\hat{p} = 0.8$ (khả năng cao là mèo) trong khi thực tế $y = 0$ (không phải mèo). Mẫu 1 và 3 dự đoán đúng và tự tin → loss nhỏ (0.105).

</details>

---

### Bài 3 — Softmax và Multiclass Cross Entropy

Mạng neural phân loại 3 lớp (mèo / chó / chim) cho ra logits:

$$z = [2.0,\ 1.0,\ 0.5]$$

Nhãn thực: lớp 0 (mèo), tức $\mathbf{y} = [1, 0, 0]$.

**(a)** Tính xác suất softmax $\hat{\mathbf{p}} = [\hat{p}_0, \hat{p}_1, \hat{p}_2]$.

**(b)** Tính $L_{\text{CE}}$.

**(c)** Nếu logits thay đổi thành $z' = [0.5,\ 1.0,\ 2.0]$ (mạng dự đoán ngược lại), tính lại $L_{\text{CE}}$. So sánh với câu (b).

*(Cho: $e^{0.5} \approx 1.649$, $e^{1.0} \approx 2.718$, $e^{2.0} \approx 7.389$)*

<details>
<summary>📋 Đáp án Bài 3</summary>

**(a) Softmax với $z = [2.0, 1.0, 0.5]$:**

$$e^{z_0} = 7.389,\quad e^{z_1} = 2.718,\quad e^{z_2} = 1.649$$
$$\text{Tổng} = 7.389 + 2.718 + 1.649 = 11.756$$

$$\hat{p}_0 = \frac{7.389}{11.756} \approx 0.628, \quad \hat{p}_1 = \frac{2.718}{11.756} \approx 0.231, \quad \hat{p}_2 = \frac{1.649}{11.756} \approx 0.140$$

$$\hat{\mathbf{p}} \approx [0.628,\ 0.231,\ 0.140]$$

Kiểm tra: $0.628 + 0.231 + 0.140 = 0.999 \approx 1$ ✓

**(b) CE loss** (nhãn đúng là lớp 0):

$$L_{\text{CE}} = -\log(\hat{p}_0) = -\log(0.628) \approx \mathbf{0.465}$$

**(c) Softmax với $z' = [0.5, 1.0, 2.0]$:**

$$e^{z'_0} = 1.649,\quad e^{z'_1} = 2.718,\quad e^{z'_2} = 7.389,\quad \text{Tổng} = 11.756$$

$$\hat{p}'_0 = \frac{1.649}{11.756} \approx 0.140$$

$$L'_{\text{CE}} = -\log(0.140) \approx \mathbf{1.966}$$

**So sánh:** $L = 0.465$ (dự đoán đúng, tự tin) vs $L' = 1.966$ (dự đoán sai, tự tin về lớp sai) → loss tăng **4.2 lần** khi mô hình đảo ngược logits hoàn toàn. Đây là minh chứng cho việc CE phạt nặng khi mô hình tự tin nhưng sai.

</details>

---

### Bài 4 — So sánh Hard Label vs Soft Label với CE

Một bài toán phân loại 4 lớp. Nhãn thực tế và hai cách mã hóa:

- **Hard label:** $\mathbf{y}_{hard} = [0,\ 1,\ 0,\ 0]$ (chắc 100% lớp 1)
- **Soft label:** $\mathbf{y}_{soft} = [0.05,\ 0.85,\ 0.05,\ 0.05]$ (label smoothing $\varepsilon = 0.15$)

Mô hình dự đoán: $\hat{\mathbf{p}} = [0.05,\ 0.80,\ 0.10,\ 0.05]$

**(a)** Tính $L_{\text{CE}}$ với hard label.

**(b)** Tính $L_{\text{CE}}$ với soft label.

**(c)** Mô hình dự đoán hoàn hảo $\hat{\mathbf{p}} = [0, 1, 0, 0]$. Tính $L_{\text{CE}}$ với hard label. Điều gì xảy ra với soft label khi $\hat{p}_1 \to 1$? Đây là lý do tại sao label smoothing giúp tránh overconfidence.

*(Dùng: $\log(0.80) \approx -0.223$, $\log(0.05) \approx -2.996$, $\log(0.10) \approx -2.303$)*

<details>
<summary>📋 Đáp án Bài 4</summary>

**(a) Hard label CE:**

Chỉ lớp 1 có $y_k = 1$:
$$L_{hard} = -\sum_k y_k \log \hat{p}_k = -\log(\hat{p}_1) = -\log(0.80) \approx \mathbf{0.223}$$

**(b) Soft label CE:**

$$L_{soft} = -\sum_k y_k^{soft} \log \hat{p}_k$$
$$= -[0.05\log(0.05) + 0.85\log(0.80) + 0.05\log(0.10) + 0.05\log(0.05)]$$
$$= -[0.05×(-2.996) + 0.85×(-0.223) + 0.05×(-2.303) + 0.05×(-2.996)]$$
$$= -[-0.150 - 0.190 - 0.115 - 0.150]$$
$$= -(-0.605) \approx \mathbf{0.605}$$

**(c) Mô hình hoàn hảo $\hat{\mathbf{p}} = [0, 1, 0, 0]$:**

- **Hard label:** $L_{hard} = -\log(1) = 0$ → loss = 0 hoàn hảo.
- **Soft label:** $L_{soft} = -[0.05\log(0) + 0.85\log(1) + ...] = -0.05×(-\infty) = +\infty$

Nhưng trong thực tế softmax không bao giờ cho $\hat{p}_k = 0$ hay $= 1$ chính xác (vì $e^z > 0$ với mọi $z$ hữu hạn). Khi $\hat{p}_1 \to 1$:

$$L_{soft} \to -(0.05 \times (-\infty) + 0.85 \times 0 + ...) \to +\infty$$

**Ý nghĩa:** Với soft label, mô hình **không bao giờ có thể đạt loss = 0** — luôn bị phạt vì xác suất các lớp khác > 0. Điều này ngăn mạng đẩy logits ra vô cực (overconfidence) và giúp regularize tốt hơn. Đây là cơ chế của **label smoothing**.

</details>

---

### Bài 5 — Truy vết ngược từ loss

Một binary classifier cho ra loss $L_{\text{BCE}} = 1.204$ với nhãn thực $y = 1$.

**(a)** Mô hình dự đoán xác suất $\hat{p}$ bằng bao nhiêu?

**(b)** Nếu loss thay đổi thành $L = 0.357$, $\hat{p}$ mới là bao nhiêu? *(Gợi ý: $e^{-0.357} \approx 0.70$)*

**(c)** Ở epoch 1, loss là 2.303. Ở epoch 10, loss là 0.105. Tính $\hat{p}$ tại cả hai epoch. Mô hình đã cải thiện như thế nào?

*(Dùng: $e^{-1.204} \approx 0.30$, $e^{-2.303} \approx 0.10$, $e^{-0.105} \approx 0.90$)*

<details>
<summary>📋 Đáp án Bài 5</summary>

Với $y = 1$: $L_{\text{BCE}} = -\log(\hat{p})$, suy ra $\hat{p} = e^{-L}$.

**(a)** $L = 1.204$:
$$\hat{p} = e^{-1.204} \approx \mathbf{0.30}$$
Mô hình chỉ gán 30% xác suất cho lớp đúng — khá tệ.

**(b)** $L = 0.357$:
$$\hat{p} = e^{-0.357} \approx \mathbf{0.70}$$
Cải thiện đáng kể — mô hình gán 70% xác suất cho lớp đúng.

**(c)** Tiến trình huấn luyện:

| Epoch | Loss | $\hat{p}$ | Nhận xét |
|-------|------|-----------|---------|
| 1 | 2.303 | $e^{-2.303} \approx 0.10$ | Mô hình chỉ tin 10% — gần như đoán ngẫu nhiên (1/class) |
| 10 | 0.105 | $e^{-0.105} \approx 0.90$ | Mô hình tin 90% — dự đoán tốt và tự tin |

Loss giảm từ 2.303 → 0.105 tương ứng $\hat{p}$ tăng từ 0.10 → 0.90. Đây là minh họa trực quan nhất của quá trình học: mô hình ngày càng **tự tin hơn về câu trả lời đúng**.

</details>

---

### Bài 6 — Câu hỏi tư duy nhanh

**(a)** Tại sao `CrossEntropyLoss` trong PyTorch nhận **logits** (chưa qua softmax) thay vì xác suất? Điều gì xảy ra nếu bạn truyền vào xác suất đã qua softmax?

**(b)** Với bài toán phân loại K lớp, mô hình khởi tạo ngẫu nhiên (uniform) dự đoán $\hat{p}_k = 1/K$ cho mọi lớp. CE loss kỳ vọng lúc đó bằng bao nhiêu? Kiểm chứng với $K = 10$ (CIFAR-10).

**(c)** Khi nào nên dùng MSE thay vì CE cho bài toán phân loại? (Gợi ý: có trường hợp nào không?)

<details>
<summary>📋 Đáp án Bài 6</summary>

**(a) Lý do PyTorch nhận logits:**

PyTorch `CrossEntropyLoss` kết hợp `LogSoftmax + NLLLoss` trong một phép toán duy nhất, vì lý do **ổn định số học (numerical stability)**:

$$\log(\text{softmax}(z_k)) = z_k - \log\sum_j e^{z_j}$$

Nếu $z_k$ rất lớn, $e^{z_k}$ tràn số (overflow `inf`). PyTorch dùng kỹ thuật **log-sum-exp trick** với max-subtraction để tránh điều này:

$$\log\sum_j e^{z_j} = c + \log\sum_j e^{z_j - c}, \quad c = \max_j z_j$$

Nếu bạn truyền vào xác suất đã qua softmax: PyTorch sẽ áp softmax một lần nữa → **double softmax** → sai hoàn toàn. Đây là bug phổ biến của người mới.

**(b) CE của mô hình uniform:**

$$L_{\text{CE}} = -\log\left(\frac{1}{K}\right) = \log(K)$$

Với $K = 10$ (CIFAR-10):
$$L_{\text{CE,init}} = \log(10) \approx \mathbf{2.303}$$

Đây là giá trị loss "baseline" khi mô hình chưa học gì. Nếu loss ban đầu của bạn **khác xa 2.303** với CIFAR-10, có thể có lỗi trong pipeline (normalization, weight init, v.v.).

**(c) MSE cho bài toán phân loại:**

Về lý thuyết, có thể dùng MSE với one-hot labels: $L = \|y - \hat{p}\|^2$.

Nhưng **không nên** vì:
- MSE giả định phân phối Gaussian của target — không phù hợp cho xác suất.
- Gradient của MSE qua softmax bị **vanish khi mô hình tự tin sai** (saturation region), khiến học rất chậm.
- CE có gradient clean hơn: $\partial L/\partial z_k = \hat{p}_k - y_k$ — đơn giản và không bị vanish.

**Trường hợp ngoại lệ:** Distillation hoặc một số bài regression on probabilities có thể dùng MSE, nhưng đây là thiết kế đặc biệt, không phải mặc định.

</details>

---

### Tổng hợp công thức và mẹo thi

| Loss | Công thức | Mẹo tính nhanh |
|------|----------|----------------|
| **MSE** | $\frac{1}{n}\sum(y-\hat{y})^2$ | Tính residual → bình phương → lấy trung bình |
| **MAE** | $\frac{1}{n}\sum\|y-\hat{y}\|$ | Tính \|residual\| → lấy trung bình |
| **BCE** ($y=1$) | $-\log(\hat{p})$ | Loss = $-\log$(xác suất dự đoán cho lớp đúng) |
| **BCE** ($y=0$) | $-\log(1-\hat{p})$ | Loss = $-\log$(1 - xác suất dự đoán) |
| **CE đa lớp** | $-\log(\hat{p}_{\text{đúng}})$ | Chỉ nhìn vào xác suất của lớp đúng, bỏ qua phần còn lại |
| **Softmax** | $\hat{p}_k = e^{z_k}/\sum_j e^{z_j}$ | Tính tổng exp trước, chia từng phần tử |
| **CE từ logits** | $z_y - \log\sum_j e^{z_j}$ | Lấy logit lớp đúng trừ log-sum-exp toàn bộ |
| **Loss init** | $\log(K)$ | CIFAR-10: $\approx 2.303$; ImageNet ($K$=1000): $\approx 6.908$ |
