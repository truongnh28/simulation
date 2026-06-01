# Bài 06 — Độ Đo Đánh Giá (Evaluation Metrics)

> **Nguồn slide:** `slides-v1/foundation/06-Metrics.pdf` — Thanh-Sach LE, HCMUT, VNU-HCM (05/02/2026)

---

## Mục lục

1. [Regression Metrics](#1-regression-metrics)
2. [Classification Metrics](#2-classification-metrics)
3. [Tổng kết](#3-tổng-kết)
4. [Bài Tập Tính Toán](#4-bài-tập-tính-toán)

---

## Giới thiệu

Có một câu hỏi tưởng đơn giản nhưng rất quan trọng: *"Mô hình của tôi tốt đến mức nào?"*

**Hàm mất mát (loss function)** trả lời câu hỏi này trong quá trình **huấn luyện** — nó là la bàn để gradient descent tối ưu hóa. Nhưng loss function không phải lúc nào cũng là thứ ta muốn đưa cho stakeholder xem, vì nó thường là một con số trừu tượng (ví dụ: cross-entropy = 0.34).

**Độ đo đánh giá (evaluation metrics)** là những thước đo được thiết kế để **con người có thể hiểu và giải thích**: sai trung bình bao nhiêu đơn vị? Tỷ lệ dự đoán đúng là bao nhiêu phần trăm? Mô hình bỏ sót bao nhiêu ca bệnh?

> **Phân biệt quan trọng:** Loss dùng để tối ưu (trong training). Metrics dùng để đánh giá (trong validation/test). Chúng có thể khác nhau — ví dụ ta tối ưu Cross-Entropy nhưng đánh giá bằng Accuracy và F1.

---

## 1. Regression Metrics

### 1.1 Đặt vấn đề

Cho mô hình hồi quy dự đoán giá trị thực:

$$\hat{y} = f(x), \quad y \in \mathbb{R}$$

**Phần dư (residual)** của mỗi mẫu $i$:

$$e_i = y_i - \hat{y}_i$$

Các metrics hồi quy về bản chất là những cách khác nhau để **tóm tắt độ lớn và cấu trúc** của tập hợp các phần dư này thành một con số duy nhất.

---

### 1.2 MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

MAE lấy trung bình của **giá trị tuyệt đối** các phần dư. Đây là cách đơn giản nhất: ta hỏi "trung bình mô hình sai bao nhiêu đơn vị?"

**Ưu điểm:**
- **Cùng đơn vị với $y$:** Nếu $y$ là giá nhà (triệu đồng), MAE = 50 có nghĩa là trung bình sai 50 triệu — trực quan hoàn toàn.
- **Bền vững với outlier:** Penalty tuyến tính — một sai số gấp đôi chỉ bị tính gấp đôi, không hơn.

---

### 1.3 MSE và RMSE

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2, \qquad \text{RMSE} = \sqrt{\text{MSE}}$$

MSE bình phương phần dư trước khi lấy trung bình. **RMSE** đơn giản là lấy căn bậc hai của MSE để đưa về cùng đơn vị với $y$.

**Tại sao RMSE lại phổ biến hơn MSE khi báo cáo?** Vì MSE có đơn vị là bình phương (ví dụ: triệu² — vô nghĩa), còn RMSE có cùng đơn vị với $y$.

**Nhạy cảm với outlier:** Một sai số gấp đôi bị phạt **gấp bốn** (vì bình phương). Điều này có nghĩa là vài điểm dữ liệu sai lớn có thể kéo MSE/RMSE lên rất cao.

> **Ví dụ step-by-step (từ slide):**
>
> $y = [3, -0.5, 2, 7]$, $\hat{y} = [2.5, 0, 2, 8]$
>
> | $i$ | $y_i$ | $\hat{y}_i$ | $e_i$ | $\|e_i\|$ | $e_i^2$ |
> |-----|--------|-------------|-------|-----------|---------|
> | 1   | 3.0    | 2.5         | 0.5   | 0.5       | 0.25    |
> | 2   | -0.5   | 0.0         | -0.5  | 0.5       | 0.25    |
> | 3   | 2.0    | 2.0         | 0.0   | 0.0       | 0.00    |
> | 4   | 7.0    | 8.0         | -1.0  | 1.0       | 1.00    |
> | **Tổng** | | | | **2.0** | **1.50** |
>
> $$\text{MAE} = \frac{2.0}{4} = 0.5$$
>
> $$\text{MSE} = \frac{1.50}{4} = 0.375, \qquad \text{RMSE} = \sqrt{0.375} \approx 0.612$$

---

### 1.4 R² — Hệ số Xác định (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}, \qquad \bar{y} = \frac{1}{n}\sum_{i=1}^n y_i$$

Tử số là **SSE (Sum of Squared Errors)** — tổng bình phương sai số của mô hình. Mẫu số là **SST (Sum of Squares Total)** — tổng bình phương độ lệch của dữ liệu so với trung bình.

**Cách đọc $R^2$:**

| Giá trị $R^2$ | Ý nghĩa |
|---------------|---------|
| $R^2 = 1$ | Dự đoán hoàn hảo ($\text{SSE} = 0$) |
| $R^2 = 0$ | Mô hình chỉ tốt bằng việc luôn dự đoán $\bar{y}$ (tệ!) |
| $R^2 < 0$ | Mô hình **tệ hơn** việc luôn đoán trung bình |

> **Phép ẩn dụ:** $R^2$ đo lường "mô hình giải thích được bao nhiêu phần trăm phương sai của dữ liệu". $R^2 = 0.85$ nghĩa là mô hình giải thích được 85% sự biến động trong $y$.

> **Ví dụ step-by-step (từ slide):**
>
> Dùng lại ví dụ trên: $\bar{y} = (3 - 0.5 + 2 + 7)/4 = 2.875$
>
> $$\text{SSE} = 1.50 \quad \text{(tính ở trên)}$$
>
> $$\text{SST} = (3-2.875)^2 + (-0.5-2.875)^2 + (2-2.875)^2 + (7-2.875)^2$$
> $$= 0.0156 + 11.3906 + 0.7656 + 17.0156 = 29.1875$$
>
> $$R^2 = 1 - \frac{1.50}{29.1875} \approx 1 - 0.0514 \approx 0.949$$
>
> Tốt! Mô hình giải thích được 94.9% phương sai của dữ liệu.

---

### 1.5 Lời khuyên thực tế: Báo cáo metric nào?

| Metric | Khi nào nên dùng |
|--------|-----------------|
| **MAE** | Khi muốn diễn giải trực quan, dữ liệu có outlier |
| **RMSE** | Khi muốn nhấn mạnh sai số lớn, phổ biến trong forecasting |
| **R²** | Khi muốn so sánh tương đối với baseline (mô hình dự đoán trung bình) |

**Luôn đi kèm với:**
- **Baseline:** Mô hình đơn giản nhất (luôn đoán $\bar{y}$) làm gốc so sánh.
- **Đơn vị và scale của $y$:** RMSE = 5 tốt hay xấu phụ thuộc vào $y$ đang đo gì.
- **Kiểm tra outlier:** Vài điểm bất thường có thể lật ngược kết luận.

> 📸 **[Cần ảnh]:** Scatter plot $y$ vs $\hat{y}$ — đường chéo 45° là perfect prediction, khoảng cách đến đường chéo là residual. *(Trang 4–7 slide hoặc vẽ bằng matplotlib)*

---

## 2. Classification Metrics

### 2.1 Đặt vấn đề và Confusion Matrix

Trong phân loại nhị phân ($y \in \{0, 1\}$), mô hình thường cho ra xác suất $\hat{p} = P(y=1|x)$, và ta cần chọn một **ngưỡng $\tau$** để ra quyết định:

$$\hat{y} = \mathbf{1}[\hat{p} \geq \tau]$$

Kết quả dự đoán được phân thành 4 loại, tóm tắt trong **Confusion Matrix**:

|               | Dự đoán = 1 | Dự đoán = 0 |
|---------------|-------------|-------------|
| **Thực tế = 1** | **TP** (True Positive)  | **FN** (False Negative) |
| **Thực tế = 0** | **FP** (False Positive) | **TN** (True Negative)  |

**Giải thích từng ô:**
- **TP:** Mô hình nói "có" và đúng là "có" → hit
- **TN:** Mô hình nói "không" và đúng là "không" → correct rejection
- **FP:** Mô hình nói "có" nhưng thực ra "không" → false alarm (Type I error)
- **FN:** Mô hình nói "không" nhưng thực ra "có" → miss (Type II error)

> **Phép ẩn dụ y tế:** Bài toán phát hiện ung thư. TP = phát hiện đúng bệnh nhân có ung thư. FN = bỏ sót bệnh nhân ung thư (nguy hiểm!). FP = báo nhầm người khỏe mạnh bị ung thư (gây lo lắng, tốn kém). Chi phí của FN và FP rất khác nhau — điều này sẽ quyết định metric nào ta ưu tiên.

> **Nhận xét cốt lõi:** Hầu hết mọi metric phân loại đều là một **hàm của (TP, FP, TN, FN)**.

> 📸 **[Cần ảnh]:** Confusion matrix dạng heatmap với màu sắc trực quan (TP/TN xanh, FP/FN đỏ). *(Trang 12 slide hoặc sklearn's `ConfusionMatrixDisplay`)*

---

### 2.2 Accuracy (Độ chính xác tổng thể)

$$\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}$$

Tỷ lệ dự đoán đúng trên tổng số mẫu.

**Khi nào Accuracy gây hiểu lầm?**

Hãy tưởng tượng bài toán phát hiện giao dịch gian lận (fraud detection): 990 giao dịch bình thường, chỉ 10 giao dịch gian lận. Một mô hình ngốc luôn nói "không gian lận" sẽ đạt Accuracy = 990/1000 = **99%** — nghe rất ấn tượng, nhưng nó bỏ sót 100% gian lận thực sự!

**Kết luận:** Với **dữ liệu mất cân bằng (imbalanced data)**, Accuracy không phải metric tốt.

---

### 2.3 Precision và Recall

$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}$$

Hai metric này đo hai khía cạnh khác nhau của chất lượng dự đoán:

**Precision (Độ chính xác):** Trong tất cả những gì mô hình dự đoán là "dương", bao nhiêu phần trăm thực sự đúng?
- Trả lời câu hỏi: *"Khi mô hình nói 'có', tôi có thể tin được không?"*
- Liên quan đến FP: Precision thấp → nhiều false alarm.

**Recall (Độ bao phủ / Sensitivity):** Trong tất cả những mẫu dương thực sự, mô hình phát hiện được bao nhiêu phần trăm?
- Trả lời câu hỏi: *"Mô hình có bỏ sót trường hợp dương nào không?"*
- Liên quan đến FN: Recall thấp → bỏ sót nhiều.

> **Phép ẩn dụ:** Tưởng tượng một bộ lọc thư rác.
> - **Precision cao, Recall thấp:** Bộ lọc rất thận trọng — chỉ chặn những gì chắc chắn là spam, nhưng để lọt nhiều spam.
> - **Recall cao, Precision thấp:** Bộ lọc rất hung hăng — chặn gần hết spam, nhưng cũng chặn nhầm nhiều thư quan trọng.

---

### 2.4 F1 Score

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

F1 là **trung bình điều hòa (harmonic mean)** của Precision và Recall. Tại sao không dùng trung bình số học bình thường?

Vì trung bình điều hòa **phạt nặng sự mất cân đối**: nếu một trong hai gần bằng 0, F1 sẽ gần bằng 0, bất kể cái kia cao thế nào. Điều này phản ánh đúng thực tế: một mô hình có Precision = 1.0 nhưng Recall = 0.01 là vô dụng trong thực tế.

> **Ví dụ step-by-step (từ slide, n=10):**
>
> $y = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]$ (index bắt đầu từ 1)
> $\hat{y} = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]$
>
> Đếm:
> - **TP = 4** (vị trí 1, 4, 6, 9: dự đoán đúng là dương)
> - **FN = 1** (vị trí 3: thực tế dương nhưng đoán âm)
> - **FP = 1** (vị trí 7: thực tế âm nhưng đoán dương)
> - **TN = 4** (các vị trí âm còn lại đoán đúng)
>
> $$\text{Accuracy} = \frac{4+4}{10} = 0.8$$
>
> $$\text{Precision} = \frac{4}{4+1} = 0.8, \quad \text{Recall} = \frac{4}{4+1} = 0.8$$
>
> $$F_1 = \frac{2 \times 0.8 \times 0.8}{0.8 + 0.8} = 0.8$$
>
> *Trong ví dụ này TP = FN = FP ngẫu nhiên bằng nhau nên các metric trùng nhau. Thực tế chúng thường khác nhau.*

---

### 2.5 Specificity và Balanced Accuracy

$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Specificity (True Negative Rate)** đo khả năng của mô hình trong việc xác định đúng các mẫu **âm tính**. Đây là "Recall cho lớp âm".

$$\text{Balanced Accuracy} = \frac{1}{2}(\text{Recall} + \text{Specificity})$$

**Balanced Accuracy** lấy trung bình của Recall (hiệu suất trên lớp dương) và Specificity (hiệu suất trên lớp âm) — xử lý cả hai lớp một cách **đối xứng**.

**Tại sao cần Balanced Accuracy?** Khi dữ liệu mất cân bằng (99 mẫu âm, 1 mẫu dương), Accuracy có thể cao nhờ TN. Nhưng Balanced Accuracy buộc mô hình phải thực sự tốt ở cả hai lớp.

---

### 2.6 Ngưỡng $\tau$ và Sự đánh đổi Precision–Recall

Mô hình cho ra xác suất $\hat{p}$, nhưng ta phải chọn ngưỡng $\tau$ để ra nhãn nhị phân:

$$\hat{y} = \mathbf{1}[\hat{p} \geq \tau]$$

**Tăng $\tau$ thường làm tăng Precision nhưng giảm Recall, và ngược lại.** Đây là một sự đánh đổi cơ bản:

- **$\tau$ cao:** Chỉ dự đoán "dương" khi rất tự tin → ít FP hơn (Precision tăng) nhưng bỏ sót nhiều hơn (Recall giảm).
- **$\tau$ thấp:** Dự đoán "dương" thường xuyên hơn → bắt được nhiều hơn (Recall tăng) nhưng nhiều false alarm hơn (Precision giảm).

> **Ví dụ (từ slide, n=6):**
>
> $y = [1, 1, 1, 0, 0, 0]$, $\hat{p} = [0.9, 0.6, 0.4, 0.7, 0.3, 0.2]$
>
> | Ngưỡng $\tau$ | Dự đoán $\hat{y}$ | TP | FN | FP | TN | Precision | Recall |
> |--------------|-------------------|----|----|----|----|-----------|--------|
> | $\tau = 0.5$ | [1,1,0,1,0,0]     | 2  | 1  | 1  | 2  | 0.667     | 0.667  |
> | $\tau = 0.8$ | [1,0,0,0,0,0]     | 1  | 2  | 0  | 3  | **1.000** | **0.333** |
>
> Tại $\tau = 0.8$: Precision hoàn hảo (không báo nhầm) nhưng bỏ sót 2/3 dương tính thực sự.

**Chọn $\tau$ như thế nào?** Phụ thuộc vào **chi phí của FP vs FN trong ứng dụng cụ thể**:
- Phát hiện ung thư → FN cực kỳ nguy hiểm → cần Recall cao → chọn $\tau$ thấp.
- Lọc nội dung độc hại → FP (chặn nhầm) gây trải nghiệm xấu → cần Precision cao → chọn $\tau$ cao.

> 📸 **[Cần ảnh]:** Đường cong PR (Precision-Recall curve) — trục x là Recall, trục y là Precision, mỗi điểm trên đường ứng với một giá trị $\tau$. **PR-AUC** là diện tích dưới đường này. *(Trang 17–18 slide; hoặc sklearn `precision_recall_curve`)*

---

### 2.7 Metrics Đa lớp: Macro vs Micro

Khi có $K > 2$ lớp, ta tính Precision/Recall/F1 cho từng lớp, rồi tổng hợp lại bằng hai cách:

#### Macro-average

$$\text{MacroF1} = \frac{1}{K} \sum_{k=1}^{K} F1_k$$

Tính F1 riêng cho từng lớp, rồi **lấy trung bình không có trọng số**. Mỗi lớp được coi là quan trọng như nhau, bất kể số lượng mẫu.

**Khi nào dùng:** Khi tất cả các lớp đều quan trọng như nhau, kể cả lớp thiểu số.

#### Micro-average

$$\text{MicroF1} = \frac{2 \cdot \text{TP}_{\text{global}}}{2 \cdot \text{TP}_{\text{global}} + \text{FP}_{\text{global}} + \text{FN}_{\text{global}}}$$

Gộp toàn bộ TP, FP, FN của tất cả các lớp lại, rồi tính một F1 duy nhất. Lớp nhiều mẫu hơn sẽ có trọng số lớn hơn.

**Khi nào dùng:** Khi số lượng mẫu quan trọng — lớp nhiều mẫu hơn xứng đáng ảnh hưởng nhiều hơn đến metric tổng.

> **Ví dụ step-by-step (từ slide, K=3, n=8):**
>
> $y = [A, A, B, B, B, C, C, C]$, $\hat{y} = [A, B, B, B, C, C, A, C]$
>
> **Confusion matrix (hàng = thực tế, cột = dự đoán):**
>
> |       | Pred A | Pred B | Pred C |
> |-------|--------|--------|--------|
> | **True A** | 1      | 1      | 0      |
> | **True B** | 0      | 2      | 1      |
> | **True C** | 1      | 0      | 2      |
>
> **Tính theo từng lớp:**
>
> | Lớp | TP | FP | FN | Precision | Recall | F1    |
> |-----|----|----|-----|-----------|--------|-------|
> | A   | 1  | 1  | 1   | 0.500     | 0.500  | 0.500 |
> | B   | 2  | 1  | 1   | 0.667     | 0.667  | 0.667 |
> | C   | 2  | 1  | 1   | 0.667     | 0.667  | 0.667 |
>
> $$\text{MacroF1} = \frac{0.5 + 0.667 + 0.667}{3} \approx 0.611$$

> 📸 **[Cần ảnh]:** Confusion matrix đa lớp dạng heatmap 3×3. *(Trang 20–21 slide; hoặc sklearn `ConfusionMatrixDisplay`)*

---

## 3. Tổng kết

### Chọn metric nào trong thực tế?

**Bài toán hồi quy:**
- Báo cáo **MAE + RMSE + R²** để có góc nhìn toàn diện.
- Luôn so với baseline (dự đoán $\bar{y}$).

**Bài toán phân loại:**
- **Dữ liệu cân bằng:** Accuracy + F1 là đủ.
- **Dữ liệu mất cân bằng:** Precision, Recall, F1, PR-AUC (và nhớ vẽ confusion matrix).
- **Luôn chỉ rõ ngưỡng $\tau$** khi báo cáo hard label metrics.

### Triết lý sâu xa

> **"Metrics không chỉ là con số — chúng mã hóa chi phí của những sai lầm."**
>
> Chọn metric sai = tối ưu hóa sai mục tiêu. Một mô hình fraud detection tốt theo Accuracy có thể là thảm họa theo Recall. Hãy luôn hỏi: *"Chi phí của FP và FN trong bài toán này là gì?"* — câu trả lời sẽ chỉ ra metric nào quan trọng nhất.

---

## Ghi chú về ảnh cần đính kèm

| # | Mô tả ảnh | Trang slide | Gợi ý nguồn |
|---|-----------|-------------|-------------|
| 1 | Scatter plot $y$ vs $\hat{y}$ với đường 45° và residuals | Trang 4–7 | `matplotlib.pyplot.scatter` |
| 2 | Confusion matrix nhị phân dạng heatmap (TP/TN xanh, FP/FN đỏ) | Trang 12 | `sklearn.metrics.ConfusionMatrixDisplay` |
| 3 | Đường cong Precision–Recall (PR Curve) minh họa trade-off | Trang 17–18 | `sklearn.metrics.precision_recall_curve` |
| 4 | Confusion matrix đa lớp 3×3 (lớp A, B, C) dạng heatmap | Trang 20–21 | `sklearn.metrics.confusion_matrix` |

---

## 4. Bài Tập Tính Toán

> Tự làm trước khi mở đáp án.

---

### Bài 1 — Tính MAE, RMSE, R²

Một mô hình dự đoán điểm thi (thang 100) của 6 học sinh:

| Học sinh | $y_i$ (thực tế) | $\hat{y}_i$ (dự đoán) |
|----------|-----------------|----------------------|
| 1 | 70 | 72 |
| 2 | 85 | 80 |
| 3 | 60 | 58 |
| 4 | 90 | 95 |
| 5 | 50 | 49 |
| 6 | 78 | 71 |

**(a)** Tính MAE.

**(b)** Tính RMSE.

**(c)** Tính R². Mô hình giải thích được bao nhiêu phần trăm phương sai dữ liệu?

**(d)** Nếu có học sinh thứ 7 với $y_7 = 55$, $\hat{y}_7 = 90$ (outlier), RMSE thay đổi như thế nào? MAE thay đổi như thế nào? Từ đó rút ra nhận xét.

<details>
<summary>📋 Đáp án Bài 1</summary>

Phần dư:

| HS | $e_i = y_i - \hat{y}_i$ | $\|e_i\|$ | $e_i^2$ |
|----|------------------------|-----------|---------|
| 1 | $-2$ | $2$ | $4$ |
| 2 | $5$ | $5$ | $25$ |
| 3 | $2$ | $2$ | $4$ |
| 4 | $-5$ | $5$ | $25$ |
| 5 | $1$ | $1$ | $1$ |
| 6 | $7$ | $7$ | $49$ |
| **Tổng** | | **22** | **108** |

**(a) MAE:**
$$\text{MAE} = \frac{22}{6} \approx \mathbf{3.67}$$

**(b) RMSE:**
$$\text{MSE} = \frac{108}{6} = 18, \qquad \text{RMSE} = \sqrt{18} \approx \mathbf{4.24}$$

**(c) R²:**

$$\bar{y} = \frac{70+85+60+90+50+78}{6} = \frac{433}{6} \approx 72.17$$

$$\text{SST} = (70{-}72.17)^2 + (85{-}72.17)^2 + (60{-}72.17)^2 + (90{-}72.17)^2 + (50{-}72.17)^2 + (78{-}72.17)^2$$
$$= 4.71 + 164.41 + 148.11 + 317.21 + 491.21 + 34.01 = 1159.66$$

$$R^2 = 1 - \frac{108}{1159.66} \approx 1 - 0.093 \approx \mathbf{0.907}$$

Mô hình giải thích được **90.7%** phương sai — khá tốt.

**(d) Thêm học sinh 7** ($e_7 = 55 - 90 = -35$, $|e_7| = 35$, $e_7^2 = 1225$):

- **RMSE mới:** $\text{MSE} = (108 + 1225)/7 = 190.4$, $\text{RMSE} = \sqrt{190.4} \approx \mathbf{13.8}$ (tăng từ 4.24 → 13.8, gấp **3.3 lần**!)
- **MAE mới:** $(22 + 35)/7 = 57/7 \approx \mathbf{8.1}$ (tăng từ 3.67 → 8.1, gấp **2.2 lần**)

**Nhận xét:** Một outlier duy nhất (35 điểm sai) kéo RMSE tăng gấp 3.3× nhưng MAE chỉ tăng 2.2×. RMSE nhạy cảm hơn MAE gần **50%** với outlier này — do bình phương khuếch đại $35^2 = 1225$ so với $35$.

</details>

---

### Bài 2 — Xây dựng Confusion Matrix từ dữ liệu thô

Một hệ thống phát hiện email spam. Nhãn thực tế và dự đoán trên 12 email:

```
y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
y_pred = [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1]
(1 = spam, 0 = không spam)
```

**(a)** Điền vào confusion matrix 2×2.

**(b)** Tính Accuracy, Precision, Recall, F1.

**(c)** Trong bài toán lọc spam, FP (chặn nhầm email quan trọng) và FN (để lọt spam) cái nào nguy hiểm hơn? Từ đó metric nào nên được ưu tiên?

<details>
<summary>📋 Đáp án Bài 2</summary>

**(a) Đếm từng vị trí** (so sánh y_true vs y_pred từng cặp):

| Pos | y_true | y_pred | Loại |
|-----|--------|--------|------|
| 1 | 1 | 1 | **TP** |
| 2 | 0 | 0 | **TN** |
| 3 | 1 | 0 | **FN** |
| 4 | 1 | 1 | **TP** |
| 5 | 0 | 1 | **FP** |
| 6 | 0 | 0 | **TN** |
| 7 | 1 | 1 | **TP** |
| 8 | 0 | 0 | **TN** |
| 9 | 1 | 0 | **FN** |
| 10 | 0 | 0 | **TN** |
| 11 | 0 | 1 | **FP** |
| 12 | 1 | 1 | **TP** |

→ **TP = 4, FN = 2, FP = 2, TN = 4**

**Confusion Matrix:**

|  | Pred Spam (1) | Pred Not (0) |
|--|--------------|--------------|
| **True Spam (1)** | TP = 4 | FN = 2 |
| **True Not (0)** | FP = 2 | TN = 4 |

**(b) Metrics:**

$$\text{Accuracy} = \frac{TP+TN}{12} = \frac{4+4}{12} = \frac{8}{12} \approx \mathbf{0.667}$$

$$\text{Precision} = \frac{TP}{TP+FP} = \frac{4}{4+2} = \frac{4}{6} \approx \mathbf{0.667}$$

$$\text{Recall} = \frac{TP}{TP+FN} = \frac{4}{4+2} = \frac{4}{6} \approx \mathbf{0.667}$$

$$F_1 = \frac{2 \times 0.667 \times 0.667}{0.667 + 0.667} = \mathbf{0.667}$$

**(c) FP vs FN trong lọc spam:**

- **FP** = chặn nhầm email quan trọng (hợp đồng kinh doanh, thư từ sếp) → người dùng mất thông tin quan trọng, **rất nguy hiểm** về mặt trải nghiệm.
- **FN** = để lọt spam vào hộp thư → phiền nhưng người dùng có thể xóa tay.

→ Trong lọc spam, **FP nguy hiểm hơn** → cần **Precision cao** → tăng ngưỡng $\tau$ để chỉ chặn khi rất tự tin là spam.

*(Ngược lại trong phát hiện ung thư: FN nguy hiểm hơn → cần Recall cao.)*

</details>

---

### Bài 3 — Tác động của ngưỡng $\tau$

Một mô hình phân loại trên 8 mẫu cho ra xác suất:

| Mẫu | $y$ | $\hat{p}$ |
|-----|-----|-----------|
| 1 | 1 | 0.92 |
| 2 | 1 | 0.75 |
| 3 | 1 | 0.61 |
| 4 | 1 | 0.38 |
| 5 | 0 | 0.85 |
| 6 | 0 | 0.55 |
| 7 | 0 | 0.29 |
| 8 | 0 | 0.10 |

**(a)** Tính Precision và Recall tại ngưỡng $\tau = 0.5$ và $\tau = 0.8$.

**(b)** Điền vào bảng sau rồi phác thảo đường cong PR:

| $\tau$ | TP | FP | FN | TN | Precision | Recall |
|--------|----|----|----|----|-----------|--------|
| 0.3 | ? | ? | ? | ? | ? | ? |
| 0.5 | ? | ? | ? | ? | ? | ? |
| 0.8 | ? | ? | ? | ? | ? | ? |
| 0.95 | ? | ? | ? | ? | ? | ? |

**(c)** Nếu bài toán này là phát hiện gian lận ngân hàng, nên chọn ngưỡng nào? Tại sao?

<details>
<summary>📋 Đáp án Bài 3</summary>

**Sắp xếp theo $\hat{p}$ giảm dần:**  
Mẫu 1 (0.92, y=1), 5 (0.85, y=0), 2 (0.75, y=1), 6 (0.55, y=0), 3 (0.61, y=1), 4 (0.38, y=1), 7 (0.29, y=0), 8 (0.10, y=0).

Tại mỗi ngưỡng, dự đoán dương nếu $\hat{p} \geq \tau$:

**$\tau = 0.95$:** Chỉ mẫu có $\hat{p} \geq 0.95$ → không có mẫu nào → TP=0, FP=0, FN=4, TN=4

**$\tau = 0.8$:** Dương: mẫu 1 (0.92), mẫu 5 (0.85) → TP=1 (mẫu 1), FP=1 (mẫu 5), FN=3, TN=3

**$\tau = 0.5$:** Dương: mẫu 1,5,2,6,3 ($\hat{p} \geq 0.5$) → TP=3 (1,2,3), FP=2 (5,6), FN=1 (4), TN=2

**$\tau = 0.3$:** Dương: mẫu 1,5,2,6,3,4 ($\hat{p} \geq 0.3$) → TP=4 (1,2,3,4), FP=2 (5,6), FN=0, TN=2

**(b) Bảng đầy đủ:**

| $\tau$ | TP | FP | FN | TN | Precision | Recall |
|--------|----|----|----|----|-----------|--------|
| 0.95 | 0 | 0 | 4 | 4 | undef (0/0) | 0.00 |
| 0.80 | 1 | 1 | 3 | 3 | **0.500** | **0.250** |
| 0.50 | 3 | 2 | 1 | 2 | **0.600** | **0.750** |
| 0.30 | 4 | 2 | 0 | 2 | **0.667** | **1.000** |

**Đường cong PR** (Recall → Precision):
```
Precision
1.0 |   *τ=0.95 (Recall=0, undefined Precision)
0.7 |      * τ=0.30 (P=0.667, R=1.0)
0.6 |   * τ=0.50 (P=0.6, R=0.75)
0.5 |      * τ=0.80 (P=0.5, R=0.25)
    └─────────────────────────── Recall
      0   0.25  0.5  0.75  1.0
```

Đường đi từ phải (Recall cao, Precision thấp hơn) sang trái (Recall thấp, Precision cao hơn) khi tăng $\tau$.

**(c) Phát hiện gian lận ngân hàng:**

FN = bỏ sót giao dịch gian lận → tiền bị mất, thiệt hại thực → **cực kỳ nguy hiểm**.

→ Cần **Recall cao** → chọn **$\tau = 0.3$** (Recall = 1.0, không bỏ sót gian lận nào).

Đánh đổi: Precision = 0.667 → 33% cảnh báo là nhầm, nhưng chấp nhận được vì nhân viên kiểm tra thêm còn tốt hơn để mất tiền thật.

</details>

---

### Bài 4 — Balanced Accuracy và Dữ liệu Mất cân bằng

Bài toán phát hiện khuyết tật sản phẩm trên dây chuyền: 950 sản phẩm tốt, 50 sản phẩm lỗi.

**Mô hình A** (luôn dự đoán "tốt"):  
TP=0, FN=50, FP=0, TN=950

**Mô hình B** (mô hình thực sự học):  
TP=40, FN=10, FP=30, TN=920

**(a)** Tính Accuracy, Recall, Specificity, và Balanced Accuracy cho cả hai mô hình.

**(b)** Mô hình nào tốt hơn? Metric nào phản ánh đúng nhất?

**(c)** Tính F1 score cho Mô hình B. F1 có ưu điểm gì so với Accuracy trong trường hợp này?

<details>
<summary>📋 Đáp án Bài 4</summary>

**Mô hình A** (luôn đoán "tốt", TP=0, FN=50, FP=0, TN=950):

$$\text{Accuracy}_A = \frac{0+950}{1000} = \mathbf{0.950}$$
$$\text{Recall}_A = \frac{0}{0+50} = \mathbf{0.000}$$
$$\text{Specificity}_A = \frac{950}{950+0} = \mathbf{1.000}$$
$$\text{Balanced Accuracy}_A = \frac{0 + 1}{2} = \mathbf{0.500}$$

**Mô hình B** (TP=40, FN=10, FP=30, TN=920):

$$\text{Accuracy}_B = \frac{40+920}{1000} = \mathbf{0.960}$$
$$\text{Recall}_B = \frac{40}{40+10} = \mathbf{0.800}$$
$$\text{Specificity}_B = \frac{920}{920+30} \approx \mathbf{0.968}$$
$$\text{Balanced Accuracy}_B = \frac{0.8 + 0.968}{2} \approx \mathbf{0.884}$$

**(b) So sánh:**

| Metric | Mô hình A | Mô hình B |
|--------|-----------|-----------|
| Accuracy | 0.950 | **0.960** |
| Recall | 0.000 | **0.800** |
| Balanced Accuracy | 0.500 | **0.884** |

Mô hình **B tốt hơn nhiều**, nhưng **Accuracy gây nhầm lẫn**: A có Accuracy 95% dù bỏ sót 100% sản phẩm lỗi. **Balanced Accuracy** phản ánh đúng nhất: A chỉ đạt 0.5 (tương đương đoán mò) trong khi B đạt 0.884.

**(c) F1 của Mô hình B:**

$$\text{Precision}_B = \frac{40}{40+30} \approx 0.571$$
$$F_{1,B} = \frac{2 \times 0.571 \times 0.8}{0.571 + 0.8} \approx \frac{0.914}{1.371} \approx \mathbf{0.667}$$

**Ưu điểm F1 so với Accuracy:** F1 = 0.667 (B) vs Accuracy = 0.950 (A, mô hình vô dụng) — F1 nhận ra rõ B tốt hơn A. Với mô hình A: TP=0 → Precision=undef, Recall=0 → F1=0. Accuracy không phân biệt được, F1 thì có.

</details>

---

### Bài 5 — Macro vs Micro Averaging

Bài toán phân loại 4 lớp (A, B, C, D) trên 100 mẫu. Confusion matrix:

|  | Pred A | Pred B | Pred C | Pred D |
|--|--------|--------|--------|--------|
| **True A** | 28 | 2 | 0 | 0 |
| **True B** | 3 | 17 | 0 | 0 |
| **True C** | 0 | 1 | 9 | 0 |
| **True D** | 0 | 0 | 2 | 38 |

**(a)** Tính TP, FP, FN cho từng lớp.

**(b)** Tính Precision, Recall, F1 cho từng lớp.

**(c)** Tính MacroF1 và MicroF1. Tại sao hai giá trị này khác nhau?

<details>
<summary>📋 Đáp án Bài 5</summary>

**(a) TP, FP, FN theo từng lớp:**

Từ confusion matrix (hàng = true, cột = pred):
- $TP_k$ = ô chéo chính của lớp k
- $FP_k$ = tổng cột k − $TP_k$ (các lớp khác bị đoán là k)
- $FN_k$ = tổng hàng k − $TP_k$ (lớp k bị đoán nhầm sang lớp khác)

| Lớp | $TP$ | $FP$ | $FN$ | Tổng thực |
|-----|------|------|------|-----------|
| A | 28 | 3 (từ B→A) | 2 (A→B) | 30 |
| B | 17 | 3 (A→B + C→B = 2+1) | 3 (B→A) | 20 |
| C | 9 | 2 (D→C) | 1 (C→B) | 10 |
| D | 38 | 0 | 2 (D→C) | 40 |

**(b) Precision, Recall, F1:**

$$P_k = \frac{TP_k}{TP_k+FP_k}, \quad R_k = \frac{TP_k}{TP_k+FN_k}, \quad F_k = \frac{2P_kR_k}{P_k+R_k}$$

| Lớp | Precision | Recall | F1 |
|-----|-----------|--------|----|
| A | $28/(28+3) = 0.903$ | $28/(28+2) = 0.933$ | $0.918$ |
| B | $17/(17+3) = 0.850$ | $17/(17+3) = 0.850$ | $0.850$ |
| C | $9/(9+2) = 0.818$ | $9/(9+1) = 0.900$ | $0.857$ |
| D | $38/(38+0) = 1.000$ | $38/(38+2) = 0.950$ | $0.974$ |

**(c) MacroF1 và MicroF1:**

$$\text{MacroF1} = \frac{0.918 + 0.850 + 0.857 + 0.974}{4} \approx \mathbf{0.900}$$

Gộp toàn bộ TP, FP, FN:

$$TP_{global} = 28+17+9+38 = 92$$
$$FP_{global} = 3+3+2+0 = 8$$
$$FN_{global} = 2+3+1+2 = 8$$

$$\text{MicroF1} = \frac{2 \times 92}{2 \times 92 + 8 + 8} = \frac{184}{200} = \mathbf{0.920}$$

**Tại sao khác nhau?**
- **MacroF1 = 0.900:** Trung bình F1 của 4 lớp theo trọng số bằng nhau — lớp C (chỉ 10 mẫu) ảnh hưởng ngang bằng lớp D (40 mẫu).
- **MicroF1 = 0.920:** Lớp D có 40 mẫu và F1 = 0.974 → **kéo MicroF1 lên cao hơn** vì đóng góp TP nhiều hơn vào global.

**Khi dùng:**
- Lớp C nhỏ nhưng quan trọng (ví dụ: lớp hiếm trong y tế) → dùng **MacroF1** để C được tính ngang bằng.
- Tổng số mẫu đúng mới quan trọng (ví dụ: hệ thống tổng hợp) → dùng **MicroF1**.

</details>

---

### Bài 6 — Câu hỏi tư duy nhanh

**(a)** Một mô hình đạt Accuracy = 99% trên bộ test. Không cần biết thêm thông tin nào khác, bạn có thể kết luận mô hình tốt không? Tại sao?

**(b)** Precision và Recall cùng tăng đồng thời có thể xảy ra không? Khi nào?

**(c)** F1 score của một mô hình luôn dự đoán dương (y_pred = 1 tất cả) là bao nhiêu? Biểu diễn theo TP, FP, FN, TN.

**(d)** Trong bài toán 3 lớp cân bằng (mỗi lớp 100 mẫu), MicroF1 và MacroF1 có bằng nhau không? Tại sao?

<details>
<summary>📋 Đáp án Bài 6</summary>

**(a) Accuracy = 99% không đủ để kết luận:**

Không. Cần biết **phân phối lớp** (class distribution). Nếu 99% mẫu thuộc lớp âm, một mô hình luôn đoán âm đạt Accuracy = 99% mà không học gì. Phải hỏi thêm: tỉ lệ lớp dương/âm là bao nhiêu? Recall của lớp thiểu số là gì?

**(b) Precision và Recall cùng tăng:**

Có thể xảy ra khi **cải thiện chất lượng mô hình** (ví dụ: thêm dữ liệu, regularization tốt hơn) — mô hình đồng thời ít báo nhầm hơn (FP giảm → Precision tăng) và ít bỏ sót hơn (FN giảm → Recall tăng).

Ngược lại, khi **chỉ thay đổi ngưỡng $\tau$**, thường là Precision và Recall **đánh đổi nhau** — tăng cái này giảm cái kia. Cả hai cùng tăng chỉ khi cải thiện mô hình thực sự.

**(c) Mô hình luôn đoán dương ($\hat{y} = 1$ tất cả):**

- FN = 0 (không bỏ sót dương nào vì đoán tất cả là dương)
- TN = 0 (không có TN vì không bao giờ đoán âm)
- FP = tổng số mẫu âm

$$\text{Recall} = \frac{TP}{TP+FN} = \frac{TP}{TP+0} = 1.0$$
$$\text{Precision} = \frac{TP}{TP+FP} = \frac{TP}{TP+\text{(số mẫu âm)}}$$

Nếu tỉ lệ dương = $r$:

$$F_1 = \frac{2 \times r \times 1.0}{r + 1.0} = \frac{2r}{1+r}$$

Ví dụ: 10% mẫu dương ($r = 0.1$): $F_1 = 2×0.1/1.1 \approx 0.182$ — thấp dù Recall = 1.

**Đây là lý do F1 không bị đánh lừa bởi mô hình "luôn đoán dương"**, khác với Recall thuần túy.

**(d) 3 lớp cân bằng — MicroF1 vs MacroF1:**

Khi mỗi lớp có cùng số mẫu (balanced), **MicroF1 = MacroF1**.

Lý do: MicroF1 gộp TP/FP/FN với trọng số theo số mẫu, MacroF1 lấy trung bình F1 các lớp. Khi tất cả lớp có cùng kích thước, hai cách tính cho cùng kết quả — "trọng số theo mẫu" = "trọng số bằng nhau".

Hai chỉ số **chỉ khác nhau khi phân phối lớp mất cân bằng**.

</details>

---

### Tổng hợp công thức và mẹo thi

| Metric | Công thức | Mẹo nhớ nhanh |
|--------|----------|---------------|
| **MAE** | $\frac{1}{n}\sum\|e_i\|$ | Mean of absolute residuals — cùng đơn vị $y$ |
| **RMSE** | $\sqrt{\frac{1}{n}\sum e_i^2}$ | Căn bậc hai của MSE — cùng đơn vị $y$ |
| **R²** | $1 - \text{SSE}/\text{SST}$ | 1 = hoàn hảo, 0 = bằng baseline, <0 = tệ hơn baseline |
| **Accuracy** | $(TP+TN)/N$ | Tỉ lệ đúng tổng thể — sai lệch khi imbalanced |
| **Precision** | $TP/(TP+FP)$ | "Khi nói dương, có đúng không?" — liên quan FP |
| **Recall** | $TP/(TP+FN)$ | "Dương thật có bị bỏ sót không?" — liên quan FN |
| **Specificity** | $TN/(TN+FP)$ | Recall của lớp âm |
| **F1** | $2PR/(P+R)$ | Harmonic mean — phạt nặng nếu P hoặc R gần 0 |
| **Balanced Acc** | $(Recall + Specificity)/2$ | F1 cho cả lớp âm — dùng khi imbalanced |
| **MacroF1** | $\frac{1}{K}\sum F1_k$ | Mỗi lớp quan trọng như nhau (dùng cho lớp hiếm) |
| **MicroF1** | $2TP_{global}/(2TP_{global}+FP_{global}+FN_{global})$ | Lớp nhiều mẫu ảnh hưởng nhiều hơn |

**Nguyên tắc chọn metric:**

```
Imbalanced data?  →  Tránh Accuracy thuần túy
FN nguy hiểm?     →  Tối ưu Recall (y tế, gian lận)
FP nguy hiểm?     →  Tối ưu Precision (spam filter, content moderation)
Muốn cân bằng?   →  F1 hoặc Balanced Accuracy
Lớp hiếm quan trọng?  →  MacroF1
```
