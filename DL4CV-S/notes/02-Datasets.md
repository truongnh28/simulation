# Bài 02: Datasets — Dữ Liệu trong Machine Learning

> **Nguồn slide:** `slides-v1/foundation/02-Datasets.pdf`
> **Giảng viên:** Thanh-Sach LE — ltsach@hcmut.edu.vn
> **Khoa KHMT, ĐHBK TP.HCM — 05/02/2026**

---

## Mục Lục

1. [Dataset là gì?](#1-dataset-là-gì)
2. [Sample, Feature, và Label](#2-sample-feature-và-label)
3. [Dataset theo Loại Bài Toán](#3-dataset-theo-loại-bài-toán)
4. [Các Kiểu Dữ Liệu trong Deep Learning](#4-các-kiểu-dữ-liệu-trong-deep-learning)
5. [Dữ Liệu Ảnh — Shape](#5-dữ-liệu-ảnh--shape)
6. [Dữ Liệu Văn Bản — Shape](#6-dữ-liệu-văn-bản--shape)
7. [Dữ Liệu Chuỗi Thời Gian — Shape](#7-dữ-liệu-chuỗi-thời-gian--shape)
8. [Dữ Liệu Bảng — Matrix View](#8-dữ-liệu-bảng--matrix-view)
9. [Chất Lượng Dữ Liệu](#9-chất-lượng-dữ-liệu)
10. [Chuẩn Hóa và Mã Hóa](#10-chuẩn-hóa-và-mã-hóa)
11. [Phân Chia Train / Val / Test](#11-phân-chia-train--val--test)
12. [Stratified Split](#12-stratified-split)
13. [Cross-Validation](#13-cross-validation)
14. [Phân Chia Dữ Liệu Thời Gian](#14-phân-chia-dữ-liệu-thời-gian)
15. [Mini-batch và Training Loop](#15-mini-batch-và-training-loop)
16. [Shuffling](#16-shuffling)
17. [Data Leakage](#17-data-leakage)
18. [Class Imbalance](#18-class-imbalance)
19. [Data Augmentation](#19-data-augmentation)
20. [Benchmarks và Nguồn Dữ Liệu](#20-benchmarks-và-nguồn-dữ-liệu)
21. [Dataset và DataLoader trong PyTorch](#21-dataset-và-dataloader-trong-pytorch)

---

## 1. Dataset là gì?

> 📌 **Ảnh:** Slide trang 3 — `02-Datasets.pdf#page=3`
> Hộp "Definition" với D = {(x_i, y_i)}^N_{i=1}; hộp "Key View" với ML = learning f: x → y from data.

Về mặt toán học, một **dataset** là tập hợp các cặp (đầu vào, đầu ra):

$$\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^{N}$$

- $\mathbf{x}_i$: **feature vector** — thể hiện thông tin về mẫu thứ $i$
- $\mathbf{y}_i$: **target / label** — câu trả lời đúng cho mẫu thứ $i$
- $N$: tổng số mẫu

**Quan điểm cốt lõi:** Machine learning = học hàm $f: \mathbf{x} \to \mathbf{y}$ từ dữ liệu.

> 💡 Dataset không chỉ là "tập hợp dữ liệu thô" — nó phải bao gồm cả **nhãn** (với supervised learning) và phải đại diện cho phân phối thực tế bạn muốn mô hình hoạt động tốt.

---

## 2. Sample, Feature, và Label

> 📌 **Ảnh:** Slide trang 4 — `02-Datasets.pdf#page=4`
> Hộp "Single Sample" với vector cột x = [x_1, x_2, ..., x_d]^T ∈ ℝ^{d×1} và 3 bullet points.

Phân biệt ba khái niệm cơ bản:

Một **mẫu đơn lẻ** (single sample) là một vector cột:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} \in \mathbb{R}^{d \times 1}$$

| Khái niệm | Định nghĩa | Ví dụ |
|---|---|---|
| **Feature** (đặc trưng) | Một thuộc tính đo được của mẫu | Chiều cao, giá trị pixel $(i,j)$, tần số âm thanh |
| **Sample** (mẫu) | Một quan sát hoàn chỉnh — toàn bộ vector $\mathbf{x}$ | Một ảnh khuôn mặt, một câu văn bản |
| **Label** (nhãn) | Đầu ra mong muốn $y$ tương ứng với $\mathbf{x}$ | Tên người, nhãn lớp, giá trị giá nhà |

> 💡 **Ví dụ:** Bài toán nhận dạng khuôn mặt: sample = một ảnh, feature = giá trị pixel, label = tên người.

---

## 3. Dataset theo Loại Bài Toán

> 📌 **Ảnh trang 5 (Regression):** `02-Datasets.pdf#page=5`
> "Form: D = {(x_i, y_i)}, Target is scalar y_i ∈ ℝ, Example: house price, temperature, age."

> 📌 **Ảnh trang 6 (Classification):** `02-Datasets.pdf#page=6`
> "Form: D = {(x_i, y_i)}, Target is class label y_i ∈ {1,...,K}, Binary K=2, Multiclass K>2."

### Dataset cho Regression

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N \quad \text{với } y_i \in \mathbb{R}$$

- Target là một **số thực vô hướng** (scalar)
- Ví dụ: dự đoán giá nhà, nhiệt độ, tuổi tác

### Dataset cho Classification

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N \quad \text{với } y_i \in \{1, \ldots, K\}$$

- Target là một **nhãn lớp** (class label)
- **Binary**: $K = 2$ (spam/không spam)
- **Multiclass**: $K > 2$ (phân loại 10 chữ số, 1000 loài)

---

## 4. Các Kiểu Dữ Liệu trong Deep Learning

> 📌 **Ảnh:** Slide trang 7 — `02-Datasets.pdf#page=7`
> Hộp "Beyond tabular vectors" (Tabular/Images/Sequences/Graphs) và "Supervised vs unsupervised".

Một điểm mạnh của deep learning là xử lý **trực tiếp dữ liệu thô** mà không cần feature engineering thủ công.

| Kiểu dữ liệu | Biểu diễn toán học | Ứng dụng |
|---|---|---|
| **Tabular** | $\mathbf{x}_i \in \mathbb{R}^d$ | Dữ liệu bệnh nhân, tài chính — ML cổ điển |
| **Images** | $\mathbf{x}_i \in \mathbb{R}^{C \times H \times W}$ | Nhận dạng ảnh, phát hiện vật thể (CNN) |
| **Sequences** | $\mathbf{x}_i = (x_1, \ldots, x_T)$ | Văn bản, chuỗi thời gian, âm thanh (RNN/Transformer) |
| **Graphs** | Nodes + edges + features | Mạng xã hội, phân tử hóa học (GNN) |

**Supervised vs Unsupervised:** Nhãn $y_i$ có thể **không tồn tại** — clustering, self-supervised learning, generative models hoạt động trực tiếp trên $\mathbf{x}_i$.

---

## 5. Dữ Liệu Ảnh — Shape

> 📌 **Ảnh:** Slide trang 8 — `02-Datasets.pdf#page=8`
> Hộp "Single image" (x_i ∈ ℝ^{C×H×W}, C/H/W, PyTorch NCHW) và "Batch of images" (X_batch ∈ ℝ^{B×C×H×W}, CIFAR-10 example shape (32,3,32,32)).

**Một ảnh đơn lẻ:**
$$\mathbf{x}_i \in \mathbb{R}^{C \times H \times W}$$

| Ký hiệu | Ý nghĩa | Ví dụ |
|---|---|---|
| $C$ | Số kênh màu | $3$ (RGB), $1$ (grayscale) |
| $H$ | Chiều cao (height) | $32$ pixels (CIFAR-10) |
| $W$ | Chiều rộng (width) | $32$ pixels (CIFAR-10) |

**PyTorch convention: NCHW** (channel-first). Lưu ý TensorFlow dùng NHWC (channel-last) — nguồn gốc nhiều lỗi khi chuyển code.

**Một batch ảnh:**
$$\mathbf{X}_{\text{batch}} \in \mathbb{R}^{B \times C \times H \times W}$$

GPU yêu cầu **4D tensor** này. Ví dụ: batch 32 ảnh CIFAR-10 → shape `(32, 3, 32, 32)`.

> 💡 B ảnh được xếp chồng dọc theo **chiều đầu tiên** (batch dimension).

---

## 6. Dữ Liệu Văn Bản — Shape

> 📌 **Ảnh:** Slide trang 9 — `02-Datasets.pdf#page=9`
> Hộp "Single sequence" (token IDs x_i ∈ ℝ^T; after embedding x_i ∈ ℝ^{T×D}) và "Batch of sequences" (pad to T_max, attention mask).

**Một chuỗi đơn lẻ:**
- Dạng token ID: $\mathbf{x}_i \in \mathbb{R}^T$ (độ dài $T$, thay đổi theo từng mẫu)
- Sau embedding: $\mathbf{x}_i \in \mathbb{R}^{T \times D}$ (với $D$ = embedding dimension)

**Thách thức:** Các câu có độ dài khác nhau. Giải pháp: **padding** — thêm token `[PAD]` để mọi chuỗi có cùng độ dài $T_{\max}$.

**Một batch chuỗi:**
$$\mathbf{X}_{\text{batch}} \in \mathbb{R}^{B \times T_{\max}} \quad \text{(token IDs)} \quad \text{hoặc} \quad \mathbb{R}^{B \times T_{\max} \times D} \quad \text{(embedded)}$$

Sau padding, dùng **attention mask** hoặc **lengths vector** để mô hình không tính loss trên token padding.

---

## 7. Dữ Liệu Chuỗi Thời Gian — Shape

> 📌 **Ảnh:** Slide trang 10 — `02-Datasets.pdf#page=10`
> Hộp "Single time series" (x_i ∈ ℝ^{T×F}, T = time steps, F = features) và "Batch of time series" (X_batch ∈ ℝ^{B×T×F}, pad if needed).

**Một chuỗi thời gian đơn lẻ:**
$$\mathbf{x}_i \in \mathbb{R}^{T \times F}$$

| Ký hiệu | Ý nghĩa | Ví dụ |
|---|---|---|
| $T$ | Số bước thời gian | 100 timesteps |
| $F$ | Số đặc trưng tại mỗi bước | Số kênh cảm biến; $F=1$ nếu univariate |

**Một batch time-series:**
$$\mathbf{X}_{\text{batch}} \in \mathbb{R}^{B \times T \times F}$$

Tương tự text: nếu chuỗi có độ dài khác nhau → padding rồi mask trong model.

---

## 8. Dữ Liệu Bảng — Matrix View

> 📌 **Ảnh:** Slide trang 11 — `02-Datasets.pdf#page=11`
> Ma trận X ∈ ℝ^{N×d} (stack x_i^T as rows) và vector y ∈ ℝ^N; hộp "Why Rows?" (PyTorch/Keras convention).

Với **feature vectors** $\mathbf{x}_i \in \mathbb{R}^d$, xếp thành hàng (row-wise):

$$X = \begin{bmatrix} \mathbf{x}_1^\top \\ \mathbf{x}_2^\top \\ \vdots \\ \mathbf{x}_N^\top \end{bmatrix} \in \mathbb{R}^{N \times d} \qquad \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix}$$

**Tại sao dùng hàng (Why Rows)?**

Khớp với PyTorch / Keras: mỗi hàng = một sample; các batch là các *dãy hàng liên tiếp* → tối ưu truy cập bộ nhớ (row-contiguous storage).

---

## 9. Chất Lượng Dữ Liệu

> 📌 **Ảnh:** Slide trang 13 — `02-Datasets.pdf#page=13`
> 5 bullet points: Missing values, Noise & outliers, Label noise, Duplicate samples, Data bias — mỗi cái có nguyên nhân và giải pháp tương ứng.

> 💡 **"Garbage in, garbage out."** Mô hình tốt nhất không thể cứu vãn được dữ liệu tệ.

| Vấn đề | Biểu hiện | Giải pháp |
|---|---|---|
| **Missing values** | NaN, ô trống | **Imputation** (mean/median/model-based) hoặc loại bỏ |
| **Noise & Outliers** | Lỗi đo lường, giá trị cực đoan | Robust scaling, clipping, loại bỏ |
| **Label noise** | Nhãn gán sai | Robust losses, data cleaning, label smoothing |
| **Duplicate samples** | Cùng mẫu xuất hiện nhiều lần | Deduplicate **trước khi** chia tập |
| **Data bias** | Nhóm bị đại diện kém | Cân nhắc về fairness, thu thập thêm dữ liệu |

> ⚠️ **Label noise** đặc biệt nguy hiểm trong supervised learning vì mô hình đang cố học một **ánh xạ sai**. Ví dụ: dataset X-quang được gán nhãn bởi nhiều bác sĩ có ý kiến khác nhau.

---

## 10. Chuẩn Hóa và Mã Hóa

> 📌 **Ảnh:** Slide trang 14 — `02-Datasets.pdf#page=14`
> Hộp "Normalization" với công thức x̃_j = (x_j − μ_j)/σ_j và chú thích "fit only on training set"; hộp "Categorical features" (one-hot or embeddings); hộp "Rule" (màu đỏ) — bất kỳ thống kê nào cũng phải tính từ training set.

### Chuẩn Hóa Z-score (Zero Mean, Unit Variance)

$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

- $\mu_j$: trung bình đặc trưng $j$ trên **training set**
- $\sigma_j$: độ lệch chuẩn đặc trưng $j$ trên **training set**

**Tại sao cần chuẩn hóa?** Gradient descent hoạt động tốt hơn khi các đặc trưng có cùng tầm giá trị. Nếu feature A range 0–1 và feature B range 0–1,000,000 → gradient bị dominated bởi B → mô hình khó hội tụ.

### Mã Hóa Đặc Trưng Phân Loại

- **One-hot encoding**: Mỗi giá trị → một vector binary riêng biệt
- **Embedding**: Học vector dense qua embedding layer (phổ biến trong DL)

### ⚠️ Quy Tắc Quan Trọng Nhất

> **Bất kỳ thống kê nào dùng trong preprocessing ($\mu_j$, $\sigma_j$, vocabulary, encoder...) đều phải được tính CHỈ từ training set, rồi áp dụng lên val/test — KHÔNG tính lại.**

Tính lại trên val/test = **data leakage**.

> 💡 **Ẩn dụ:** Bạn học chuẩn hóa điểm từ kết quả thi thử. Khi thi thật, dùng cùng thang đo đó — không tính lại từ bài thi thật (vì như vậy là đã "nhìn" vào đề).

---

## 11. Phân Chia Train / Val / Test

> 📌 **Ảnh:** Slide trang 16 — `02-Datasets.pdf#page=16`
> 3 bullet points: Training set (fit params), Validation set (tune, early stopping, model selection), Test set (final evaluation once). Typical split 70-15-15 hoặc 80-10-10.

Tỉ lệ chia phổ biến: **70%–15%–15%** hoặc **80%–10%–10%** (tùy $N$).

| Tập | Mục đích | Sử dụng khi nào |
|---|---|---|
| **Training set** | Fit (tối ưu) tham số mô hình | Mỗi bước gradient descent |
| **Validation set** | Tune hyperparameters, early stopping, model selection | Sau mỗi epoch (hoặc định kỳ) |
| **Test set** | Báo cáo hiệu năng cuối cùng | **Một lần duy nhất**, sau khi xong mọi quyết định |

> ⚠️ Nếu bạn nhìn vào kết quả test nhiều lần để thay đổi mô hình, bạn đã biến test thành validation — kết quả cuối không đáng tin cậy.

---

## 12. Stratified Split

> 📌 **Ảnh:** Slide trang 17 — `02-Datasets.pdf#page=17`
> "For classification, keep class proportions similar in train/val/test." Hộp "Stratified split" và 2 bullets: prevents rare class missing, use StratifiedKFold.

Khi phân chia dữ liệu phân loại, phải đảm bảo **tỉ lệ các lớp** trong mỗi tập gần bằng tỉ lệ trong toàn bộ dataset.

**Tại sao?** Nếu dataset có 95% mèo và 5% chó, mà tập test tình cờ không có con chó nào → kết quả test không phản ánh khả năng thực của mô hình.

**Trong code (sklearn):**
```python
from sklearn.model_selection import train_test_split, StratifiedKFold

# train_test_split với stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

---

## 13. Cross-Validation

> 📌 **Ảnh:** Slide trang 18 — `02-Datasets.pdf#page=18`
> 3 bullets về K-fold: partition into K folds, train on K-1 validate on 1, average metrics. Hộp "Note" (test set still held out, CV for tuning only).

Khi dataset nhỏ, **K-fold Cross-Validation** tận dụng tối đa dữ liệu:

1. Chia training data thành $K$ phần (folds)
2. Lặp $K$ lần: mỗi lần dùng 1 fold làm validation, $K-1$ folds còn lại làm training
3. Lấy **trung bình** kết quả validation → giảm phương sai của ước lượng

```
Fold 1: [VAL] [TRN] [TRN] [TRN] [TRN]
Fold 2: [TRN] [VAL] [TRN] [TRN] [TRN]
Fold 3: [TRN] [TRN] [VAL] [TRN] [TRN]
Fold 4: [TRN] [TRN] [TRN] [VAL] [TRN]
Fold 5: [TRN] [TRN] [TRN] [TRN] [VAL]
```

> ⚠️ **Lưu ý quan trọng:** Test set **vẫn giữ tách biệt hoàn toàn**. Cross-validation chỉ dùng để tune hyperparameter hoặc chọn kiến trúc — không phải để báo cáo kết quả cuối.

---

## 14. Phân Chia Dữ Liệu Thời Gian

> 📌 **Ảnh:** Slide trang 19 — `02-Datasets.pdf#page=19`
> "For temporal data, do NOT shuffle then split: use a temporal split." 2 bullets: train earlier/val next/test future, prevents future leaking into training.

Đối với time-series, **TUYỆT ĐỐI KHÔNG xáo trộn (shuffle)** rồi chia ngẫu nhiên.

Phải dùng **temporal split**:

```
Timeline: ──────────────────────────────────────────▶
           |    TRAIN (quá khứ)   | VAL | TEST (tương lai) |
```

- **Train**: các bước thời gian sớm hơn
- **Validation**: khoảng thời gian tiếp theo
- **Test**: khoảng thời gian trong tương lai

**Tại sao?** Mục tiêu thực sự là **dự đoán tương lai**. Nếu shuffle, mô hình có thể học từ dữ liệu "tương lai" để dự đoán "quá khứ" — đây là data leakage nghiêm trọng dẫn đến kết quả ảo.

---

## 15. Mini-batch và Training Loop

> 📌 **Ảnh:** Slide trang 21 — `02-Datasets.pdf#page=21`
> Hộp "Mini-batch" với X_b ∈ ℝ^{B×d}; 3 bullets: Faster (B vs N), Generalization (noise as regularization), SGD (standard in DL, batch size is key hyperparameter).

Thay vì dùng toàn bộ dataset (quá chậm) hoặc 1 mẫu (quá nhiễu), **mini-batch** là sự cân bằng thực dụng:

$$\mathbf{X}_b \in \mathbb{R}^{B \times d}$$

### Lợi Ích của Mini-batch

| Lợi ích | Giải thích |
|---|---|
| **Tốc độ** | Tính gradient trên $B$ mẫu thay vì $N$ — nhanh hơn nhiều |
| **Khái quát hóa** | "Nhiễu" trong mini-batch gradient = dạng **regularization ngầm** — giúp thoát local minima |
| **SGD** | Standard optimization trong DL; batch size là **hyperparameter quan trọng** |

**Trade-off batch size:**
- Batch **lớn**: gradient ổn định, hội tụ nhanh, nhưng tốn RAM GPU, dễ rơi vào *sharp minima* (generalize kém hơn)
- Batch **nhỏ**: nhiều nhiễu hơn, có thể generalize tốt hơn, nhưng chậm hơn

---

## 16. Shuffling

> 📌 **Ảnh:** Slide trang 22 — `02-Datasets.pdf#page=22`
> 3 bullets: Randomly permute each epoch, Prevents model from learning order patterns, Improves convergence.

Sau mỗi **epoch**, ta **xáo trộn ngẫu nhiên** thứ tự các mẫu trước khi tạo batch mới.

**Tại sao?**
- Ngăn mô hình học **thứ tự** của dữ liệu thay vì học nội dung
- Đảm bảo mỗi batch mang tính đại diện hơn cho toàn bộ distribution
- Giúp quá trình hội tụ ổn định hơn

> ⚠️ Shuffle **chỉ áp dụng cho training set**. Validation và test không cần shuffle (kết quả không phụ thuộc thứ tự).

---

## 17. Data Leakage

> 📌 **Ảnh:** Slide trang 23 — `02-Datasets.pdf#page=23`
> Hộp "Definition" (màu đỏ): "Information from test set leaks into training." 2 ví dụ: normalize using whole dataset, augment after splitting incorrectly. Hộp "Rule": test set untouched until final.

### Định Nghĩa

**Data leakage** xảy ra khi thông tin từ test set "rò rỉ" vào quá trình huấn luyện.

### Hai Ví Dụ Điển Hình

1. **Chuẩn hóa dùng toàn bộ dataset:** Tính $\mu$ và $\sigma$ trên cả train+test, sau đó áp dụng cho tất cả → mô hình đã "nhìn thấy" thống kê của test set.

2. **Augment trước khi split:** Augment data trước, sau đó chia → các bản sao của cùng một mẫu xuất hiện ở cả train và test.

**Thứ tự đúng:**
```
Đúng:  Split → Fit preprocessing on TRAIN → Apply to all
Sai:   Fit preprocessing on ALL → Split
```

### ⚠️ Quy Tắc

> **Test set phải được giữ nguyên (untouched) cho đến khi đánh giá cuối cùng.**

**Hậu quả của leakage:** Kết quả test trông rất tốt nhưng khi deploy ra thực tế sẽ thất bại — đây là một trong những nguyên nhân phổ biến nhất khiến mô hình "hoạt động tốt trong lab nhưng thất bại ngoài đời thực".

---

## 18. Class Imbalance

> 📌 **Ảnh:** Slide trang 25 — `02-Datasets.pdf#page=25`
> 4 bullets: model predicts majority (accuracy misleading), use Precision/Recall/F1/AUC, Data-level (SMOTE/undersampling), Algorithm-level (class weights/focal loss/threshold tuning).

### Vấn Đề

Khi một hoặc nhiều lớp có số lượng mẫu ít hơn nhiều so với các lớp khác:

- Mô hình học cách *lười biếng* — luôn dự đoán lớp đa số
- **Accuracy bị đánh lừa**: 99% accuracy khi 99% là lớp đa số, nhưng không bao giờ phát hiện được lớp thiểu số

> 💡 **Ví dụ:** Phát hiện gian lận tín dụng — 99% giao dịch bình thường, 1% gian lận. Mô hình luôn đoán "bình thường" → accuracy 99% nhưng recall gian lận = 0%.

### Giải Pháp

**Ở cấp độ Metrics (đánh giá):**
- Dùng **Precision, Recall, F1-score** thay vì accuracy
- **PR curve** (Precision-Recall curve)
- **ROC-AUC**

**Ở cấp độ Dữ liệu:**
| Phương pháp | Mô tả |
|---|---|
| **Oversampling** | Tăng số lượng lớp thiểu số (ví dụ: SMOTE tạo mẫu tổng hợp) |
| **Undersampling** | Giảm số lượng lớp đa số |
| **Kết hợp cả hai** | Oversample thiểu số + undersample đa số |

**Ở cấp độ Thuật toán:**
| Phương pháp | Mô tả |
|---|---|
| **Class weights** | Phạt nặng hơn khi sai trên lớp thiểu số trong loss function |
| **Focal Loss** | Tự động giảm trọng số cho mẫu dễ, tập trung vào mẫu khó |
| **Threshold tuning** | Điều chỉnh ngưỡng quyết định (thay vì mặc định 0.5) |

---

## 19. Data Augmentation

> 📌 **Ảnh:** Slide trang 26 — `02-Datasets.pdf#page=26`
> Hộp "Images" (crop, flip, rotate, color jitter, cutout, MixUp — applied on the fly); hộp "Text/sequences" (back-translation, synonym replacement, random insert/delete); hộp "Rule" (màu đỏ): augment only training set.

**Ý tưởng:** Tăng kích thước hiệu quả của tập dữ liệu bằng cách tạo **biến thể** của mẫu huấn luyện *mà không thay đổi nhãn*.

### Augmentation cho Ảnh

| Kỹ thuật | Mô tả |
|---|---|
| **Geometric** | Crop (cắt), flip (lật), rotate (xoay), scale |
| **Photometric** | Color jitter, brightness/contrast thay đổi |
| **Erasing** | Cutout — che một vùng ngẫu nhiên |
| **Mixing** | MixUp (pha trộn 2 ảnh), CutMix (ghép vùng từ ảnh khác) |

Áp dụng **on the fly** (ngẫu nhiên mỗi epoch) → tối đa hóa sự đa dạng.

### Augmentation cho Văn Bản / Chuỗi

| Kỹ thuật | Mô tả | Lưu ý |
|---|---|---|
| **Back-translation** | Dịch sang ngôn ngữ trung gian rồi dịch lại | Cẩn thận thay đổi nghĩa |
| **Synonym replacement** | Thay từ bằng từ đồng nghĩa | Phải giữ nguyên label |
| **Random insert/delete** | Thêm/xóa từ ngẫu nhiên | Cẩn thận với nhãn |

### ⚠️ Quy Tắc Quan Trọng

> **Augmentation chỉ áp dụng cho TRAINING SET. Validation và test phải giữ nguyên bản gốc để đánh giá công bằng.**

---

## 20. Benchmarks và Nguồn Dữ Liệu

> 📌 **Ảnh trang 28 (Common Benchmarks):** `02-Datasets.pdf#page=28`
> 4 bullets: Vision (MNIST/CIFAR/ImageNet/COCO), Tabular (UCI/Kaggle), Text (GLUE/SQuAD/WikiText), Recommendation/Graphs (MovieLens/OGB).

> 📌 **Ảnh trang 29 (Where to Get Data):** `02-Datasets.pdf#page=29`
> 4 bullets: Public repos, APIs/crawling, Synthetic data, document source/license.

### Tại Sao Cần Benchmark?

Benchmark là các **bộ dữ liệu chuẩn** với splits và metrics được cộng đồng đồng thuận → cho phép so sánh kết quả giữa các paper và implementation một cách **công bằng**.

### Các Benchmark Phổ Biến

| Lĩnh vực | Benchmark | Mô tả |
|---|---|---|
| **Vision** | MNIST | Chữ số viết tay 0–9, 70k ảnh 28×28 |
| | CIFAR-10/100 | 60k ảnh 32×32, 10/100 lớp |
| | ImageNet | 1.2M ảnh, 1000 lớp — "Olympics" của CV |
| | COCO | Object detection & segmentation |
| **Tabular** | UCI ML Repository | Nhiều dataset cổ điển |
| | Kaggle | Competition datasets |
| **Text** | GLUE / SuperGLUE | 9 tasks NLU |
| | SQuAD | Question Answering |
| | WikiText | Language modeling |
| **Graphs** | MovieLens | Recommendation system |
| | OGB (Open Graph Benchmark) | Graph ML tasks |

### Nơi Lấy Dữ Liệu

| Nguồn | Ghi chú |
|---|---|
| **Public repos** | UCI, Kaggle, Hugging Face Datasets, torchvision/torchtext |
| **APIs / Web crawling** | Tôn trọng Terms of Use và bản quyền |
| **Synthetic data** | Khi dữ liệu thực tế khan hiếm hoặc nhạy cảm |

> 💡 **Nguyên tắc tái lập (Reproducibility):** Luôn ghi lại rõ nguồn dữ liệu, giấy phép, và quy trình tiền xử lý.

---

## 21. Dataset và DataLoader trong PyTorch

> 📌 **Ảnh trang 31 (Dataset vs DataLoader):** `02-Datasets.pdf#page=31`
> Hộp "Dataset" (implements __len__ và __getitem__, stores/loads samples, apply transforms); hộp "DataLoader" (mini-batching, shuffling, parallel loading num_workers, pin_memory).

> 📌 **Ảnh trang 32 (Key Parameters):** `02-Datasets.pdf#page=32`
> 4 bullets: batch_size (speed vs memory/gradient noise), num_workers (parallel, 0=main), pin_memory (CPU→GPU speedup), drop_last (consistent epoch updates).

### Dataset vs DataLoader — Hai Vai Trò Phân Biệt

> 💡 **Ẩn dụ:** `Dataset` là thư viện sách. `DataLoader` là thủ thư — quyết định bao nhiêu sách lấy ra mỗi lần, theo thứ tự nào, và dùng bao nhiêu nhân viên phục vụ.

#### `torch.utils.data.Dataset` — Quản lý *dữ liệu*

- Implements `__len__()`: trả về số lượng mẫu
- Implements `__getitem__(idx)`: trả về mẫu thứ `idx` (và nhãn)
- Có thể áp dụng **transforms** (augmentation, normalization) khi trả về mẫu

```python
class MyDataset(Dataset):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
```

#### `DataLoader` — Quản lý *luồng dữ liệu*

- **Mini-batching**: Gộp nhiều mẫu thành một batch tensor
- **Shuffling**: Xáo trộn sau mỗi epoch
- **Parallel loading**: `num_workers` luồng song song — tải dữ liệu không chặn GPU

```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,        # chỉ train
    num_workers=4,
    pin_memory=True,     # khi dùng CUDA
    drop_last=False
)
```

### Các Tham Số Quan Trọng

| Tham số | Ý nghĩa | Ghi chú thực tiễn |
|---|---|---|
| `batch_size` | Số mẫu trong một batch | Trade-off: lớn → nhanh nhưng tốn RAM; nhỏ → nhiễu nhiều hơn |
| `num_workers` | Số luồng song song | `0` = main process; thường set 4–8 |
| `pin_memory` | Ghim data vào RAM không phân trang | Tăng tốc CPU→GPU khi dùng CUDA |
| `drop_last` | Bỏ batch cuối nếu không đủ `batch_size` | Đảm bảo số updates mỗi epoch nhất quán |
| `shuffle` | Xáo trộn mỗi epoch | `True` cho train, `False` cho val/test |

---

## Tóm Tắt Bài 02

| Khái niệm | Nội dung cốt lõi |
|---|---|
| **Dataset** | $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N$; feature = thuộc tính; label = ground-truth |
| **DL Shapes** | Image $(B,C,H,W)$; Text/TS $(B,T)$ hoặc $(B,T,F)$ + padding & mask |
| **Data Quality** | Missing values, noise, label noise, duplicates, bias — xử lý trước khi train |
| **Preprocessing** | Fit normalization/encoding **chỉ trên train set**; tránh data leakage |
| **Train/Val/Test** | Stratified split cho classification; temporal split cho time-series |
| **Cross-validation** | K-fold khi data ít; test set vẫn tách biệt |
| **Class imbalance** | Metrics (F1, AUC); oversampling/undersampling; class weights/focal loss |
| **Augmentation** | **Chỉ trên training set**; val/test giữ nguyên bản gốc |
| **Benchmarks** | MNIST, CIFAR, ImageNet, COCO, GLUE, SQuAD — dùng standard splits |
| **Dataset** | Implements `__getitem__`; áp dụng transforms |
| **DataLoader** | Handles batching, shuffling, parallel workers |
