# Bài 13: Phát hiện Đối tượng trong Ảnh

> **Nguồn:** `Detection.VNE.pdf` (85 trang) — YOLOv8, YOLO11, DETR và các mô hình hiện đại  
> **Phong cách:** Ghi chú giảng đường — giải thích từng khái niệm, thêm lý do & ví dụ số cụ thể.

---

## Mục lục

1. [Giới thiệu — Object Detection là gì?](#1-giới-thiệu)
2. [Taxonomy — Bức tranh tổng quan](#2-taxonomy)
3. [Tổ chức dữ liệu và nhãn YOLO](#3-tổ-chức-dữ-liệu-và-nhãn)
4. [Kiến trúc YOLO — Backbone, Neck, Head](#4-kiến-trúc-yolo)
5. [Chi tiết Đầu phát hiện (Detection Head)](#5-chi-tiết-đầu-phát-hiện)
6. [TaskAlignedAssigner — Gán nhãn One-to-Many](#6-taskalignedassigner)
7. [Loss Function trong YOLO](#7-loss-function)
8. [YOLOv8 vs YOLO11](#8-yolov8-vs-yolo11)
9. [Bước Suy diễn (Inference) và NMS](#9-bước-suy-diễn-và-nms)
10. [Đánh giá Kết quả — IoU, AP, mAP](#10-đánh-giá-kết-quả)
11. [Mô hình Nâng cao — YOLOv26, DETR, Hungarian Matching](#11-mô-hình-nâng-cao)
12. [Tóm tắt](#12-tóm-tắt)

---

## 1. Giới thiệu

### 1.1 Object Detection là gì?

**Phân loại (Classification)** trả lời: "Trong ảnh này có gì?" → một nhãn cho toàn ảnh.

**Phát hiện (Detection)** trả lời hai câu hỏi cùng lúc: **"Có gì?"** và **"Ở đâu?"** → tập các hộp giới hạn kèm nhãn.

**Định nghĩa hình thức:**

$$\hat{Y} = \{(\hat{b}_i,\; \hat{c}_i,\; \hat{s}_i)\}_{i=1}^{N}$$

- $\hat{b}_i$: **bounding box** — hình chữ nhật bao quanh vật thể.
- $\hat{c}_i$: **class** — nhãn lớp (xe, người, chó, ...).
- $\hat{s}_i$: **confidence score** — độ tin cậy của dự đoán.
- $N$: **biến số** — không biết trước khi inference (khác với classification luôn có output cố định).

**Ứng dụng thực tế:**
- CCTV: cảnh báo người/phương tiện xâm nhập.
- Bán lẻ: phân tích kệ hàng, tự động hóa thanh toán.
- Giao thông: phát hiện vi phạm, ước tính lưu lượng xe.
- Robotics: nhận biết vật thể để gắp và tránh vật cản.
- Y tế: định vị tổn thương, dụng cụ phẫu thuật.

### 1.2 So sánh các tác vụ Computer Vision

| Task | Output | Chi tiết không gian | Metric |
|------|--------|---------------------|--------|
| Classification | Một nhãn/ảnh | Không có | Accuracy |
| **Detection** | **Boxes + nhãn** | **Mức đối tượng** | **mAP** |
| Segmentation | Nhãn mỗi pixel | Mức pixel | mIoU / Dice |

### 1.3 Bounding Box Formats

Có hai cách biểu diễn bounding box phổ biến:

```
xyxy = (x1, y1, x2, y2)     # góc trên-trái và góc dưới-phải (pixel)
cxcywh = (cx, cy, w, h)     # tâm box và kích thước (thường chuẩn hóa về [0,1])
```

**Khi nào dùng cái nào?**
- **xyxy:** thuận tiện cho tính IoU và clipping (cắt box vào biên ảnh).
- **cxcywh chuẩn hóa:** định dạng trong file nhãn YOLO — dễ scale bất kể độ phân giải ảnh.

**Chuyển đổi:**
```python
# xyxy → cxcywh
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
w  = x2 - x1
h  = y2 - y1

# cxcywh → xyxy
x1 = cx - w/2
y1 = cy - h/2
x2 = cx + w/2
y2 = cy + h/2
```

---

## 2. Taxonomy

### 2.1 Bức tranh tổng quan Object Detection

```
Object Detection
│
├─ Supervised
│   ├─ Two-stage (R-CNN family)
│   │   → Faster R-CNN (Detectron2)
│   │   → Region Proposal → Classify each proposal
│   │
│   ├─ One-stage (YOLO/SSD/RetinaNet)
│   │   → YOLOv8, YOLO11 (Ultralytics)
│   │   → Dự đoán trực tiếp trên grid: không cần region proposal
│   │
│   └─ End-to-end NMS-free (DETR)
│       → DETR, DINO, RT-DETR
│       → Không cần NMS — học matching trực tiếp
│
├─ Semi/Weakly supervised
│   → Pseudo-labeling, Teacher-Student
│
└─ Open-vocabulary / Zero-shot
    → CLIP-aligned, GLIP, Grounding DINO
    → Phát hiện lớp chưa thấy khi train
```

**Trọng tâm bài học: One-stage (YOLO) và End-to-end (DETR).**

---

## 3. Tổ chức Dữ liệu và Nhãn

### 3.1 Cấu trúc thư mục YOLO (Ultralytics)

```
dataset/
├── data.yaml           # khai báo đường dẫn và danh sách lớp
├── train/
│   ├── images/         # ảnh training
│   └── labels/         # file .txt tương ứng (cùng tên)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**`data.yaml`:**
```yaml
path: /path/to/dataset
train: train/images
val:   val/images
test:  test/images

nc: 3               # số lớp
names: ['xe', 'nguoi', 'xe_dap']
```

**Quy tắc quan trọng:**
- Mỗi ảnh có **đúng 1 file nhãn** cùng tên (001.jpg ↔ 001.txt).
- Tách riêng train/val/test để **tránh rò rỉ dữ liệu (data leakage)**.
- Tỉ lệ tham khảo: train 70-80%, val 10-15%, test 10-15%.

### 3.2 Định dạng nhãn YOLO (file .txt)

Mỗi dòng = một object:

```
class_id  cx  cy  w  h
0         0.5 0.4 0.3 0.2    # xe ở giữa ảnh, chiếm 30% chiều rộng, 20% chiều cao
1         0.1 0.2 0.05 0.1   # người ở góc trên-trái
```

Tất cả tọa độ được **chuẩn hóa về [0,1]** so với kích thước ảnh.

### 3.3 Tensor nhãn khi training (batch)

Khi batch nhiều ảnh, số object khác nhau mỗi ảnh → cần padding:

- $N_{\max}$: số object tối đa trong batch → pad các ảnh ít object hơn.
- `loc_gt` $\in \mathbb{R}^{B \times N_{\max} \times 4}$: tọa độ GT (cxcywh chuẩn hóa).
- `cls_gt` $\in \mathbb{Z}^{B \times N_{\max}}$: chỉ số lớp 0...$N_c-1$.
- `mask` $\in \{0,1\}^{B \times N_{\max}}$: 1 = slot có GT hợp lệ, 0 = padding → bỏ qua khi tính loss.

---

## 4. Kiến trúc YOLO

### 4.1 Sơ đồ tổng quan

```
Image (B, 3, 640, 640)
    │
    ▼
[Backbone]  →  P3(B,C3,80,80) / P4(B,C4,40,40) / P5(B,C5,20,20)
    │
    ▼
[Neck FPN/PAN]  →  P3*(B,C3,80,80) / P4*(B,C4,40,40) / P5*(B,C5,20,20)
    │
    ▼
[Head]  →  loc(B,Na,4Nr) + cls(B,Na,Nc)
    │
    ▼
[NMS]  →  Outputs (B, N_det, 6)   # 6 = [x1,y1,x2,y2, conf, class]
```

**Với ảnh 640×640:**
- $N_a = 80^2 + 40^2 + 20^2 = 6400 + 1600 + 400 = \mathbf{8400}$ điểm anchor tổng cộng.

### 4.2 Backbone — Trích xuất đặc trưng đa mức

**Vai trò:** Đọc ảnh và tạo ra các **feature map ở nhiều độ phân giải** (multi-scale features).

**Tại sao cần đa mức?**
- Feature map **mịn (fine)** — stride nhỏ, ô nhỏ → bắt vật thể **nhỏ** (nhiều chi tiết).
- Feature map **thô (coarse)** — stride lớn, ô lớn → bắt vật thể **lớn** (ngữ cảnh rộng).

| Mức | Stride | Feature map (640×640) | Nhạy với |
|-----|--------|----------------------|----------|
| **P3** | 8 | 80×80 | Vật thể nhỏ |
| **P4** | 16 | 40×40 | Vật thể trung |
| **P5** | 32 | 20×20 | Vật thể lớn |

**Ví dụ:** Con người cao 30px trên ảnh → rơi vào P3 (stride 8). Xe tải cao 200px → rơi vào P5 (stride 32).

**Các khối trong backbone:**

| Khối | Vai trò |
|------|---------|
| **Conv** | Giảm không gian, tăng kênh đặc trưng |
| **C2f (YOLOv8)** | CSP-style — tăng biểu diễn, giữ FLOPs hợp lý |
| **C3k2 (YOLO11)** | Thay C2f, hiệu quả hơn |
| **SPPF** | Spatial Pyramid Pooling Fast — gom ngữ cảnh ở tầng sâu |
| **C2PSA (YOLO11)** | Thêm self-attention sau SPPF — tăng chọn lọc đặc trưng |

### 4.3 Neck — Hợp nhất đa tỷ lệ (FPN/PAN)

**Vấn đề:** P3 giàu chi tiết nhưng thiếu ngữ cảnh. P5 giàu ngữ cảnh nhưng thiếu chi tiết. Cần kết hợp cả hai.

**Feature Pyramid Network (FPN):** truyền thông tin ngữ nghĩa từ tầng sâu **xuống** tầng nông.

$$P5 \xrightarrow{\text{upsample}} \text{merge with } P4 \xrightarrow{\text{upsample}} \text{merge with } P3$$

**Path Aggregation Network (PAN):** truyền thông tin vị trí/chi tiết từ tầng nông **ngược lên** tầng sâu.

$$P3^* \xrightarrow{\text{downsample}} \text{merge with } P4^* \xrightarrow{\text{downsample}} \text{merge with } P5^*$$

**Kết quả:** $P3^*, P4^*, P5^*$ — mỗi mức vừa có chi tiết cục bộ **vừa** có ngữ cảnh toàn cục.

📸 [Cần ảnh: FPN + PAN topology — slide page 17] — Tìm: "FPN PAN feature pyramid network"

### 4.4 Head — Dự đoán box và class

**Anchor-free design (YOLOv8/v11):** Thay vì dùng anchor box kích thước định sẵn, mỗi **điểm tham chiếu** (ô trên feature map) dự đoán trực tiếp khoảng cách đến 4 cạnh box.

**Tạo điểm tham chiếu (anchor points):**

Với feature map $m$ có stride $s_m$ và kích thước $H_m \times W_m$:

$$x_c = (j + 0.5) \cdot s_m, \quad y_c = (i + 0.5) \cdot s_m$$

Ô $(i, j)$ trên feature map → tâm tham chiếu $(x_c, y_c)$ trong không gian ảnh gốc.

**Tại mỗi điểm tham chiếu, head dự đoán:**

$$\text{box}: 4 \times N_r \text{ logits (DFL)}$$
$$\text{class}: N_c \text{ logits (sigmoid)}$$

---

## 5. Chi tiết Đầu phát hiện

### 5.1 Kiến trúc Detection Head (từng mức P3/P4/P5)

```
P3, P4, P5
│
├── [Feature Refiner] (per scale, shared weights)
│       Conv2d → BN → SiLU
│       Conv2d → BN → SiLU
│
├── [Conv2d loc] (out_channels = 4×Nr, k=1)  → loc3/loc4/loc5
│
└── [Conv2d cls] (out_channels = Nc, k=1)    → cls3/cls4/cls5

Aggregator: concat theo Na dimension
→ loc_pred: (B, Na, 4×Nr)
→ cls_pred: (B, Na, Nc)
```

### 5.2 Distribution Focal Loss (DFL) — Dự đoán phân phối khoảng cách

**Ý tưởng:** Thay vì dự đoán trực tiếp $l, t, r, b$ (left, top, right, bottom từ anchor point), YOLO hiện đại dự đoán **phân phối xác suất** trên $N_r$ mức rời rạc (bin) cho mỗi cạnh.

**Ví dụ:** $N_r = 16$ → bin $\{0, 1, 2, ..., 15\}$ pixel từ anchor.

**Độ lệch thực sự = kỳ vọng:**

$$\hat{d} = \sum_{k=0}^{N_r-1} k \cdot p_k$$

**Tại sao phân phối tốt hơn giá trị đơn?**

Nếu GT box edge nằm giữa hai pixel, dự đoán một con số duy nhất gây loss lớn. Phân phối có thể đặt xác suất vào hai bin kề nhau → loss nhỏ hơn, gradient ổn định hơn.

### 5.3 Từ chỉ số k → vị trí trên ảnh

$N_a = 8400$ điểm được đánh số từ 0 đến 8399, ghép theo thứ tự P3 → P4 → P5:

```
k ∈ [0, n3-1]           → mức P3 (n3 = 80×80 = 6400)
k ∈ [n3, n3+n4-1]       → mức P4 (n4 = 40×40 = 1600)
k ∈ [n3+n4, Na-1]       → mức P5 (n5 = 20×20 = 400)
```

Từ chỉ số local $t$ trên mức $m$ (kích thước $H_m \times W_m$):

$$i = \lfloor t / W_m \rfloor, \quad j = t \bmod W_m$$

$$x_c = (j + 0.5) s_m, \quad y_c = (i + 0.5) s_m$$

### 5.4 Class Prediction

- Với mỗi anchor point, dự đoán $N_c$ logit → $N_c$ xác suất độc lập qua **sigmoid** (multi-label có thể).
- **Training:** sigmoid + BCE loss.
- **Inference:** class = argmax; lọc theo ngưỡng conf để loại box yếu.

---

## 6. TaskAlignedAssigner

### 6.1 Vấn đề Gán nhãn (Label Assignment)

Mô hình sinh $N_a = 8400$ dự đoán. Mỗi ảnh có $K$ GT objects. Cần xác định: **dự đoán nào là "positive" (học theo GT), dự đoán nào là "negative" (học là nền)?**

**One-to-One (DETR):** Mỗi GT ghép **đúng một** prediction → Hungarian algorithm.

**One-to-Many (YOLO dense):** Một GT có thể ghép **nhiều** predictions cùng lúc → nhiều gradient tích cực hơn → học nhanh hơn.

Ký hiệu: $m_{ik} \in \{0,1\}$: dự đoán $i$ được gán cho GT $k$.

$$\sum_{i=1}^{N_a} m_{ik} \geq 1 \quad \text{(one-to-many)}$$

### 6.2 Điểm Căn chỉnh (Alignment Score)

TaskAlignedAssigner (từ TOOD/Ultralytics) chọn positive dựa trên điểm kết hợp **classification + localization**:

$$t_{ik} = \left(\sigma(z_{i,c_k})\right)^\alpha \cdot \left(\text{IoU}(b_i, b_k^{\text{gt}})\right)^\beta, \quad \alpha, \beta > 0$$

- $\sigma(z_{i,c_k})$: xác suất sigmoid của class $c_k$ tại dự đoán $i$.
- $\text{IoU}(b_i, b_k^{\text{gt}})$: IoU giữa box dự đoán $i$ và GT $k$.
- $t_{ik}$ cao khi **cả class đúng lẫn box gần** GT.

**Quy trình:**
1. Với mỗi GT $k$, tiền lọc ứng viên theo IoU (chỉ xét những điểm gần GT về mặt hình học).
2. Tính $t_{ik}$ cho ứng viên.
3. Giữ **top-r** chỉ số $i$ có $t_{ik}$ cao nhất → các cặp $(i, k)$ đó có $m_{ik} = 1$.

---

## 7. Loss Function

### 7.1 Tổng quan

$$\mathcal{L} = w_{\text{box}} \mathcal{L}_{\text{box}} + w_{\text{cls}} \mathcal{L}_{\text{cls}} + w_{\text{dfl}} \mathcal{L}_{\text{dfl}}$$

| Thành phần | Tính trên | Vai trò |
|------------|-----------|---------|
| $\mathcal{L}_{\text{cls}}$ | Tất cả $N_a$ vị trí | Phân loại class (ô dương: one-hot, ô âm: zero) |
| $\mathcal{L}_{\text{box}}$ | Chỉ ô dương ($m_{ik}=1$) | Hồi quy box geometry |
| $\mathcal{L}_{\text{dfl}}$ | Chỉ ô dương ($m_{ik}=1$) | Phân phối khoảng cách cạnh |

### 7.2 Classification Loss (BCE)

Với logit $z_{i,c}$, xác suất $\hat{p}_{i,c} = \sigma(z_{i,c})$, nhãn $y_{i,c} \in \{0,1\}$:

$$\mathcal{L}_{\text{cls}} = -\frac{1}{N_a} \sum_{i=1}^{N_a} \sum_{c=1}^{N_c} \left[ y_{i,c} \log \hat{p}_{i,c} + (1 - y_{i,c}) \log(1 - \hat{p}_{i,c}) \right]$$

### 7.3 Box Regression Loss (CIoU)

**IoU (Intersection over Union):**

$$\text{IoU} = \frac{|\hat{B} \cap B^{\text{gt}}|}{|\hat{B} \cup B^{\text{gt}}|}$$

**CIoU:** Mở rộng IoU để penalize thêm khoảng cách trung tâm và tỉ lệ aspect ratio:

$$\mathcal{L}_{\text{box}} = 1 - \text{CIoU}(\hat{b}, b^{\text{gt}})$$

Trung bình trên các mẫu dương.

### 7.4 Distribution Focal Loss (DFL)

Với cạnh thật $y$ (offset liên tục từ anchor point):

- $y^- = \lfloor y \rfloor$, $y^+ = \lceil y \rceil$ (hai bin cận nhau)
- Trọng số nhãn mềm: $w^- = y^+ - y$, $w^+ = y - y^-$ (thỏa $w^- + w^+ = 1$)

**Loss một cạnh:**

$$\mathcal{L}_{\text{dfl}}^{(cnh)} = -\left(w^- \log \hat{p}_{y^-} + w^+ \log \hat{p}_{y^+}\right)$$

**Ví dụ cụ thể (slide page 32-33):** $y = 2.45$, $y^- = 2$, $y^+ = 3$:

$$w^- = 3 - 2.45 = 0.55, \quad w^+ = 2.45 - 2 = 0.45$$

$$\mathcal{L}_{\text{dfl}} = -(0.55 \log \hat{p}_2 + 0.45 \log \hat{p}_3)$$

Khi $y$ nguyên (ví dụ $y=5$): $\mathcal{L}_{\text{dfl}} = -\log \hat{p}_5$ (CE thông thường).

Cộng loss của cả 4 cạnh $l, t, r, b$ cho các ô dương.

---

## 8. YOLOv8 vs YOLO11

### 8.1 Bảng so sánh theo 3 module

| Module | YOLOv8 | YOLO11 |
|--------|--------|--------|
| **Backbone** | Conv + **C2f** + SPPF | Conv + **C3k2** + SPPF + **C2PSA** |
| **Neck** | FPN/PAN với **C2f** | FPN/PAN với **C3k2** |
| **Head — cấu trúc** | Decoupled: box branch + class branch | Giống YOLOv8 về nguyên lý |
| **Head — impl** | Conv-BN-SiLU blocks | **DWConv + PWConv** (Depthwise + Pointwise) |
| **Box regression** | DFL: $4 \times N_r$ kênh | DFL: $4 \times N_r$ kênh |
| **Classification** | $N_c$ logits, sigmoid + BCE | $N_c$ logits, sigmoid + BCE |

### 8.2 Khác biệt cốt lõi

**C2f (YOLOv8):** CSP-based, cân bằng giữa biểu diễn và tốc độ.

**C3k2 (YOLO11):** Thay C2f, hướng đến biểu diễn hiệu quả hơn.

**C2PSA (YOLO11):** Thêm **cross-attention** sau SPPF → mô hình "nhìn" toàn cục tốt hơn ở backbone.

**DWConv + PWConv (YOLO11 head):**
- **DWConv (Depthwise):** mỗi channel học spatial filter riêng → ít param hơn.
- **PWConv (Pointwise = 1×1 conv):** trộn thông tin giữa channels.
- Kết quả: ít param, ít FLOPs → **nhanh hơn trên edge GPU/mobile**.

> **Thông điệp:** YOLOv8 và YOLO11 có cùng triết lý detection head (anchor-free, DFL, sigmoid/BCE). YOLO11 cải tiến cách **trích xuất và tinh luyện feature** trước khi đưa vào head.

### 8.3 Cách đọc file YAML Ultralytics đúng

Trong YAML của Ultralytics, phần `head:` thực ra bao gồm cả **Neck (FPN/PAN)** và **Detection Head** (không phải chỉ head đơn thuần):

```
YAML backbone: → Backbone về mặt học thuật
YAML head:     → Neck + Detection Head về mặt học thuật
```

Khi đọc code YOLO:
- `backbone.py` / `block.py` → hiểu các khối trích xuất
- `head.py` → hiểu `Detect` head
- `loss.py` / `tal.py` → hiểu cách gán nhãn và tính loss

---

## 9. Bước Suy diễn và NMS

### 9.1 Quy trình Suy diễn (Inference Pipeline)

```
Ảnh gốc
    ↓
[Tiền xử lý] resize/letterbox về imgsz=640, chuẩn hóa pixel [0,1]
    ↓
[Forward Pass] Backbone → Neck → Head
    ↓ loc(B,Na,4Nr) + cls(B,Na,Nc)
[Giải mã box] DFL → 4 offset → box xyxy trên ảnh đầu vào
    ↓
[Lọc conf] bỏ box có confidence < ngưỡng (mặc định 0.25)
    ↓
[NMS per class] loại box trùng lặp (IoU > τ, mặc định 0.45)
    ↓
[Ánh xạ tọa độ] letterbox inverse → tọa độ trên ảnh gốc
    ↓
Outputs: (N_det, 6) với 6 = [x1,y1,x2,y2, conf, class_id]
```

**Letterbox:** resize ảnh giữ tỉ lệ (không distort), pad phần còn lại bằng màu xám. Khi đảo ngược: trừ padding $(dx, dy)$, chia tỉ lệ $r$.

### 9.2 Non-Maximum Suppression (NMS)

**Vấn đề:** Cùng một vật thể, nhiều anchor point gần đó cùng phát hiện → nhiều box chồng lấn cho cùng một vật.

**NMS giải quyết:** Chỉ giữ box tốt nhất, loại bỏ các box "trùng lặp".

**Thuật toán NMS (per class):**

```
Input: danh sách boxes với confidence scores
1. Sort boxes theo confidence (giảm dần)
2. Lấy box đầu (cao điểm nhất) → đưa vào kết quả
3. Với mỗi box còn lại:
   - Tính IoU với box vừa chọn
   - Nếu IoU > τ → loại bỏ (box này là duplicate)
   - Nếu IoU ≤ τ → giữ lại cho vòng tiếp theo
4. Lặp bước 2-3 với danh sách còn lại
5. Dừng khi hết boxes
```

**Ví dụ cụ thể (từ slide, τ = 0.5):**

Có 5 box: A1(0.95), A2(0.82), A3(0.74), B1(0.90), B2(0.79).

```
Bước 1: Sort → [A1=0.95, B1=0.90, A2=0.82, B2=0.79, A3=0.74]
Bước 2: Chọn A1
  IoU(A1,B1) = 0.15 < 0.5 → giữ B1
  IoU(A1,A2) = 0.68 > 0.5 → loại A2  ✗
  IoU(A1,B2) = 0.10 < 0.5 → giữ B2
  IoU(A1,A3) = 0.73 > 0.5 → loại A3  ✗
Bước 3: Chọn B1
  IoU(B1,B2) = 0.64 > 0.5 → loại B2  ✗

Kết quả: {A1, B1}   ← 2 boxes cho 2 vật thể khác nhau
```

**Tham số NMS:**
- `iou_threshold` ($\tau$): cao → giữ nhiều box → precision thấp hơn. Thấp → loại nhiều → recall thấp hơn.
- `conf_threshold`: lọc sơ trước NMS để giảm tải.

**Lưu ý:** NMS chỉ dùng ở **inference**, không phải trong loss training.

---

## 10. Đánh giá Kết quả

### 10.1 IoU — Intersection over Union

$$\text{IoU} = \frac{|\hat{B} \cap B^{\text{gt}}|}{|\hat{B} \cup B^{\text{gt}}|}$$

- IoU ∈ [0, 1]. IoU = 1: box khớp hoàn hảo. IoU = 0: không chồng lấn.
- Dùng làm ngưỡng khớp: nếu cùng class và IoU ≥ $\tau$ → xem là **match**.

📸 [Cần ảnh: minh họa IoU = area(intersection)/area(union) — slide page 55]

### 10.2 TP / FP / FN

| Tên | Nghĩa |
|-----|-------|
| **TP** (True Positive) | Đúng class + IoU ≥ $\tau$ — phát hiện đúng |
| **FP** (False Positive) | Sai class, box sai, hoặc box trùng lặp — báo nhầm |
| **FN** (False Negative) | Vật thật nhưng model bỏ sót |

> **Good detector = many TP + few FP + few FN**

**Lưu ý:** Nếu cùng một GT có 3 box dự đoán khớp (IoU đủ, đúng class), chỉ box đầu tiên (cao điểm nhất) là TP; 2 box còn lại là FP (duplicate detection).

### 10.3 Precision và Recall

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad \text{(trong số phát hiện, bao nhiêu đúng?)}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad \text{(trong số GT, bao nhiêu được tìm thấy?)}$$

**Đánh đổi Precision-Recall:**
- Hạ ngưỡng confidence → phát hiện nhiều hơn → Recall ↑ nhưng FP tăng → Precision ↓.
- Tăng ngưỡng → Precision ↑ nhưng bỏ sót nhiều → Recall ↓.

### 10.4 PR Curve và Average Precision (AP)

**PR Curve:** Quét nhiều giá trị ngưỡng confidence, mỗi giá trị cho một cặp (Recall, Precision) → nối lại thành đường cong.

```
Precision
1.0 |●●●
    |   ●●
    |     ●●
0.5 |       ●
    |        ●
  0 +----------→ Recall
    0    0.5    1.0
```

Curve **càng cao và càng rộng** → detector càng tốt.

**Average Precision (AP):** Diện tích dưới PR Curve (AUC):

$$\text{AP} = \int_0^1 P(R)\, dR$$

Tính cho **một lớp** tại một ngưỡng IoU cụ thể:
- **AP50:** IoU threshold = 0.50 (dễ hơn)
- **AP75:** IoU threshold = 0.75 (khó hơn, box phải chính xác hơn)

### 10.5 Mean Average Precision (mAP)

$$\text{mAP} = \frac{1}{N_c} \sum_{c=1}^{N_c} \text{AP}_c$$

Trung bình AP trên tất cả lớp → **metric chuẩn** cho object detection.

**COCO metrics:**
- `mAP@50`: IoU ≥ 0.50 (PASCAL VOC style)
- `mAP@50:95`: trung bình AP tại các ngưỡng IoU 0.50, 0.55, ..., 0.95 (COCO style — khó hơn nhiều)

### 10.6 Các lỗi thường gặp khi đánh giá

| Lỗi | Hậu quả |
|-----|---------|
| Train và val dùng resize khác nhau | mAP không phản ánh thực tế |
| Sai mapping class-id | TP trở thành FP |
| Rò rỉ dữ liệu train/val | mAP ảo cao |
| NMS config khác baseline | Không so sánh được với paper |
| Label sai hoặc box quá lỏng | GT không chuẩn → đánh giá sai |

> **Nguyên tắc:** `Bad Evaluation ⇒ Bad Conclusion`

### 10.7 Định dạng dữ liệu phổ biến

| Format | Cấu trúc | Dùng bởi |
|--------|----------|----------|
| **YOLO** | `.txt` per image: `class cx cy w h` | Ultralytics |
| **COCO** | JSON annotations tập trung | COCO benchmark |
| **VOC** | XML per image | PASCAL VOC |

→ Cần chuyển đổi đúng format trước khi train/eval!

### 10.8 Augmentation và Pipeline

**Data augmentation thường dùng cho detection:**
- **Flip** (horizontal): phổ biến, không đổi nhãn.
- **HSV jitter**: thay đổi màu sắc.
- **Mosaic**: ghép 4 ảnh thành 1 → học context nhỏ, tăng diversity.
- **MixUp**: blend 2 ảnh + 2 nhãn theo tỉ lệ.

**Throughput issue:** Loader chậm làm GPU ngồi chờ dữ liệu → GPU utilization thấp.
- Tăng `workers` (số worker DataLoader).
- Dùng `cache` (lưu ảnh đã xử lý vào RAM).
- Dùng fast storage (SSD NVMe thay HDD).

---

## 11. Mô hình Nâng cao

### 11.1 YOLOv26 — NMS-free Dual Heads

**Vấn đề của YOLO truyền thống:** One-to-many assignment + NMS → NMS là heuristic, không học được, có thể loại nhầm.

**YOLOv26 giải quyết:** Thêm **hai detection head** song song:

```
Neck (P3*, P4*, P5*)
      │
      ├─ [One-to-Many Head] ← YOLO assignment truyền thống
      │   Dense supervision, recall tốt
      │   (dùng khi training)
      │
      └─ [One-to-One Head] ← Hungarian-style assignment
          Mỗi GT → đúng một prediction
          Output sạch, ít duplicate
          (dùng khi inference → không cần NMS)
```

**Cùng kiến trúc mạng, khác mục tiêu huấn luyện:**
- Training: cả hai head cùng đóng góp vào loss.
- Inference: chỉ dùng one-to-one head → **không cần NMS**.

**Kết quả:** High Recall (từ one-to-many) + Clean Prediction (từ one-to-one).

### 11.2 DETR — Detection Transformer

#### Ý tưởng cốt lõi

Thay vì duyệt 8400 anchor points như YOLO, DETR dùng **$N_q$ object queries** (thường $N_q = 100$) học được. Mỗi query "hỏi" ảnh về một vật thể → dự đoán class + box hoặc "no-object".

**Không cần anchor, không cần NMS** → end-to-end hoàn toàn.

#### Kiến trúc DETR (theo tensor)

**Bước 1: Trích xuất đặc trưng bằng CNN:**

$$I = \text{CNN}(\text{Images}), \quad \text{Images}: (B, 3, H_i, W_i) \to I: (B, C, H, W)$$

**Bước 2: Flatten và thêm positional encoding:**

$$X = \text{Flatten}(I) + \text{PE}, \quad (B, HW, C)$$

**Bước 3: Encoder (self-attention):**

$$F = \text{Encoder}(X, X, X), \quad (B, HW, C)$$

Gom ngữ cảnh toàn cục — mỗi vị trí "nhìn" mọi vị trí khác.

**Bước 4: Decoder với object queries:**

$$Q_f = \text{Decoder}(Q_0, F, F), \quad (B, N_q, C)$$

- $Q_0$: $N_q$ query vectors học được (`nn.Embedding(Nq, C)`).
- Decoder cross-attends vào $F$ → query "tìm kiếm" vật thể trong feature map.

**Bước 5: Prediction heads:**

$$\hat{p}: (B, N_q, N_c+1), \quad \hat{b}: (B, N_q, 4)$$

($N_c+1$ = $N_c$ lớp thật + 1 "no-object")

**Bước 6: Hungarian Matching + Loss:**

$$\sigma^* = \text{Hungarian}(Y, \hat{Y})$$

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_1 \|b - \hat{b}_{\sigma^*}\|_1 + \lambda_g \mathcal{L}_{\text{giou}}$$

📸 [Cần ảnh: DETR architecture với CNN + Encoder + Decoder + Queries — slide page 71]

#### Inference DETR (không cần NMS)

1. Tiền xử lý ảnh.
2. CNN + Encoder → feature $F$.
3. Decoder + $N_q$ queries → embeddings $Q_f$.
4. Heads → class logits + box chuẩn hóa.
5. Lọc: bỏ query có score thấp / no-object, giữ top-k.
6. Hậu xử lý tọa độ về ảnh gốc.

**Không cần NMS** vì training đã học one-to-one mapping qua Hungarian.

### 11.3 Hungarian Matching — Khớp tối ưu toàn cục

**Bài toán:** $N_{\text{gt}}$ GT objects, $N_q$ predictions. Tìm phép gán $\sigma$ (bijection) sao cho tổng chi phí nhỏ nhất.

**Ma trận chi phí:**

$$C_{ij} = C_{ij}^{\text{cls}} + \lambda_1 C_{ij}^{L1} + \lambda_g C_{ij}^{\text{giou}}$$

trong đó:
- $C_{ij}^{\text{cls}} = -\log \hat{p}_j(c_i)$: class cost (class đúng → cost nhỏ)
- $C_{ij}^{L1} = \|b_i - \hat{b}_j\|_1$: L1 box cost
- $C_{ij}^{\text{giou}} = \mathcal{L}_{\text{giou}}(b_i, \hat{b}_j)$: GIoU cost

**Ví dụ từ slide (3 GT, 4 queries):**

$$C = \begin{bmatrix} 0.32 & 0.81 & 0.44 & 0.96 \\ 0.77 & 0.25 & 0.58 & 0.69 \\ 0.66 & 0.49 & 0.21 & 0.73 \end{bmatrix}$$

Nếu chọn từng hàng độc lập: GT1→Q1(0.32), GT2→Q2(0.25), GT3→Q3(0.21) → may mắn không trùng.

**Ràng buộc one-to-one:** $\sigma(i) \neq \sigma(i')$ với $i \neq i'$ → mỗi GT ghép đúng một query.

**Giải pháp tối ưu:**

$$\sigma^* = \arg\min_{\sigma \in S_{N_q}} \sum_{i=1}^{N_{\text{gt}}} C_{i, \sigma(i)}$$

Thuật toán Hungarian chạy trong $O(N^3)$.

**Sau matching:**
- Cặp $(i, \sigma^*(i))$: tính full loss (cls + box + giou).
- Query không ghép GT ($Q_4$ trong ví dụ): học "no-object".

---

## 12. Tóm tắt

### 12.1 Các điểm cốt lõi

| Khái niệm | Công thức / Ý chính |
|-----------|---------------------|
| **Output detector** | $\hat{Y} = \{(\hat{b}_i, \hat{c}_i, \hat{s}_i)\}$ — bbox + class + conf |
| **$N_a$ anchors** | $80^2 + 40^2 + 20^2 = 8400$ với ảnh 640×640 |
| **DFL** | Phân phối xác suất trên $N_r$ bin; offset = $\sum k \cdot p_k$ |
| **TaskAlignedAssigner** | $t_{ik} = \sigma(z)^\alpha \cdot \text{IoU}^\beta$ — kết hợp cls + loc |
| **YOLO Loss** | $w_{\text{box}} L_{\text{CIoU}} + w_{\text{cls}} L_{\text{BCE}} + w_{\text{dfl}} L_{\text{DFL}}$ |
| **IoU** | $\|\hat{B} \cap B^{\text{gt}}\| / \|\hat{B} \cup B^{\text{gt}}\|$ |
| **AP** | Diện tích dưới PR Curve — cho một lớp |
| **mAP** | Trung bình AP trên $N_c$ lớp |
| **NMS** | Sort by conf → greedy: keep top, suppress IoU > $\tau$ |
| **DETR** | $N_q$ queries → Hungarian matching → end-to-end, NMS-free |
| **YOLOv26** | Dual heads: one-to-many (train) + one-to-one (infer, NMS-free) |

### 12.2 Lộ trình đọc code YOLO

```
data.yaml         → hiểu cấu trúc dataset
block.py          → hiểu C2f / C3k2 / C2PSA / SPPF
backbone.py       → hiểu cách xây feature pyramid
neck.py / head.py → hiểu FPN/PAN và Detect head
tal.py            → hiểu TaskAlignedAssigner
loss.py           → hiểu BCE + CIoU + DFL
val.py            → hiểu tính mAP
```

### 12.3 So sánh YOLO vs DETR

| | YOLO (one-stage) | DETR (end-to-end) |
|-|-----------------|-------------------|
| **Prediction** | $N_a$ dense anchors | $N_q$ sparse queries |
| **Assignment** | One-to-many (TaskAligned) | One-to-one (Hungarian) |
| **Post-proc** | Cần NMS | Không cần NMS |
| **Training** | Nhanh, ổn định | Cần nhiều epoch hơn |
| **Inference** | Fast (dense + NMS) | Có thể nhanh (không NMS) |
| **Flexibility** | Cần tune NMS params | End-to-end, ít hyperpar |

### 12.4 Quick Reference — Code YOLO (Ultralytics)

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')          # nano; hoặc yolo11n.pt

# Train
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'
)

# Inference
results = model('image.jpg', conf=0.25, iou=0.45)
for r in results:
    boxes = r.boxes.xyxy    # (N, 4) bounding boxes
    confs = r.boxes.conf    # (N,) confidence scores
    classes = r.boxes.cls   # (N,) class indices

# Evaluate mAP
metrics = model.val(data='data.yaml')
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

---

📸 **Ảnh slide quan trọng cần bổ sung:**

| Slide | Nội dung | Nguồn gợi ý |
|-------|----------|-------------|
| Page 7 | Minh họa detection thực tế (CCTV, xe cộ) | Camera thực tế |
| Page 15-16 | Backbone multi-scale P3/P4/P5 | "feature pyramid YOLO" |
| Page 17 | FPN + PAN topology | "FPN PAN feature pyramid" |
| Page 24-26 | Detection head architecture + shapes | "YOLO detection head" |
| Page 32-33 | DFL nhãn mềm hai bin | "distribution focal loss DFL" |
| Page 47-52 | NMS step-by-step (5 bước) | Vẽ lại từ slide |
| Page 59 | PR Curve đẹp | "precision recall curve" |
| Page 71-73 | DETR architecture tensor view | "DETR architecture" |
| Page 75-78 | Hungarian matching ma trận | "Hungarian algorithm assignment" |
