# Bài 14 — Phân Đoạn Ảnh (Image Segmentation)
**DL4CV-S · HCMUT · Giảng viên: Lê Thanh Sạch**

---

## Mục lục
1. [Giới thiệu và định nghĩa bài toán](#1-giới-thiệu)
2. [Kiến trúc lõi: Encoder–Decoder và UNet](#2-kiến-trúc-unet)
3. [Hàm mất mát](#3-hàm-mất-mát)
4. [Độ đo và đánh giá](#4-độ-đo)
5. [Data Pipeline và Loader](#5-data-pipeline)
6. [Chủ đề nâng cao](#6-nâng-cao)
7. [Tổng kết](#7-tổng-kết)

---

## 1. Giới thiệu

### 1.1 Image Segmentation là gì?

**Phân đoạn ảnh** (image segmentation) là bài toán **gán nhãn lớp cho từng pixel** trong ảnh — còn gọi là **dense prediction** (dự đoán dày đặc). Đây là mức độ hiểu ảnh sâu nhất trong bộ ba: classification → detection → segmentation.

> **Phép ẩn dụ:** Classification trả lời "trong ảnh có gì?", Detection trả lời "nó ở đâu?" bằng hộp giới hạn, còn Segmentation trả lời "mỗi pixel thuộc về vật gì?" — như tô màu từng ô trong tranh ghép hình.

**Góc nhìn tensor:**

```
Batch đầu vào:  X ∈ ℝ^(B×C×H×W)
Logits đầu ra:  Z ∈ ℝ^(B×K×H×W)  — K lớp, mỗi lớp có H×W logit map
Nhãn dự đoán:  Ŷ = argmax_k Z[b, k, h, w]  — class index map ∈ {0,...,K-1}^(H×W)
```

📸 [Cần ảnh: So sánh output của classification (1 label), detection (bounding boxes), segmentation (per-pixel color map)]

### 1.2 So sánh ba bài toán thị giác

| Task | Output | Cấu trúc đầu ra | Metric điển hình |
|------|--------|-----------------|------------------|
| **Classification** | 1 label/ảnh | Thấp — 1 vector | Accuracy, Top-5 |
| **Detection** | Boxes + classes | Trung bình — danh sách | mAP |
| **Segmentation** | Mask / masks | Cao — toàn không gian | mIoU, Dice, PQ |

**Takeaway:** Segmentation kết hợp đặc trưng thị giác với bố cục không gian — mỗi quyết định phân loại đều phải có vị trí.

### 1.3 Các loại phân đoạn ảnh

#### a) Semantic Segmentation (Phân đoạn ngữ nghĩa)

- Mỗi pixel nhận **một nhãn lớp** (road, sky, person, ...) — **không** phân biệt từng đối tượng riêng lẻ.
- Đầu ra: class index map `Y ∈ {1,...,K}^(H×W)` cho mỗi ảnh.
- **Hai xe đạp cạnh nhau = cùng màu**, không tách biệt thành xe 1 và xe 2.

**Ứng dụng:** Scene parsing (ADE20K, Cityscapes), phân vùng mô/tạng y khoa, bản đồ lớp phủ đất.

📸 [Cần ảnh: Ảnh đường phố với semantic segmentation — đường, xe, người, trời mỗi màu khác nhau]

#### b) Instance Segmentation (Phân đoạn thực thể)

- **Tách riêng từng đối tượng** thuộc cùng một lớp: "đối tượng nào là đối tượng nào".
- Đầu ra: tập các mask `{m_i}`, kèm class id và confidence score.
- **Hai xe đạp = hai màu riêng biệt**, dù cùng lớp "bicycle".

**Khó hơn semantic segmentation:** Cần xử lý grouping (gom pixel → instance) và duplicate suppression; các lỗi kiểu detection (FP/FN) cũng xuất hiện.

#### c) Panoptic Segmentation (Phân đoạn toàn diện)

- Kết hợp **things** (đối tượng đếm được: xe, người, chó) và **stuff** (vùng nền vô định hình: bầu trời, đường, cỏ).
- Các vùng **không chồng lấn** và **phủ toàn bộ pixel** — mỗi pixel thuộc đúng một vùng.

**Metric: Panoptic Quality (PQ)**

```
PQ = SQ × RQ

SQ (Segmentation Quality) = Σ_{(p,g)∈TP} IoU(p,g) / |TP|
                           = IoU trung bình trên các cặp match đúng

RQ (Recognition Quality) = |TP| / (|TP| + ½|FP| + ½|FN|)
                          = F1-score của detection
```

> **Giải thích trực giác:** SQ đo mask có khớp tốt không (chất lượng vùng), RQ đo có phát hiện đúng không (chất lượng nhận dạng). PQ cần cả hai tốt mới cao.

📸 [Cần ảnh: Panoptic segmentation — things (xe, người) có ID riêng, stuff (đường, trời) phủ nền]

### 1.4 Tại sao segmentation quan trọng?

- **Medical imaging:** Biên của cơ quan/khối u ảnh hưởng trực tiếp đến quyết định lâm sàng — phẫu thuật 1mm sai vị trí có thể gây nguy hiểm.
- **Autonomous driving / robotics:** Vùng di chuyển được (drivable area), lane detection, phát hiện vật cản.
- **Satellite / industrial:** Phân loại thửa đất, phát hiện lỗi bề mặt vật liệu, vùng thực vật.

**Góc nhìn kỹ thuật:** Lỗi segmentation thường nằm ở **boundary** và **small object**, không chỉ là "sai lớp" ở vùng rộng.

### 1.5 Logits và xác suất theo pixel

Mô hình tạo ra **logit** `z_{i,k}` cho mỗi pixel `i` và lớp `k`. Softmax theo chiều lớp cho xác suất:

$$p_{i,k} = \frac{\exp(z_{i,k})}{\sum_{c=1}^{K} \exp(z_{i,c})}$$

- **Khi huấn luyện:** Loss tính trên logits (PyTorch `CrossEntropyLoss` nhận logits trực tiếp — không cần softmax trước).
- **Khi triển khai:** `ŷ_i = argmax_k p_{i,k}` tạo class map.

---

## 2. Kiến trúc UNet

### 2.1 Dense Prediction Pipeline — Tổng quan

Mọi mô hình segmentation đều theo cấu trúc tổng quát:

```
X → Backbone → Neck (tùy chọn) → Head → Z
         ↑              ↑             ↑
   Trích đặc trưng   Tinh chế    Chiếu sang K lớp
```

- **Backbone:** Bộ máy trích đặc trưng và tái dựng cấu trúc không gian.
- **Neck:** Tầng tùy chọn để refinement/fusion/adaptation trước head.
- **Head:** Phép chiếu sang đầu ra bài toán — thường là `Conv2d 1×1` với K filters.

> **Phép ẩn dụ:** Backbone là bộ não hiểu nội dung ảnh, Neck là bộ lọc thông minh tinh chỉnh hiểu biết đó, Head là chiếc bút viết câu trả lời cuối cùng.

### 2.2 UNet-style Backbone

UNet (Ronneberger et al., MICCAI 2015) là mô hình tiêu chuẩn cho dense prediction:

```
BackboneUNet = Encoder + Bottleneck + Decoder + SkipConnections
```

📸 [Cần ảnh: Kiến trúc UNet — hình chữ U với encoder bên trái, bottleneck ở đáy, decoder bên phải, mũi tên skip connections ngang]

**Luồng dữ liệu:**

```
Input (B,3,H,W)
  → Enc1: DoubleConv → (B,64,H,W)    ─────────────────┐ skip1
  → Pool → (B,64,H/2,W/2)                              │
  → Enc2: DoubleConv → (B,128,H/2,W/2)  ─────────────┐ │ skip2
  → Pool → (B,128,H/4,W/4)                            │ │
  → Enc3: DoubleConv → (B,256,H/4,W/4)  ────────────┐ │ │ skip3
  → Pool → (B,256,H/8,W/8)                          │ │ │
  → Bottleneck:      → (B,512,H/8,W/8)              │ │ │
  → Up3+skip3: concat→ (B,512,H/4,W/4) ←────────────┘ │ │
  → Up2+skip2: concat→ (B,256,H/2,W/2) ←──────────────┘ │
  → Up1+skip1: concat→ (B,128,H,W)     ←────────────────┘
  → Head (Conv1×1) → Logits (B,K,H,W)
```

**Tensor shapes ví dụ** (c = [64, 128, 256, 512]):

| Stage | Typical shape |
|-------|---------------|
| Input | (B, 3, H, W) |
| Encoder level ℓ | (B, c_ℓ, H_ℓ, W_ℓ) với H_ℓ giảm dần |
| Bottleneck | (B, c_L×2, H_L, W_L) — thô nhất, context mạnh nhất |
| Decoder level ℓ | upsample dần về (B, ·, H, W) |
| Logits | (B, K, H, W) |

### 2.3 Các thành phần của UNet

#### a) Encoder — Thu nhỏ và học đặc trưng

- **Conv blocks:** Xếp chồng convolutions, BN, ReLU — học texture cục bộ dần thành pattern ngữ nghĩa.
- **Downsampling:** MaxPool2d(2) hoặc strided conv — giảm (H,W), mở rộng receptive field.
- Tạo feature hierarchy: từ chi tiết mịn (edges/textures) → biểu diễn thô hơn (objects/parts).

#### b) Bottleneck — Context toàn cục

- Feature map **sâu nhất, thô nhất** — có context mạnh nhất (receptive field lớn nhất).
- Thường có nhiều kênh nhất (512, 1024); chi phí tính toán trên mỗi pixel cao nhưng số pixel ít hơn.

> **Phép ẩn dụ:** Bottleneck như "bước dừng và suy nghĩ toàn cảnh" — mô hình đã thu nhỏ ảnh xuống còn 1/16 kích thước gốc và có góc nhìn toàn cục trước khi phóng to trở lại.

#### c) Decoder — Phóng to và khôi phục chi tiết

- **Upsampling:** `ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)` hoặc bilinear interpolate + conv — tăng dần độ phân giải về gần kích thước đầu vào.
- **Feature fusion:** Kết hợp thông tin ngữ nghĩa thô từ decoder với tín hiệu không gian mịn từ skip connections.
- Nếu không có skip connections, upsampling đơn thuần tạo mask mờ.

#### d) Skip Connections — Tuyến đường tắt cho chi tiết biên

- **Vấn đề:** Pooling làm mất các biên và cấu trúc mảnh có tần số cao.
- **Giải pháp:** Skip truyền trực tiếp thông tin định vị từ encoder sang decoder cùng cấp — mô hình có thể "nhìn lại" bản đồ đặc trưng gốc.

**Concat vs Addition:**

| | Concatenation | Addition |
|--|--------------|----------|
| **Cách làm** | `torch.cat([x_up, skip], dim=1)` | `x_up + skip` |
| **Yêu cầu** | Không cần số kênh khớp | Phải cùng số kênh |
| **Ưu điểm** | Giữ đặc trưng tách biệt, chi tiết biên rõ hơn | Nhẹ hơn |
| **Dùng khi** | UNet chuẩn — khi biên quan trọng | ResNet-style residuals |

**Insight:** Skip connection là cơ chế chính giúp UNet tạo biên sắc nét — không có skip, mô hình phải "đoán lại" thông tin vị trí từ đặc trưng đã nén.

### 2.4 Neck — Tầng tinh chế trước Head

```
F_neck = g_φ(F_backbone),    Z = Conv_{1×1}(F_neck)
```

Neck là tầng tùy chọn nằm giữa backbone và head:

| Neck type | Ví dụ | Vai trò |
|-----------|-------|---------|
| **Identity** | Classical UNet | Không có neck |
| **Conv neck** | một/nhiều Conv2d blocks | Tinh chỉnh kênh |
| **ASPP neck** | DeepLabv3+ | Context đa tỉ lệ |
| **Attention neck** | SE / CBAM / Transformer attention | Tái trọng số kênh/không gian |
| **Fusion neck** | FPN / BiFPN / SegFormer MLP | Multi-scale fusion |
| **Boundary neck** | edge refinement module | Sắc nét hóa biên |

### 2.5 Segmentation Head

Head chiếu số kênh đặc trưng sang đầu ra bài toán — thường là **Conv2d 1×1**:

```
F ∈ ℝ^(B×C×H×W)  →  Head = Conv2d(C, K, kernel_size=1)  →  Z ∈ ℝ^(B×K×H×W)
```

- Head thường **nhỏ** nhưng quyết định ngữ nghĩa của đầu ra.
- **Không dùng softmax trong head** nếu dùng `CrossEntropyLoss` — PyTorch tích hợp sẵn log-softmax vào CE loss để ổn định số học.

**Outputs khi inference:**

| Trường hợp | Output | Cách tính |
|-----------|--------|-----------|
| Multi-class (K>2) | Hard labels | `pred = logits.argmax(dim=1)` |
| Binary | Soft probs | `probs = torch.sigmoid(logits)` |
| Probability maps | Per-class probs | `probs = torch.softmax(logits, dim=1)` |

### 2.6 Classification vs Segmentation Backbone

| | Classification | Segmentation |
|--|---------------|-------------|
| **Sau encoder** | Global pooling → vector | Bottleneck + decoder → map |
| **Output spatial** | Mất thông tin vị trí | Khôi phục không gian |
| **Head** | FC layer | Conv 1×1 |
| **Pipeline** | image → encoder → pool → FC | image → encoder → bottleneck → decoder → dense map |

### 2.7 Ánh xạ các mô hình phổ biến

| Model | Backbone | Neck | Head |
|-------|---------|------|------|
| **Classical UNet** | Enc-Bot-Dec + skips | Identity/none | 1×1 Conv |
| **ResUNet** | ResNet encoder + UNet decoder | optional Conv | 1×1 Conv |
| **DeepLabv3+** | CNN encoder + decoder | ASPP + refinement | 1×1 Conv |
| **SegFormer** | Hierarchical Transformer encoder | MLP fusion neck | linear / 1×1 |
| **Mask2Former** | Pixel encoder + Transformer decoder | query/mask refinement | mask-class heads |
| **SAM** | Image encoder + prompt/mask decoder | prompt-conditioned | mask prediction |

**Cách đọc paper segmentation bất kỳ:**
1. Encoder/visual backbone là gì?
2. Mô hình khôi phục độ phân giải không gian bằng cách nào?
3. Có skip connection hoặc multi-scale fusion nào?
4. Có neck không, và neck thực hiện refinement kiểu gì?
5. Head có định dạng đầu ra như thế nào?
6. Loss và metric nào định nghĩa thành công?

> **Takeaway:** Phần lớn paper thay đổi backbone, neck, head, loss, hoặc training protocol — hiếm khi có ý tưởng hoàn toàn mới.

### 2.8 CNN vs Transformer Segmentation

| | CNN UNet | Transformer (ViT/SegFormer) |
|--|---------|----------------------------|
| **Inductive bias** | Locality, translation equivariance | Không có — học từ data |
| **Thế mạnh** | Local prior mạnh; hiệu quả ở độ phân giải vừa | Trộn thông tin xa tốt hơn |
| **Hạn chế** | Receptive field hữu hạn trước bottleneck | Tốn bộ nhớ hơn ở full resolution |
| **Data efficiency** | Tốt với ít data | Cần nhiều data hơn |

**Hybrid (phổ biến nhất hiện nay):** Transformer encoder (global context) + CNN/UNet decoder (spatial detail) — cân bằng hai phía.

### 2.9 Memory Footprint

VRAM tăng theo: batch size × độ phân giải × số kênh skip × độ rộng decoder.

**Các kỹ thuật tiết kiệm bộ nhớ:**
- Patch training (chia ảnh lớn thành patches nhỏ hơn)
- Mixed precision (fp16/bf16)
- Gradient checkpointing
- Giảm base channels

> **Thực tế:** Segmentation thường chạm giới hạn bộ nhớ **trước** khi chạm giới hạn độ chính xác. Bộ nhớ là tài nguyên số 1 cần quản lý.

### 2.10 Code PyTorch — Toàn bộ UNet

#### DoubleConv Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
```

> Hai conv mỗi stage là conv block tối thiểu rất phổ biến — đủ để học cả edge detector và semantic filter trong một block.

#### Encoder Step (Down)

```python
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)     # feature map trước pool → lưu cho skip connection
        return self.pool(skip), skip
```

> Trả về cả tensor đã pool (cho nhánh sâu hơn) và skip (cho decoder concat).

#### Decoder Step (Up)

```python
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)  # in_ch vì concat

    def forward(self, x, skip):
        x = self.up(x)                          # upsample ×2
        # Crop skip nếu kích thước lẻ (ảnh H/W không chia hết cho 2^n)
        if skip.shape[-2:] != x.shape[-2:]:
            skip = skip[..., :x.size(-2), :x.size(-1)]
        x = torch.cat([x, skip], dim=1)         # concat theo kênh
        return self.conv(x)
```

#### Segmentation Head

```python
class SegHead(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, n_classes, kernel_size=1)

    def forward(self, x):
        return self.proj(x)   # logits (B, K, H, W) — không softmax ở đây!
```

#### UNet đầy đủ

```python
class UNet(nn.Module):
    def __init__(self, n_classes, in_ch=3, c=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        prev = in_ch
        for ch in c:
            self.downs.append(Down(prev, ch))
            prev = ch
        self.bot = DoubleConv(prev, prev * 2)   # bottleneck: 512 → 1024
        self.ups = nn.ModuleList()
        rev = list(reversed(c))
        bot_ch = prev * 2                        # 1024
        for i, ch in enumerate(rev):
            in_up = bot_ch if i == 0 else rev[i-1]
            self.ups.append(Up(in_up, ch))
        self.head = SegHead(rev[-1], n_classes)  # 64 → K

    def forward(self, x):
        skips = []
        for down in self.downs:
            x, s = down(x)
            skips.append(s)
        x = self.bot(x)
        for up, sk in zip(self.ups, reversed(skips)):
            x = up(x, sk)
        return self.head(x)   # (B, K, H, W)
```

#### Training & Inference

```python
# Training step
criterion = nn.CrossEntropyLoss(ignore_index=255)
logits = model(images)                  # (B, K, H, W)
loss = criterion(logits, masks_long)    # masks_long: (B, H, W) long tensor
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    logits = model(images)
    pred = logits.argmax(dim=1)         # (B, H, W) class index map
    probs = torch.softmax(logits, dim=1)  # (B, K, H, W) nếu cần probability
```

**Lưu ý quan trọng về kích thước:**

```python
# Nếu model output khác kích thước target:
logits = model(images)    # (B, K, H', W') — có thể nhỏ hơn
target = masks_long       # (B, H, W)
# Resize logits (KHÔNG resize mask bằng bilinear!)
logits = F.interpolate(logits, size=target.shape[-2:],
                       mode='bilinear', align_corners=False)
loss = criterion(logits, target)
```

> **⚠️ Cảnh báo:** **Không bao giờ** bilinear-resize class-index masks — `bilinear` sẽ nội suy giữa các class id số nguyên tạo ra nhãn không tồn tại (ví dụ class 1 và class 2 → class 1.5). Với mask phải dùng `nearest-neighbor`.

---

## 3. Hàm Mất Mát

### 3.1 Tại sao cần nhiều hơn một softmax CE?

- **Mất cân bằng lớp:** Trong ảnh y khoa, foreground (khối u) có thể chỉ chiếm 1-5% pixels — CE tối ưu background quá nhiều.
- **Mờ ở biên:** CE xử lý từng pixel độc lập, không ưu tiên khu vực biên vật thể.
- **Overlap metrics:** mIoU/Dice là metric đánh giá thực tế — CE thuần không trực tiếp tối ưu chúng.

> **Phép ẩn dụ:** Chỉ dùng CE thuần khi dữ liệu mất cân bằng giống như đánh giá học sinh theo điểm trung bình khi 95% câu hỏi là câu dễ — học sinh chỉ cần trả lời đúng câu dễ là đạt điểm cao mà không cần học phần khó.

### 3.2 Pixel Cross-Entropy (CE)

**Công thức:**

$$\mathcal{L}_{CE} = -\frac{1}{|I|} \sum_{i \in I} \log p_{i, y_i}$$

Với `p_{i,yi}` là xác suất model gán cho nhãn đúng `y_i` của pixel `i`.

**Ưu điểm:**
- Baseline ổn định, dễ debug.
- Hỗ trợ `ignore_index` để bỏ qua void pixels (nhãn 255 trong PASCAL VOC/Cityscapes).
- Đạo hàm trơn, hội tụ ổn định.

**Nhược điểm:**
- Nhạy cảm với class imbalance — background pixels chiếm đa số sẽ dominate gradient.

**Với binary head:** Dùng `BCEWithLogitsLoss` theo pixel hoặc channel (stable hơn BCE + sigmoid riêng).

```python
ce = nn.CrossEntropyLoss(ignore_index=255)
loss = ce(logits, target)  # logits: (B,K,H,W), target: (B,H,W) long
```

### 3.3 Dice Coefficient và Dice Loss

**Dice coefficient** (còn gọi là F1 score trên pixel):

$$\text{Dice}_k = \frac{2 \sum_i p_{i,k} g_{i,k} + \epsilon}{\sum_i p_{i,k} + \sum_i g_{i,k} + \epsilon}$$

Với `g_{i,k}` là ground truth (binary: 1 nếu pixel i thuộc lớp k), `p_{i,k}` là predicted probability.

> **Giải thích:** Tử số `2×|A∩B|` = 2 lần vùng giao, mẫu số `|A|+|B|` = tổng kích thước hai tập. Dice = 1 khi hoàn hảo, 0 khi không chồng lấn.

**Dice Loss:**

$$\mathcal{L}_{Dice} = 1 - \text{Dice}$$

**Đặc điểm:**
- **Mạnh với foreground nhỏ** — tập trung vào overlap của lớp quan tâm, không bị áp đảo bởi background.
- Có thể kém ổn định **lúc đầu** (khi `Σp` gần 0, gradient bất ổn) → cần ε.
- Khi dùng trong training: dùng **soft Dice** từ xác suất (không threshold); khi báo cáo: tính từ **hard masks** sau argmax.

### 3.4 IoU / Jaccard và IoU Loss

**IoU per class:**

$$\text{IoU}_k = \frac{TP_k}{TP_k + FP_k + FN_k}$$

**IoU Loss:**

$$\mathcal{L}_{IoU} = 1 - \text{IoU}$$

Hoặc dùng surrogate mềm **Lovász loss** (differentiable IoU) cho training ổn định hơn.

**Quan hệ Dice–IoU:**

$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$

Dice luôn ≥ IoU; khi IoU = 0.5 → Dice = 0.667.

### 3.5 Ví dụ tính toán số học

**Bài toán:** Binary segmentation với mask 3×3:

```
Ground Truth G:          Prediction Ŷ:
┌─────────────┐          ┌─────────────┐
│  1  │  1  │  0  │      │  1  │  0  │  0  │
│  1  │  0  │  0  │      │  1  │  1  │  0  │
│  0  │  0  │  0  │      │  0  │  0  │  0  │
└─────────────┘          └─────────────┘
```

**Bước 1: Đếm TP, FP, FN**
- Pixel (0,0): G=1, Ŷ=1 → **TP**
- Pixel (0,1): G=1, Ŷ=0 → **FN**
- Pixel (1,0): G=1, Ŷ=1 → **TP**
- Pixel (1,1): G=0, Ŷ=1 → **FP**

→ TP=2, FP=1, FN=1

**Bước 2: Tính Dice và IoU**

$$\text{Dice} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} = \frac{2 \times 2}{4 + 1 + 1} = \frac{4}{6} \approx 0.667$$

$$\text{IoU} = \frac{TP}{TP + FP + FN} = \frac{2}{2 + 1 + 1} = \frac{2}{4} = 0.5$$

**Kiểm tra:** Dice = 2×0.5/(1+0.5) = 0.667 ✓

**Bước 3: Tính Loss**

$$\mathcal{L}_{Dice} = 1 - 0.667 = 0.333$$
$$\mathcal{L}_{IoU} = 1 - 0.5 = 0.5$$

> **Nhận xét:** Overlap tốt hơn ⇒ Dice/IoU cao hơn ⇒ DiceLoss/IoULoss thấp hơn. Mô hình cải thiện khi TP tăng, FP và FN giảm.

### 3.6 Hybrid Loss: CE + Dice

**Công thức:**

$$\mathcal{L} = \lambda_{CE} \mathcal{L}_{CE} + \lambda_{Dice} \mathcal{L}_{Dice}$$

**Lý do kết hợp:**
- **CE:** Ổn định tối ưu, gradient ở tất cả pixels ngay từ đầu.
- **Dice:** Kéo mô hình về chất lượng overlap, penalize mất cân bằng.
- Thông thường: `λ_CE = λ_Dice = 0.5` hoặc tune theo validation.

**Code PyTorch — CE + Dice Loss:**

```python
ce = nn.CrossEntropyLoss(ignore_index=255)

def ce_dice_loss(logits, target, eps=1e-6, lam=0.5):
    # logits: (B, K, H, W), target: (B, H, W) long
    loss_ce = ce(logits, target)

    # Tính soft Dice từ probabilities
    valid = (target != 255)
    t = target.clone()
    t[~valid] = 0  # safe index để tạo one_hot

    one_hot = F.one_hot(t, num_classes=logits.size(1))   # (B,H,W,K)
    one_hot = one_hot.permute(0, 3, 1, 2).float()        # (B,K,H,W)
    probs = torch.softmax(logits, dim=1)                  # (B,K,H,W)

    valid = valid.unsqueeze(1)                            # (B,1,H,W) broadcast
    inter = (probs * one_hot * valid).sum((0, 2, 3))     # sum over B,H,W
    denom = ((probs + one_hot) * valid).sum((0, 2, 3))
    loss_dice = 1 - ((2 * inter + eps) / (denom + eps)).mean()

    return loss_ce + lam * loss_dice
```

### 3.7 Weighted CE — Xử lý mất cân bằng lớp

**Ý tưởng:** Tăng penalty cho lớp hiếm, giảm cho lớp phổ biến:

$$\mathcal{L}_{wCE} = -\frac{1}{|I|} \sum_{i \in I} w_{y_i} \log p_{i, y_i}$$

**Tính class weights từ inverse frequency:**

```python
# Ví dụ: lớp background chiếm 90%, foreground 10%
# w_background = 1/0.90 ≈ 1.11
# w_foreground  = 1/0.10 = 10.0
# (thường smooth hoặc clip để tránh quá cực đoan)

class_counts = torch.tensor([9000, 1000], dtype=torch.float)
weights = 1.0 / class_counts
weights = weights / weights.sum() * len(weights)  # normalize
ce = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=255)
```

### 3.8 Focal Loss và Tversky Loss

**Focal Loss** (RetinaNet):

$$\mathcal{L}_{Focal} = -(1 - p_{i,y_i})^\gamma \log p_{i,y_i}$$

- **Nhấn mạnh pixels khó** (p thấp) — giảm đóng góp của pixels dễ (p cao).
- Hữu ích khi background pixels áp đảo — focal giảm ảnh hưởng của background easy examples.

**Tversky Loss:**

$$T_k = \frac{TP_k}{TP_k + \alpha \cdot FP_k + \beta \cdot FN_k}$$

$$\mathcal{L}_{Tversky} = 1 - T_k$$

- Cho phép **cân bằng FP/FN bất đối xứng** — ví dụ trong y khoa: bỏ sót lesion nguy hiểm hơn báo động giả → đặt β > α để penalize FN nhiều hơn.
- Dice là trường hợp đặc biệt với α = β = 0.5.

### 3.9 Hướng dẫn chọn loss trong thực hành

| Tình huống | Gợi ý |
|-----------|-------|
| Bắt đầu project mới | CE + ignore_index + kiểm tra augmentation nghiêm túc |
| Overlap metrics bị chững | Thêm Dice/IoU term |
| Mất cân bằng lớp nghiêm trọng | Weighted CE hoặc Focal |
| Background áp đảo | Focal Loss |
| Cần ưu tiên recall (bỏ sót nguy hiểm) | Tversky với β > α |
| Tune λ | Theo **validation metric**, không chỉ nhìn training loss |

---

## 4. Độ Đo

### 4.1 Tại sao evaluation protocol quan trọng?

- Metrics định nghĩa thế nào là "tốt hơn" — sai protocol dẫn đến chọn model sai cho production.
- **Engineering pitfalls:**
  - Logits và mask phải có cùng (H, W) sau resize.
  - `ignore_index` phải nhất quán trong cả loss lẫn metrics.
  - Sai khác resize/crop có thể làm IoU/Dice sai âm thầm vài điểm %.

### 4.2 IoU và mIoU (Mean Intersection over Union)

**Per-class IoU:**

$$\text{IoU}_k = \frac{TP_k}{TP_k + FP_k + FN_k}$$

**Mean IoU — macro average qua tất cả lớp:**

$$\text{mIoU} = \frac{1}{K} \sum_{k=1}^{K} \text{IoU}_k$$

> **"Macro average"** = tất cả lớp được tính ngang nhau — lớp hiếm (pedestrian) và lớp phổ biến (road) đều đóng góp 1/K vào mIoU. Đây là lý do mIoU phản ánh performance trên lớp hiếm tốt hơn pixel accuracy.

**Cách tính TP, FP, FN** cho multi-class:
- `TP_k` = số pixels mà GT=k và Pred=k (hit đúng lớp k)
- `FP_k` = số pixels mà GT≠k nhưng Pred=k (mô hình nhầm sang lớp k)
- `FN_k` = số pixels mà GT=k nhưng Pred≠k (bỏ sót lớp k)

### 4.3 Dice (F1) Score per Class

$$\text{Dice}_k = \frac{2 \cdot TP_k}{2 \cdot TP_k + FP_k + FN_k}$$

- Rất phổ biến trong **medical imaging** — thường báo cáo kèm mIoU.
- Mean Dice = `(1/K) Σ Dice_k` — tương tự mIoU nhưng metric khác.

### 4.4 Hausdorff Distance 95% (HD95)

- Đo sai khác biên bằng **khoảng cách giữa các contour** (pixel):

$$HD(A, B) = \max\left(\max_{a \in A} \min_{b \in B} d(a,b),\ \max_{b \in B} \min_{a \in A} d(a,b)\right)$$

- **HD95** = dùng **95th percentile** thay vì max — giảm ảnh hưởng của outlier (một pixel biên sai xa tạo HD cực lớn).

**Khi nào dùng:** Bài toán nhạy với biên: surgical margins, cấu trúc mảnh như mạch máu/dây thần kinh.

> **Phép ẩn dụ:** Nếu mIoU là "tỉ lệ vùng đúng", HD95 là "mask của bạn lệch biên bao nhiêu mm" — câu hỏi đặc biệt quan trọng trong phẫu thuật.

### 4.5 Pixel Accuracy và Tại Sao Dễ Gây Hiểu Lầm

$$\text{PixAcc} = \frac{\sum_k TP_k}{\sum_k (TP_k + FN_k)}$$

**Vấn đề:** Với dữ liệu mất cân bằng, background chiếm đa số.

**Ví dụ minh họa:** Ảnh y khoa với 95% background, 5% lesion.  
- Mô hình đoán toàn background → PixAcc = 95% (cao!) nhưng mIoU ≈ 0% (phá sản).
- Mô hình giỏi lesion → PixAcc = 90% thấp hơn nhưng mIoU = 70% (tốt hơn nhiều).

→ **Luôn kiểm tra per-class IoU/Dice**, không chỉ nhìn aggregate metrics.

### 4.6 Hướng dẫn chọn và báo cáo metric

| Tình huống | Metric khuyến nghị |
|-----------|-------------------|
| Mất cân bằng lớp | mIoU / mean Dice |
| Chất lượng biên | HD95 hoặc boundary-F |
| Instance segmentation | AP family |
| Panoptic segmentation | PQ = SQ × RQ |
| Medical imaging | Dice + HD95 |

**Khi báo cáo kết quả — phải ghi rõ:**
- Split protocol (train/val/test ratio, ngẫu nhiên hay stratified)
- Danh sách lớp và ignore rules (void class có bị loại không?)
- Tiêu chí chọn checkpoint (best val mIoU? last epoch?)
- Có dùng TTA (test-time augmentation) không?

---

## 5. Data Pipeline

### 5.1 Cấu trúc thư mục điển hình

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── val/
│       └── img_003.jpg
├── masks/
│   ├── train/
│   │   ├── img_001.png   # ← tên file tương ứng
│   │   └── img_002.png
│   └── val/
│       └── img_003.png
└── splits.csv             # hoặc JSON ánh xạ đường dẫn tường minh
```

**Quy tắc:** Tên file image và mask phải tương ứng một-một. Dùng CSV/JSON để tránh lỗi khi file bị đổi tên.

### 5.2 Mask Encoding

| Encoding | Mô tả | Dùng khi |
|---------|-------|---------|
| **Index map** | Mỗi pixel = class id (integer) như `0, 1, 2, ...` | CrossEntropyLoss — phổ biến nhất |
| **One-hot** | (K, H, W) binary tensor | Một số loss (Dice), tốn bộ nhớ hơn K lần |
| **RGB color map** | Mỗi màu = một lớp (ví dụ Cityscapes) | Visualization, cần decode trước khi dùng |

> **Thực hành:** Lưu mask dưới dạng PNG (lossless) với index map — JPEG nén lossy sẽ tạo giá trị interpolated không phải class id.

### 5.3 Tensor Shapes sau khi Load

```
Image tensor:  (3, H, W)          → batch: (B, 3, H, W)  float32
Mask tensor:   (H, W)  dtype=long  → batch: (B, H, W)      int64
```

Kích thước không gian của logits và mask **phải khớp** sau mọi resize/crop.

### 5.4 Quy tắc Augmentation

**Quy tắc vàng:** Biến đổi hình học phải được áp dụng **cùng một transform** cho cả image và mask.

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(height=512, width=512),
    A.Rotate(limit=30, p=0.5),
    # Chỉ áp dụng color jitter cho image, không cho mask:
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
], additional_targets={'mask': 'mask'})

result = transform(image=img_np, mask=mask_np)
img_aug, mask_aug = result['image'], result['mask']
```

**⚠️ Interpolation cho mask: chỉ dùng `nearest`** (không phải bilinear hay cubic) khi resize/rotate mask label.

```python
# Resize logits OK với bilinear (float tensor)
logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

# Resize mask PHẢI dùng nearest (integer tensor)
mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').long().squeeze(1)
```

### 5.5 Split Discipline và Leakage

**Nguy cơ leakage:**
- Các bản augmented của cùng một sample vào các split khác nhau → data leakage.
- Near-duplicate frames từ video datasets (frame t và frame t+1 quá giống nhau).
- Các bệnh nhân trong y khoa — cùng patient không được xuất hiện ở cả train lẫn val/test.

**Nguyên tắc chia split:**
- Chia theo **patient/video/scene/source** khi phù hợp — không chia theo ảnh.
- Kiểm tra histogram của lớp phân bố giống nhau ở train và val.

> **Cảnh báo:** Leakage có thể tạo mIoU/Dice cao giả và làm hệ thống rất yếu khi triển khai thật — đây là lỗi phổ biến nhất trong papers medical segmentation.

### 5.6 Visual Sanity Checks — Checklist trước khi train

1. ✅ Kiểm tra overlay image + mask bằng matplotlib (vẽ chồng hai ảnh, alpha=0.5)
2. ✅ Kiểm tra color map và ánh xạ class-id
3. ✅ Kiểm tra mask interpolation là nearest-neighbor
4. ✅ Kiểm tra ignore_label có đúng (255 hay 0?) trong cả loss lẫn metrics
5. ✅ Kiểm tra một mini-batch sau augmentation — mask có bị lệch không?
6. ✅ Vẽ histogram tần suất lớp — có mất cân bằng không?

### 5.7 Code PyTorch Dataset và DataLoader

**SegDataset:**

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        msk = np.array(Image.open(self.mask_paths[idx]))  # index map PNG

        if self.transforms:
            result = self.transforms(image=img, mask=msk)
            img, msk = result['image'], result['mask']

        # Convert: numpy HWC → tensor CHW, mask → long
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor, torch.from_numpy(msk).long()
```

**DataLoader:**

```python
loader = torch.utils.data.DataLoader(
    ds,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,       # faster CPU→GPU transfer
    drop_last=True,        # tránh batch size=1 với BatchNorm
)
```

**Lưu ý:** Dùng `collate_fn` nếu tensor có kích thước biến đổi cần padding (ví dụ ảnh không cùng kích thước).

### 5.8 Ignore Index, Class Imbalance, Checklist Lỗi

```python
# ignore_index cho void pixels (PASCAL VOC: 255, Cityscapes: 255)
ce = nn.CrossEntropyLoss(ignore_index=255)
# Phải nhất quán giữa loss và metrics!

# Tính class weights từ histogram
class_counts = compute_class_histogram(train_loader)  # (K,) tensor
weights = 1.0 / (class_counts + 1)                   # +1 tránh /0
weights = weights / weights.mean()                    # normalize
```

**Checklist lỗi thường gặp:**

| Lỗi | Triệu chứng | Cách debug |
|-----|------------|-----------|
| Bilinear resize mask | mIoU thấp bất thường ở biên | Kiểm tra interpolation mode |
| Off-by-one nhãn | Loss không giảm, class 0 bị bỏ qua | Print unique values của mask |
| Train/val leakage | Val metric quá tốt, drop mạnh khi test | Kiểm tra patient/scene split |
| Resize mismatch (H≠H') | Loss NaN hoặc error | Assert shape sau resize |
| Ignore label không nhất quán | mIoU tăng giả | Dùng cùng ignore_index cho loss và eval |

---

## 6. Nâng Cao

### 6.1 DeepLab + ASPP (Atrous Spatial Pyramid Pooling)

**Vấn đề:** Standard CNN encoder giảm spatial resolution → mất detail. Atrous (dilated) conv giải quyết điều này.

**Atrous Convolution:**
- Dilated conv với rate `r` = conv với filter được "giãn ra" r-1 zeros giữa các kernel weights.
- Receptive field tăng như stride `r`, nhưng **không làm giảm độ phân giải**.
- Mở rộng receptive field mà không downsampling sớm.

**ASPP (Atrous Spatial Pyramid Pooling):**
- Kết hợp atrous conv với nhiều dilation rates khác nhau (ví dụ: 1, 6, 12, 18) song song.
- Mỗi branch capture context ở tỉ lệ khác nhau.
- Concat tất cả branches → rich multi-scale context.

**Khi nào dùng:** Cần context ngữ nghĩa mạnh cho scene parsing với chi phí tính toán vừa phải.

📸 [Cần ảnh: DeepLabv3+ architecture — encoder với ASPP neck, decoder nhẹ kết hợp low-level features]

### 6.2 SegFormer — Transformer cho Segmentation

**Kiến trúc:**
- **Hierarchical transformer encoder** (Mix Transformer - MiT): tạo multi-scale feature maps (giống pyramid của CNN) với patch merging ở mỗi stage.
- **Lightweight MLP decoder:** Chỉ là MLP fusion của các feature maps từ các scale khác nhau — không cần attention hay skip connection phức tạp.

**Ưu điểm:**
- Nhẹ hơn ViT thuần nhờ hierarchical design.
- Decoder cực nhẹ (không cần receptive field lớn vì encoder đã có global context).

**Khi nào dùng:** Cần độ chính xác tốt với decoder hiệu quả cho semantic segmentation.

### 6.3 Mask2Former — Framework Thống Nhất

**Ý tưởng cốt lõi:** Thay vì dense prediction, dùng **mask classification** với transformer decoder queries.

- `N` learned object queries → decoder predicts `N` (mask, class) pairs.
- Huấn luyện với **Hungarian matching** (giống DETR) để gán query → ground truth.
- Một framework duy nhất cho cả semantic, instance, và panoptic segmentation.

**Khi nào dùng:** Muốn thống nhất semantic/instance/panoptic trong một framework — không cần thiết kế 3 model khác nhau.

### 6.4 SAM (Segment Anything Model)

**Segment Anything (Kirillov et al., Meta AI, ICCV 2023):**

- **Promptable foundation model:** Nhận prompts (points, boxes, masks) → tạo mask cho vùng được chỉ định.
- **Kiến trúc:**
  - Image encoder (ViT-H): encode ảnh một lần duy nhất.
  - Prompt encoder: encode điểm/hộp/mask.
  - Mask decoder (lightweight): kết hợp hai embedding → tạo mask.

**Đặc điểm:**
- Có thể segment **bất kỳ vật gì** trong ảnh không có nhãn.
- Hữu ích cho interactive annotation và rapid prototyping.
- **Hạn chế:** Có thể cần adaptation (fine-tuning/SAM 2) khi domain shift mạnh (ảnh y khoa vs ảnh tự nhiên).

📸 [Cần ảnh: SAM interface — user click một điểm → SAM tạo mask cho vật thể]

### 6.5 Weak Supervision

**Bài toán:** Annotation mask pixel-level rất tốn kém (Cityscapes mất ~1.5h/ảnh).

**Weak supervision:** Dùng nhãn "yếu" thay vì mask đầy đủ:
- **Image-level tags:** chỉ biết ảnh có lớp nào (không biết ở đâu).
- **Bounding boxes:** biết vùng gần đúng.
- **Scribbles:** vài đường vẽ bằng tay.

**Phương pháp phổ biến:** Pseudo-label loops — dùng model yếu để tạo pseudo masks, dùng pseudo masks để train model mạnh hơn, lặp lại.

**Trade-off:** Giảm chi phí gán nhãn nhưng tăng độ phức tạp thuật toán và yêu cầu xử lý nhiễu.

### 6.6 Semi-supervised Segmentation

**Bài toán:** Có một số ít ảnh có mask (labeled), và nhiều ảnh không có mask (unlabeled).

**Phương pháp:**
- **Consistency regularization:** Augmentations khác nhau của cùng ảnh không labeled phải tạo output tương tự.
- **Confidence filtering:** Chỉ dùng pseudo labels khi model confident (threshold).
- **Mean Teacher:** Model chính (student) + EMA model (teacher) cung cấp pseudo labels.

**Khi nào dùng:** Có nhiều ảnh không nhãn nhưng annotation pixel-level đắt đỏ.

### 6.7 Foundation Model Direction

**Xu hướng hiện tại:**
- Dùng **shared backbones** (DINOv2, SAM encoder, CLIP) + **adapters** cho downstream dense tasks.
- Không cần train backbone từ đầu — chỉ train adapter/head với data nhỏ.

**Takeaway:** Xem foundation model như **teacher** hoặc **feature provider** — không phải lúc nào cũng là final runtime model (vì quá nặng cho inference thực tế).

### 6.8 Tài nguyên thực hành

- **Surveys & benchmarks:** Cityscapes benchmark, COCO Panoptic, ADE20K.
- **Implementations sẵn có:**
  - [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) — nhiều backbone × decoder combinations.
  - [`mmsegmentation`](https://github.com/open-mmlab/mmsegmentation) — OpenMMLab, hỗ trợ nhiều SOTA models.
  - [`Detectron2`](https://github.com/facebookresearch/detectron2) — Meta AI, instance + panoptic.

---

## 7. Tổng Kết

### 7.1 Công thức thực hành cho project mới

1. **Chốt task type:** Semantic / instance / panoptic — và **metric chính** (mIoU, Dice, PQ?).
2. **Xây dựng dataset contract + visualization sanity checks** — kiểm tra overlay image+mask, histogram lớp, interpolation mode.
3. **Bắt đầu với UNet + CE** — sau đó thêm overlap loss (Dice/IoU) nếu metric chững.
4. **Debug bằng per-class metrics và qualitative grids** — tìm lớp nào mô hình fail, không chỉ nhìn mIoU tổng.

### 7.2 Các lỗi thường gặp (và cách tránh)

| Lỗi | Hậu quả | Phòng tránh |
|-----|---------|-------------|
| **Bilinear resize mask** | Nhãn bị interpolated, mIoU sai | Luôn dùng `mode='nearest'` cho masks |
| **Ignore labels không nhất quán** | Loss giảm nhưng metric không tăng | Đồng nhất `ignore_index` trong loss và eval |
| **Tune trên test set** | Overfitting, false confidence | Chỉ nhìn val set khi tune, test set chỉ dùng một lần |
| **Metric và loss không ăn khớp** | Optimize sai objective | Đảm bảo loss reflect metric (Dice loss → Dice metric) |
| **Chỉ báo cáo pixel accuracy** | Che giấu failure ở lớp hiếm | Luôn báo cáo per-class IoU kèm mIoU |
| **Train/val leakage từ video** | Validation quá tốt, deployment thất bại | Split theo video/scene, không theo frame |

### 7.3 Sinh viên cần nhớ gì?

- **Dense outputs cần spatial reconstruction:** decoder + skip connections là cần thiết — không thể dùng classification backbone thuần túy.
- **Loss và metric phải khớp** với label encoding và chi phí lỗi của bài toán — không có one-size-fits-all loss.
- **Chất lượng data pipeline** chi phối rất nhiều lỗi kiểu "model không học" — kiểm tra pipeline trước khi đổ lỗi cho model.

### 7.4 Bảng tổng hợp toàn bộ bài

| Chủ đề | Khái niệm chính | Công cụ/Code |
|--------|----------------|-------------|
| **Task types** | Semantic/Instance/Panoptic; PQ=SQ×RQ | — |
| **UNet pipeline** | Enc-Bot-Dec-Skip; Z∈R^(B,K,H,W) | `Down`, `Up`, `SegHead` |
| **Skip connections** | Concat > Add cho biên sắc nét | `torch.cat([x_up, skip], dim=1)` |
| **Neck taxonomy** | Identity/Conv/ASPP/Attention/Fusion/Boundary | DeepLabv3+, FPN, SegFormer |
| **Loss: CE** | `-Σ log p_{i,yi}`, ignore_index | `nn.CrossEntropyLoss(ignore_index=255)` |
| **Loss: Dice** | `1 - 2TP/(2TP+FP+FN)`, mạnh với foreground nhỏ | Soft Dice từ probs |
| **Loss: Hybrid** | `λ_CE * L_CE + λ_Dice * L_Dice` | `ce_dice_loss(logits, target)` |
| **Metrics** | mIoU, Dice, HD95, PixAcc | Per-class breakdown! |
| **Data pipeline** | index map PNG, nearest interp, patient-split | `SegDataset.__getitem__` |
| **Advanced** | ASPP, SegFormer, Mask2Former, SAM | segmentation_models_pytorch |

### 7.5 Phép ẩn dụ cuối cùng

> Segmentation là bài toán **structured prediction** — tối ưu cục bộ (từng pixel) nhưng đánh giá toàn cục (IoU của vùng). UNet là mental model CNN cốt lõi cho dense decoding. Engineering (loaders, losses, metrics, protocols) là kỹ năng hạng nhất — model architecture chiếm 20%, pipeline và debugging chiếm 80% thời gian thực tế.

---

*Ghi chú học tập: Bài này kết nối chặt với Bài 13 (Object Detection — instance segmentation cần backbone detection), Bài 12 (Transformer — SegFormer/Mask2Former dùng transformer encoder), và Bài 10 (CNN — UNet encoder dựa trên ResNet/EfficientNet).*
