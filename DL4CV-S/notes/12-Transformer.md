# Bài 12: Attention Mechanism và Transformer

> **Nguồn:** `slides-v1/transformer/Transformer.pdf` (11 trang)  
> **Phong cách:** Ghi chú giảng đường — giải thích từng công thức, thêm lý do & ví dụ số cụ thể.

---

## Mục lục

1. [Động lực — Tại sao cần Attention?](#1-động-lực--tại-sao-cần-attention)
2. [Machine Translation — Encoder-Decoder](#2-machine-translation--encoderdecoder)
3. [Scaled Dot-Product Attention](#3-scaled-dot-product-attention)
4. [Multi-Head Self-Attention](#4-multi-head-self-attention)
5. [Kiến trúc Transformer đầy đủ](#5-kiến-trúc-transformer-đầy-đủ)
6. [Positional Encoding](#6-positional-encoding)
7. [Vision Transformer (ViT)](#7-vision-transformer-vit)
8. [Tóm tắt & So sánh](#8-tóm-tắt--so-sánh)

---

## 1. Động lực — Tại sao cần Attention?

### 1.1 Giới hạn của RNN

Trước Transformer, chuỗi (văn bản, ảnh theo chuỗi token) được xử lý bằng RNN/LSTM:

```
x1 → [RNN] → h1
x2 → [RNN] → h2  (dùng h1)
x3 → [RNN] → h3  (dùng h2)
...
```

**Vấn đề 1 — Vanishing gradient:** h1 phải "sống sót" qua nhiều bước để ảnh hưởng h10. Gradient suy giảm theo cấp số nhân → token đầu câu bị quên.

**Vấn đề 2 — Sequential computation:** Không thể tính h3 nếu chưa có h2 → không song song hóa được trên GPU → chậm.

**Vấn đề 3 — Fixed-size bottleneck:** Encoder RNN nén toàn bộ câu vào một vector $h_N$ → mất thông tin với câu dài.

### 1.2 Ý tưởng Attention — "Tra cứu mềm"

**Phép ẩn dụ:** Hãy tưởng tượng bạn đang tìm kiếm từ trong từ điển:
- **Query (Q):** Câu hỏi bạn đặt ra — "Tôi đang tìm gì?"
- **Key (K):** Nhãn của mỗi mục trong từ điển — "Mục này về gì?"
- **Value (V):** Nội dung thực sự của mỗi mục — "Mục này chứa gì?"

Trong từ điển cứng: hoặc match hoàn toàn, hoặc không. Attention là **tra cứu mềm**: tính mức độ liên quan của Q với mọi K, rồi lấy tổ hợp trọng số của các V tương ứng.

**Kết quả:** Mỗi token được phép "nhìn" tất cả các token khác cùng lúc — không phụ thuộc tuần tự. Đây là lý do Transformer song song hóa tốt và bắt được phụ thuộc tầm xa (long-range dependencies).

---

## 2. Machine Translation — Encoder-Decoder

### 2.1 Bài toán dịch máy

Từ slide page 3-4: Dịch từ tiếng Việt sang tiếng Anh:

```
Input:   "Tôi thích kỹ thuật học sâu"
Output:  "I like deep learning technique"
```

**Phân phối có điều kiện:**

$$P(\text{next-word} \mid \text{Inputs},\; \text{previous-predicted-words})$$

Mô hình cần ước tính xác suất này tại mỗi bước sinh.

### 2.2 Quá trình Encode-Decode

**Bước 1 — Encode:** Đọc toàn bộ câu nguồn và tạo biểu diễn ngữ cảnh.

$$\text{Encode}(\text{``Tôi thích kỹ thuật học sâu''}) \rightarrow \text{context}$$

**Bước 2-8 — Decode từng token** (autoregressive):

```
Decode(<START>)                          → "I"
Decode(<START> I)                        → "like"
Decode(<START> I like)                   → "deep"
Decode(<START> I like deep)              → "learning"
Decode(<START> I like deep learning)     → "technique"
Decode(<START> I like deep learning technique) → <EOS>
→ Stop, trả về: "I like deep learning technique"
```

Mỗi bước decoder nhận: (1) tất cả output trước đó, (2) thông tin từ encoder qua **cross-attention**.

### 2.3 Kiến trúc tổng thể (từ slide page 3)

```
         INPUT SENTENCE                    OUTPUT TOKENS
              ↓                                  ↓
    [Embedding + Positional]          [Embedding + Positional]
              ↓                                  ↓
         Encoder #1                         Decoder #1
         Encoder #2                         Decoder #2
            ...             ←context→         ...
         Encoder #N                         Decoder #N
                                               ↓
                                           [Linear]
                                           [Softmax]
                                               ↓
                                   P(next-word) ∈ ℝ^{vocab-size}
```

Output của Softmax là phân phối xác suất trên toàn bộ vocabulary (thường 30k-50k tokens).

📸 [Cần ảnh: kiến trúc Transformer đầy đủ "Attention is All You Need" — slide page 3] — Tìm: "transformer architecture encoder decoder diagram"

---

## 3. Scaled Dot-Product Attention

### 3.1 Định nghĩa

Đây là building block cơ bản nhất. Cho ba ma trận:

- **Q** (Queries): shape $(N, S, F_Q)$
- **K** (Keys): shape $(N, S, F_K)$ — thường $F_K = F_Q$
- **V** (Values): shape $(N, S, F_V)$

trong đó $N$ = batch size, $S$ = sequence length, $F$ = feature dimension.

**Công thức:**

$$\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{QK^\top}{\sqrt{F_K}}\right) V$$

### 3.2 Giải thích từng bước (từ slide page 6)

```python
# Step 1: Tính score matrix
C = Q @ K.T          # shape: (N, S, S) — "token i quan tâm token j bao nhiêu?"

# Step 2: Scale
C = C / sqrt(F_K)    # chia căn bậc hai của chiều K

# Step 3: Optional Masking
C = C + Mask         # Mask chứa 0 (giữ) hoặc -inf (che)

# Step 4: Softmax theo từng hàng
C = Softmax(C, dim=-1)  # mỗi hàng sum = 1 → trọng số attention

# Step 5: Weighted sum of Values
F = C @ V            # shape: (N, S, F_V) — output attention
```

### 3.3 Tại sao chia $\sqrt{F_K}$?

**Vấn đề:** Với $F_K$ lớn (ví dụ 512), tích vô hướng $QK^\top$ có giá trị lớn → các giá trị sau Softmax rất lệch (gần 0 hoặc 1) → **gradient nhỏ** (Softmax bão hòa).

**Ví dụ số:**

| $F_K$ | Giá trị QK điển hình | Softmax output | Gradient |
|--------|----------------------|----------------|----------|
| 8 | ~2.8 | phân phối đều | ổn định |
| 512 | ~22.6 | gần one-hot | rất nhỏ |

Chia $\sqrt{F_K}$ đưa về phương sai ~1 → Softmax hoạt động tốt.

### 3.4 Mask là gì?

**Padding Mask:** Khi sequence trong batch có độ dài khác nhau, padding tokens không nên được attend. Gán $-\infty$ cho vị trí padding → Softmax($-\infty$) = 0 → không đóng góp.

**Causal Mask (trong Decoder):** Token ở vị trí $i$ chỉ được nhìn token từ vị trí $1$ đến $i$ (không nhìn tương lai). Che tam giác trên của ma trận attention bằng $-\infty$.

```
Causal mask cho S=4:
     pos1  pos2  pos3  pos4
pos1 [  0   -inf  -inf  -inf ]
pos2 [  0    0   -inf  -inf ]
pos3 [  0    0    0   -inf ]
pos4 [  0    0    0    0   ]
```

### 3.5 Ví dụ số — Self-Attention đơn giản

Câu: ["Tôi", "thích", "học"]. $F_K = F_V = 2$ (nhỏ cho dễ tính).

```
Q = [[1,0], [0,1], [1,1]]  # query của 3 token
K = [[1,0], [0,1], [1,1]]  # key của 3 token (Self-Attention: Q=K=V nguồn)
V = [[1,2], [3,0], [2,1]]  # value của 3 token

# Step 1: score = Q @ K^T
C = [[1,0,1],    # "Tôi" → [Q1·K1, Q1·K2, Q1·K3] = [1,0,1]
     [0,1,1],    # "thích"
     [1,1,2]]    # "học"

# Step 2: scale (sqrt(2) ≈ 1.41)
C_scaled = [[0.71, 0, 0.71],
            [0, 0.71, 0.71],
            [0.71, 0.71, 1.41]]

# Step 3: Softmax theo hàng
# "học" (hàng 3) attend nhiều nhất đến chính nó (score 1.41 cao nhất)
```

---

## 4. Multi-Head Self-Attention

### 4.1 Tại sao Multi-Head?

**Vấn đề của Single-Head:** Attention chỉ học được **một loại quan hệ** giữa các token — ví dụ chỉ quan hệ cú pháp (subject-verb), hoặc chỉ quan hệ ngữ nghĩa (co-reference).

**Giải pháp:** Chạy $H$ attention heads song song với các projection khác nhau (Q, K, V được chiếu vào không gian con khác nhau) → mỗi head học một loại quan hệ khác nhau → concat kết quả.

**Phép ẩn dụ:** Như nhìn câu qua nhiều "kính" khác nhau — một kính thấy cấu trúc cú pháp, kính khác thấy co-reference, kính khác thấy ngữ nghĩa.

### 4.2 Công thức và Shape (từ slide page 7)

**Input:** $(N, S, F_{\text{in}})$

**Với mỗi head $h$:**

$$Q_h = X W_h^Q \in \mathbb{R}^{N \times S \times F_Q}, \quad K_h = X W_h^K \in \mathbb{R}^{N \times S \times F_K}, \quad V_h = X W_h^V \in \mathbb{R}^{N \times S \times F_V}$$

**Theo dõi shapes (từ slide page 7):**

```
Input X:         (N, S, F_in)
                    ↓
Linear Q/K/V:   (N, S, heads·F_Q) / (N, S, heads·F_K) / (N, S, heads·F_V)
                    ↓ Reshape (split heads)
Per-head Q/K/V: (N, heads, S, F_Q) / (N, heads, S, F_K) / (N, heads, S, F_V)
                    ↓ Scaled Dot-Product Attention (per head)
Per-head output:(N, heads, S, F_V)
                    ↓ Concat (= Reshape + transpose)
Concatenated:   (N, S, heads·F_V)
                    ↓ Linear projection out
Output:         (N, S, F_out)
```

**Công thức đầy đủ:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$

$$\text{head}_h = \text{Attention}(QW_h^Q,\; KW_h^K,\; VW_h^V)$$

### 4.3 Reshape và Concat thực chất là gì?

Slide ghi chú: *"Reshape: actually = reshape, then transpose"* và *"Concat: actually = transpose, then reshape"*.

Thực ra hai thao tác này chỉ là **xem lại cùng một tensor với stride khác** — không copy dữ liệu. Đây là lý do implementation rất hiệu quả trong PyTorch.

```python
# Split heads
# (N, S, H*F_Q) → (N, S, H, F_Q) → (N, H, S, F_Q)
Q = Q.reshape(N, S, H, F_Q).transpose(1, 2)

# Concat heads
# (N, H, S, F_V) → (N, S, H, F_V) → (N, S, H*F_V)
out = out.transpose(1, 2).reshape(N, S, H * F_V)
```

### 4.4 Code Multi-Head Self-Attention (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # F_Q = F_K = F_V = d_model / H

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        N, S, _ = x.shape
        H, D = self.num_heads, self.d_k

        # Project and split into heads: (N, S, d_model) → (N, H, S, D)
        Q = self.W_q(x).reshape(N, S, H, D).transpose(1, 2)
        K = self.W_k(x).reshape(N, S, H, D).transpose(1, 2)
        V = self.W_v(x).reshape(N, S, H, D).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(D)  # (N, H, S, S)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)                 # (N, H, S, S)
        out = attn @ V                                   # (N, H, S, D)

        # Concat heads: (N, H, S, D) → (N, S, d_model)
        out = out.transpose(1, 2).reshape(N, S, self.d_model)
        return self.W_o(out)
```

---

## 5. Kiến trúc Transformer Đầy đủ

### 5.1 Encoder Block (từ slide page 5)

```
Input: (N, S, F_in)
    │
    ├─ [Multi-Head Self-Attention] ──────────────────────┐
    │   Q = K = V = Input (self-attention)               │ (skip)
    │   Output: (N, S, F_out)                            │
    │                                                    │
    └─ Add & LayerNorm: LayerNorm(Input + Attention_out) ┘
    │
    ├─ [Feed-Forward Network]
    │   Linear(F_out → 4·F_out) → ReLU → Linear(4·F_out → F_out)
    │
    └─ Add & LayerNorm: LayerNorm(prev + FFN_out)
    │
    Output: (N, S, F_out)
```

**Self-Attention trong Encoder:** Q, K, V đều đến từ **cùng một input** — mỗi token được phép nhìn tất cả token khác trong cùng câu (không có causal mask).

**Tại sao LayerNorm sau Add?**

**Residual connection (+):** Như ResNet — giúp gradient chạy thẳng, tránh vanishing gradient. Training sâu (N encoder stacked) mà không bị suy giảm gradient.

**LayerNorm:** Normalize theo chiều feature (khác BatchNorm normalize theo chiều batch) — phù hợp với sequence vì độ dài khác nhau giữa các sample.

### 5.2 Decoder Block (từ slide page 8)

```
Target Input (shifted right):
    │
    ├─ [Masked Multi-Head Self-Attention]  ← causal mask: không nhìn tương lai
    │
    └─ Add & LayerNorm
    │
    ├─ [Encoder-Decoder Cross-Attention]
    │   Q: từ Decoder output
    │   K, V: từ ENCODER output (last layer)  ← "nhìn vào câu nguồn"
    │
    └─ Add & LayerNorm
    │
    ├─ [Feed-Forward Network]
    │
    └─ Add & LayerNorm
    │
    Output: (N, S_target, F_out)
```

**Cross-Attention là điểm mấu chốt:**
- Q đến từ decoder: "Token tôi đang sinh cần thông tin gì từ câu nguồn?"
- K, V đến từ encoder: "Câu nguồn cung cấp thông tin gì?"
- Decoder được phép "nhìn" vào toàn bộ encoder output tại mỗi bước sinh.

**Phép ẩn dụ:** Khi dịch từ "học" sang "learning", decoder (đang ở vị trí "learning") gửi query "tôi đang tìm từ nghĩa là 'học'" → cross-attention tìm ra encoder token "học" có attention weight cao nhất → lấy value của nó để decode.

### 5.3 Feed-Forward Network (FFN)

Sau mỗi attention block là một FFN nhỏ áp dụng **cho từng token độc lập**:

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

- $W_1$: $(F_{\text{model}}, 4 F_{\text{model}})$ — thường giãn rộng 4 lần (ví dụ 512 → 2048).
- $W_2$: $(4 F_{\text{model}}, F_{\text{model}})$ — thu hẹp lại.

**Vai trò:** Sau attention (tổng hợp ngữ cảnh), FFN áp dụng phép biến đổi phi tuyến **cá nhân** cho từng token — giống như "xử lý thông tin đã thu thập".

### 5.4 Số lượng tham số

Với $d_{\text{model}}$ và $H$ heads (mỗi head $d_k = d_v = d_{\text{model}}/H$):

| Thành phần | Parameters |
|------------|-----------|
| Multi-Head Attention | $4 d_{\text{model}}^2$ ($W_Q, W_K, W_V, W_O$) |
| FFN (4x expansion) | $2 \times 4 d_{\text{model}}^2 = 8 d_{\text{model}}^2$ |
| **Mỗi encoder block** | $\approx 12 d_{\text{model}}^2$ |
| GPT-3 ($d=12288$, 96 layers) | ~175 tỷ tham số |

---

## 6. Positional Encoding

### 6.1 Vấn đề: Transformer không có thứ tự

Scaled Dot-Product Attention là **permutation equivariant** — hoán đổi vị trí các token không làm thay đổi kết quả (chỉ hoán đổi output tương ứng). Nghĩa là Transformer không "biết" token nào đứng trước token nào!

**Ví dụ:** "Chó cắn người" vs "Người cắn chó" — Transformer cần biết thứ tự mới phân biệt được.

### 6.2 Positional Encoding (từ bài báo gốc)

Cộng trực tiếp thông tin vị trí vào embedding:

$$\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

$$\text{PE}(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

trong đó $\text{pos}$ là vị trí token (0, 1, 2, ...), $i$ là chiều embedding, $d$ là tổng số chiều.

**Tại sao sin/cos?**
- Mỗi vị trí có một "fingerprint" độc nhất.
- Có tính chất: $PE(\text{pos}+k)$ là hàm tuyến tính của $PE(\text{pos})$ → mô hình học được offset tương đối.
- Không cần học thêm tham số (fixed).

**Trong thực tế hiện đại:** Dùng **Learnable Positional Embedding** — cũng thêm vector vị trí, nhưng học từ dữ liệu (ViT, BERT, GPT đều dùng).

```python
# Learnable positional embedding (ViT style)
self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

# Thêm vào token embeddings:
x = token_embed + self.pos_embed[:, :S, :]
```

📸 [Cần ảnh: biểu đồ positional encoding sin/cos — "positional encoding transformer visualization"]

---

## 7. Vision Transformer (ViT)

### 7.1 Từ NLP sang Computer Vision

Transformer được thiết kế cho chuỗi token (văn bản). Ảnh là lưới 2D — làm sao đưa vào Transformer?

**Ý tưởng của ViT** (Dosovitskiy et al., 2020): **Chia ảnh thành các patch**, mỗi patch là một "token".

### 7.2 Pipeline ViT

```
Ảnh gốc: (H, W, C) = (224, 224, 3)
    │
    ├─ Chia thành P×P patches (thường P=16): 
    │  → (H/P · W/P) patches = 14×14 = 196 patches
    │
    ├─ Flatten mỗi patch: (P²·C) = (16²·3) = 768 chiều
    │
    ├─ Linear projection (patch embedding):
    │  → mỗi patch: vector 768 chiều
    │
    ├─ Thêm [CLS] token ở đầu (như BERT): 197 tokens tổng
    │
    ├─ Thêm Positional Embedding (learnable 2D)
    │
    └─ Stack N Transformer Encoder blocks
    │
    ├─ Lấy [CLS] token output
    │
    └─ [MLP Head] → class probabilities
```

**[CLS] token** là token đặc biệt không tương ứng patch nào. Sau N layers, [CLS] đã "hỏi thăm" (attend đến) tất cả patches khác → biểu diễn toàn cục của ảnh → dùng để phân loại.

📸 [Cần ảnh: ViT pipeline — chia patch → token → Transformer — slide page 9] — Tìm: "Vision Transformer ViT architecture"

### 7.3 Tại sao ViT quan trọng?

| | CNN | ViT |
|-|-----|-----|
| **Inductive bias** | Translation equivariance, locality | Không có bias — học từ dữ liệu |
| **Receptive field** | Tăng dần qua layers | **Toàn cục** từ layer đầu (self-attention) |
| **Dữ liệu cần** | Tốt với ít dữ liệu | Cần nhiều dữ liệu hơn (hoặc pre-training) |
| **Scalability** | Tốt | **Tốt hơn** khi scale (ViT-G, 22B params) |
| **Sota** | Trước 2020 | Hiện tại dẫn đầu nhiều benchmark |

**Kết luận quan trọng:** ViT không có "bias" rằng pixel gần nhau liên quan hơn pixel xa (như CNN với local receptive field) — nó học hoàn toàn từ dữ liệu. Với đủ dữ liệu (ImageNet-21k, JFT-300M), ViT vượt CNN.

### 7.4 Biến thể và ứng dụng ViT

| Biến thể | Ý tưởng chính | Ứng dụng |
|----------|--------------|----------|
| **DeiT** | Knowledge distillation để train ViT với ít data | Classification |
| **Swin Transformer** | Attention trong cửa sổ dịch chuyển (shifted windows) | Detection, Segmentation |
| **MAE** | Masked Autoencoder — che 75% patches và reconstruct | Self-supervised pre-training |
| **ViT-SAM** | Segment Anything Model của Meta | Image Segmentation |
| **CLIP** | Contrastive Language-Image Pre-Training | Zero-shot classification |

---

## 8. Tóm tắt & So sánh

### 8.1 Các thành phần cốt lõi

| Thành phần | Công thức / Ý chính |
|-----------|---------------------|
| **Scaled Dot-Product Attention** | $\text{Softmax}(QK^\top/\sqrt{F_K}) \cdot V$ |
| **Tại sao chia** $\sqrt{F_K}$ | Tránh Softmax bão hòa khi $F_K$ lớn |
| **Multi-Head Attention** | $H$ heads học $H$ loại quan hệ khác nhau |
| **Self-Attention** | Q = K = V = cùng input |
| **Cross-Attention** | Q từ decoder, K/V từ encoder |
| **Causal Mask** | Chặn nhìn tương lai trong decoder |
| **Positional Encoding** | Cộng sin/cos hoặc learnable vector vào embedding |
| **LayerNorm** | Normalize theo chiều feature sau Add |
| **FFN** | Linear → ReLU → Linear cho từng token |
| **ViT Patch** | Ảnh $H\times W\times C$ → $(HW/P^2)$ tokens × $P^2C$ chiều |

### 8.2 Quy trình tính Attention — Tóm tắt

```
Step 1: C = Q @ K.T          → shape (N, S, S): score ma trận
Step 2: C = C / sqrt(F_K)    → scale để tránh bão hòa
Step 3: C = C + Mask          → optional: padding/causal mask  
Step 4: C = Softmax(C)        → trọng số attention, mỗi hàng sum=1
Step 5: F = C @ V             → tổ hợp trọng số của Values
```

### 8.3 Phức tạp tính toán

**Attention:** $O(S^2 \cdot d)$ — bình phương theo độ dài chuỗi $S$. Với $S = 10000$ (tài liệu dài), điều này rất tốn kém → các nghiên cứu như Longformer, FlashAttention giải quyết vấn đề này.

**ViT:** $S = (H/P)(W/P)$ patches — với $P=16$ và ảnh $224\times224$: $S = 196$, rất nhỏ → Attention hiệu quả.

### 8.4 Transformer vs CNN

| | CNN | Transformer/ViT |
|-|-----|----------------|
| **Phụ thuộc không gian** | Local (receptive field) | Global (mọi token thấy nhau) |
| **Xử lý chuỗi** | Cần 1D conv | Native |
| **Computational cost** | $O(S \cdot k^2 \cdot d)$ | $O(S^2 \cdot d)$ |
| **Inductive bias** | Translation equivariance | Không có |
| **Small data** | Tốt hơn | Kém hơn (cần pre-training) |
| **Scale** | Khó scale lên | Rất tốt (ViT-G, GPT-4) |

### 8.5 Code: PyTorch built-in

```python
import torch
import torch.nn as nn

# Built-in Multi-Head Attention
mha = nn.MultiheadAttention(
    embed_dim=512,   # d_model
    num_heads=8,     # H heads → d_k = 64
    dropout=0.1,
    batch_first=True # input shape: (N, S, d_model)
)

x = torch.randn(32, 10, 512)  # batch=32, seq=10, dim=512
out, attn_weights = mha(x, x, x)  # Q=K=V=x → self-attention
# out shape: (32, 10, 512)
# attn_weights shape: (32, 10, 10)

# Encoder Layer (Attention + FFN + LayerNorm)
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048,
    dropout=0.1, batch_first=True
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
out = encoder(x)  # (32, 10, 512)
```

---

📸 **Ảnh slide quan trọng cần bổ sung:**

| Slide/Ref | Nội dung | Nguồn gợi ý |
|-----------|----------|-------------|
| Page 3 (EN) | Transformer encoder-decoder architecture | "attention is all you need architecture" |
| Page 5 (EN) | Encoder block: MHA + Add&Norm | "transformer encoder block diagram" |
| Page 6-7 (EN) | Multi-Head Attention với shapes | "multi-head attention diagram shapes" |
| Page 8 (EN) | Decoder block với cross-attention | "transformer decoder cross attention" |
| Page 9 (EN) | ViT: patch → token | "vision transformer ViT architecture patch" |
| Extra | Positional encoding sin/cos heatmap | "positional encoding visualization" |
| Extra | Attention map visualization trên ảnh | "ViT attention map visualization" |
