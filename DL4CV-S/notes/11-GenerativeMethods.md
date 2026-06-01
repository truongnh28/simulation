# Bài 11: Phương pháp Sinh (Generative Methods)

> **Nguồn:** `slides-v1/generative/GenerativeMethods.pdf` (57 trang) + `GenerativeMethods.vne.pdf` (64 trang)  
> **Phong cách:** Ghi chú giảng đường — giải thích từng bullet, thêm lý do & ví dụ số cụ thể.

---

## Mục lục

1. [Sinh vs. Phân biệt (Generative vs. Discriminative)](#1-sinh-vs-phân-biệt)
2. [Mô hình sinh cho ảnh — Bức tranh tổng quan](#2-mô-hình-sinh-cho-ảnh)
3. [Variational Autoencoder (VAE)](#3-variational-autoencoder-vae)
   - 3.1 Quan niệm (Trực giác)
   - 3.2 Góc nhìn tính toán
   - 3.3 Góc nhìn toán học — ELBO
4. [Generative Adversarial Networks (GAN)](#4-generative-adversarial-networks-gan)
   - 4.1 Quan niệm
   - 4.2 Góc nhìn tính toán
   - 4.3 Mục tiêu toán học
5. [Denoising Diffusion Probabilistic Models (DDPM)](#5-denoising-diffusion-ddpm)
   - 5.1 Quan niệm
   - 5.2 Góc nhìn tính toán
   - 5.3 Toán học (tùy chọn)
6. [Sinh văn bản — LM kiểu GPT](#6-sinh-văn-bản)
7. [Tổng hợp tác vụ và Ứng dụng](#7-tổng-hợp-tác-vụ-và-ứng-dụng)
8. [So sánh ba họ mô hình ảnh](#8-so-sánh-vae--gan--ddpm)
9. [Tóm tắt](#9-tóm-tắt)

---

## 1. Sinh vs. Phân biệt

### 1.1 Hai trường phái học máy

Trước giờ, cả khóa tập trung vào **mô hình phân biệt (discriminative)**: học $p(y \mid x)$ hoặc một ranh giới quyết định (decision boundary). Ví dụ: phân loại ảnh chó/mèo — mô hình chỉ cần biết "đây là chó hay mèo", không cần hiểu ảnh chó trông như thế nào.

**Mô hình sinh (generative)** đi xa hơn: học **phân phối dữ liệu** $p(x)$ hoặc $p(x, y)$ — tức là học "ảnh chó thực tế phân phối ra sao". Đây là nhiệm vụ khó hơn nhiều.

| Tiêu chí | Discriminative | Generative |
|----------|---------------|------------|
| Học gì | $p(y \mid x)$ hoặc ranh giới | $p(x)$ hoặc $p(x, y)$ |
| Dùng cho | Phân loại, hồi quy | Sinh mẫu, tăng cường dữ liệu, nén |
| Ví dụ | CNN phân loại | VAE, GAN, DDPM |

### 1.2 Tại sao cần $p(x)$?

Biết $p(x)$ mở ra nhiều ứng dụng mà discriminative model không làm được:

- **Lấy mẫu (Sampling):** vẽ $\tilde{x} \sim p(x)$ — sinh ảnh mới, câu văn mới, bản ghi mới trông giống thực tế. Ứng dụng: tạo nội dung, nghệ thuật AI.
- **Tăng cường dữ liệu (Data augmentation):** tổng hợp thêm ví dụ cho lớp hiếm (imbalanced dataset). Thay vì oversampling đơn thuần, ta sinh ảnh mới thực sự đa dạng.
- **Điền khuyết & khử nhiễu (Imputation & denoising):** nếu biết $p(x)$, có thể suy ra phần bị thiếu hay che khuất — MRI reconstruction, ảnh bị nhiễu.
- **Nén / học biểu diễn:** học **mã tiềm ẩn (latent code)** $z$ gọn hơn $x$ nhiều lần — như nén ảnh thông minh biết ngữ nghĩa.

### 1.3 Góc nhìn xác suất

Với cặp $(x, y)$:

$$p(y \mid x) = \frac{p(x, y)}{p(x)}, \quad p(x \mid y) = \frac{p(x, y)}{p(y)}$$

**Phân loại qua Bayes:**

$$p(y \mid x) \propto p(x \mid y)\, p(y)$$

Đây là cơ sở của Naive Bayes classifier — mô hình sinh có thể được dùng để phân loại. Nhưng trong DL hiện đại, mô hình sinh thường dùng để **sinh dữ liệu** và học biểu diễn là chính.

---

## 2. Mô hình Sinh cho Ảnh

### 2.1 Bài toán hình thức

**Ảnh** $x \in \mathbb{R}^{H \times W \times C}$ là vector chiều rất cao. Ví dụ ảnh màu $128 \times 128 \times 3 = 49{,}152$ chiều. Không gian các ảnh có thể tồn tại là $\mathbb{R}^{49152}$ — nhưng ảnh thực tế chỉ chiếm một **đa tạp (manifold) mỏng** trong không gian đó.

**Mục tiêu:** học $p(x)$ sao cho mẫu $\tilde{x} \sim p$:
- Trông thực với người (realistic).
- Bao phủ các mode chính của phân phối — không chỉ nhớ vẹt training set.
- Có thể điều khiển (conditional): sinh ảnh theo nhãn lớp, theo prompt văn bản.

**Các bộ dữ liệu chuẩn:** MNIST (28×28 grayscale, 70k ảnh), CIFAR-10 (32×32 màu), CelebA (khuôn mặt), ImageNet (1000 lớp, rất khó).

### 2.2 Bức tranh tổng quan — 5 họ mô hình

```
Ảnh x ∈ ℝ^{HWC}
│
├─ Autoencoder (AE): x →[Encoder]→ z →[Decoder]→ x̂
│   Tất định, nén không có khả năng sinh
│
├─ VAE: x →[Enc]→ q_φ(z|x) →[sample]→ z →[Dec]→ x̂
│   Xác suất, ELBO, sinh bằng cách sample z ∼ p(z)
│
├─ GAN: z →[G]→ x̃, phân biệt bởi [D] với x_real
│   Đối kháng, sinh một bước, thường sắc nét hơn
│
├─ Diffusion (DDPM): x_0 →[thêm nhiễu T bước]→ x_T ∼ N(0,I)
│   học đảo ngược: x_T →...→ x_0
│
└─ Autoregressive: p(x) = ∏_i p(x_i | x_{<i})
    PixelCNN, PixelRNN — pixel-by-pixel
```

📸 [Cần ảnh: so sánh ảnh sinh từ VAE vs GAN vs Diffusion trên MNIST — slide page 55-56] — Tìm trên Google Images: "VAE GAN Diffusion MNIST comparison"

---

## 3. Variational Autoencoder (VAE)

### 3.1 Quan niệm — Trực giác

#### Autoencoder thông thường (AE) — điểm xuất phát

**Phép ẩn dụ:** AE giống máy nén ảnh — encoder "ép" ảnh vào code nhỏ, decoder "phục hồi" lại. Nhưng code này **tất định**: cùng một ảnh luôn cho cùng một code. Vì vậy, nếu ta vẽ ngẫu nhiên một code $z$ rồi đưa qua decoder, kết quả thường vô nghĩa.

$$z = f(x), \quad \hat{x} = g(z)$$

Nhược điểm: **Không có mô hình sinh rõ ràng** — không biết vùng nào trong không gian $z$ hợp lệ.

#### VAE — thêm tính xác suất

VAE giải quyết bằng cách làm encoder **xác suất**:
- Thay vì trả về một điểm $z$, encoder trả về một **phân phối** $q_\phi(z \mid x)$.
- Thường chọn $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x),\, \sigma^2_\phi(x) I)$ — encoder xuất ra $(\mu, \log\sigma^2)$.
- Decoder $p_\theta(x \mid z)$ học cách từ $z$ sinh ra $x$.

**Đường sinh (inference time):**

$$z \sim p(z) = \mathcal{N}(0, I) \longrightarrow \hat{x} \sim p_\theta(x \mid z)$$

Không cần encoder khi sinh — chỉ cần sample từ prior đơn giản rồi qua decoder.

| | AE | VAE |
|-|----|----|
| Encoder ra | Điểm $z$ tất định | Phân phối $q_\phi(z\|x)$ |
| Sinh mới | Không thể | Có: sample $z \sim p(z)$ |
| Loss | Reconstruction only | Reconstruction + KL |

#### Thành phần VAE

- **Prior** $p(z) = \mathcal{N}(0, I)$: phân phối tiên nghiệm đơn giản — nguồn để lấy mẫu khi sinh.
- **Decoder** $p_\theta(x \mid z)$: mô hình sinh, tham số $\theta$.
- **Encoder** $q_\phi(z \mid x)$: xấp xỉ hậu nghiệm, tham số $\phi$. **Chỉ dùng khi training** — tính KL và sample $z$ để reconstruction.

📸 [Cần ảnh: sơ đồ AE vs VAE — latent space structure — slide page 13-15] — Tìm: "VAE vs AE latent space"

### 3.2 Góc nhìn Tính toán

#### Đồ thị tính toán (training)

```
x ──[Encoder φ]──► (μ, log σ²)
                        │
              ε ~ N(0,I)│  (noise injected)
                        ▼
              z = μ + σ ⊙ ε   ◄── Reparameterization trick
                        │
              [Decoder θ]
                        │
                        ▼
                        x̂

Loss = L_rec(x, x̂)  +  β · KL(q_φ(z|x) ‖ p(z))
         (BCE or MSE)      (match prior; β=1 vanilla VAE)
```

**Tại sao cần Reparameterization Trick?**

Vấn đề: $z \sim q_\phi(z \mid x)$ là **phép lấy mẫu ngẫu nhiên** — không khả vi (not differentiable), gradient không đi qua được đến $\phi$.

**Giải pháp:** viết lại:

$$\varepsilon \sim \mathcal{N}(0, I), \quad z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon$$

Bây giờ $z$ là **hàm tất định của $(\mu, \sigma)$** và $\varepsilon$ (noise cố định). Gradient từ decoder qua $z$ đến $(\mu, \sigma)$ chạy được — gradient không "dừng" ở phép sample nữa.

#### Ba chế độ sử dụng VAE sau khi train

| Chế độ | Cách làm | Dùng encoder? |
|--------|----------|---------------|
| **Reconstruction** | $x \to \text{Enc} \to z{=}\mu \to \text{Dec} \to \hat{x}$ | Có |
| **Generation** | $z \sim p(z) \to \text{Dec} \to \hat{x}$ | **Không** |
| **Encode** | $x \to \text{Enc} \to z$ (visualization, features) | Có |

#### Tại sao mẫu VAE thường "mờ" (soft)?

- ELBO được tối ưu khuyến khích decoder "trung hòa" — nếu một số chi tiết không chắc chắn, decoder có xu hướng lấy trung bình của các khả năng → blur.
- Đây là **giới hạn cơ bản** của mô hình likelihood, không phải lỗi do train chưa đủ — thêm epoch chỉ giúp một phần.
- GAN và Diffusion giải quyết vấn đề này theo cách khác.

📸 [Cần ảnh: ảnh tái tạo VAE mờ vs GAN sắc — slide page 22] — Tìm: "VAE blurry samples vs GAN"

### 3.3 Góc nhìn Toán học — ELBO

#### Bước 0: Hai vấn đề không tính được trực tiếp

Mô hình sinh: prior $p(z)$, decoder $p_\theta(x \mid z)$.

**Vấn đề 1 — Marginal (evidence):**

$$p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz$$

Tích phân này **không có dạng đóng** với decoder linh hoạt (neural network). Không tính được.

**Vấn đề 2 — True posterior:**

$$p_\theta(z \mid x) = \frac{p_\theta(x \mid z)\, p(z)}{p_\theta(x)}$$

Mẫu số $p_\theta(x)$ không tính được → hậu nghiệm thực cũng không tính được.

**Ý tưởng biến phân:** thay vì tính $p_\theta(z \mid x)$ thật, ta dùng **hậu nghiệm xấp xỉ** $q_\phi(z \mid x)$ (encoder) và lập luận bằng KL divergence.

#### Bước 1: Đẳng thức trọng số (importance identity)

$$p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz = \mathbb{E}_{z \sim q_\phi(\cdot \mid x)}\!\left[\frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)}\right]$$

Lấy logarithm:

$$\log p_\theta(x) = \log \mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)}\right]$$

**Vẫn khó:** log nằm ngoài kỳ vọng.

#### Bước 2: Jensen's Inequality → ELBO

Hàm $\log$ là **hàm lõm (concave)**, Jensen cho: $\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]$.

Áp dụng:

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi}\!\left[\log p_\theta(x \mid z) + \log p(z) - \log q_\phi(z \mid x)\right] =: \mathcal{L}_{\text{ELBO}}(x)$$

Gộp lại:

$$\boxed{\mathcal{L}_{\text{ELBO}}(x) = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)]}_{\text{(A) Reconstruction}} - \underbrace{KL\!\left(q_\phi(z \mid x) \,\|\, p(z)\right)}_{\text{(B) Regularization}}}$$

#### Bước 3: Đọc ELBO từng số hạng

**(A) Reconstruction term:** "Decoder phải làm $x$ có xác suất cao từ các $z$ lấy từ encoder." — Trong code: dùng BCE (decoder Bernoulli) hoặc MSE (decoder Gaussian).

**(B) KL term:** "Encoder không được tạo ra các code $z$ quá xa prior $p(z) = \mathcal{N}(0,I)$." — Nếu code $z$ của mỗi ảnh nằm khắp nơi trong không gian, sample $z \sim \mathcal{N}(0,I)$ sẽ rơi vào vùng vô nghĩa → KL "kéo" encoder về gần prior.

**Công thức đóng cho KL (Gaussian-Gaussian):**

$$KL\!\left(\mathcal{N}(\mu,\sigma^2) \,\|\, \mathcal{N}(0, I)\right) = \frac{1}{2}\sum_j\!\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$$

Đây là lý do $\beta$-VAE tồn tại: $\beta > 1$ tăng trọng số KL → không gian tiềm ẩn có cấu trúc rõ hơn, nhưng reconstruction kém hơn.

#### Bước 4: Tại sao ELBO là cận dưới

Có thể chứng minh:

$$\log p_\theta(x) = \mathcal{L}_{\text{ELBO}}(x) + \underbrace{KL\!\left(q_\phi(z \mid x) \,\|\, p_\theta(z \mid x)\right)}_{\geq 0}$$

Vì KL ≥ 0 luôn, nên ELBO ≤ log $p_\theta(x)$ — ELBO là **cận dưới của log-evidence**.

**Ý chính:** Cực đại ELBO = vừa tối ưu cận dưới khả tính, vừa ép $q_\phi$ bám sát hậu nghiệm thực $p_\theta(z \mid x)$.

#### Bước 5: Từ ELBO thành loss huấn luyện thực tế

$$J(\theta, \phi; x) \approx -\log p_\theta(x \mid z) + KL\!\left(q_\phi(z \mid x) \,\|\, p(z)\right)$$

trong đó $z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$.

**β-VAE:**

$$J_\beta \approx -\log p_\theta(x \mid z) + \beta \cdot KL\!\left(q_\phi(z \mid x) \,\|\, p(z)\right)$$

#### Lộ trình ELBO — tóm tắt 5 bước

| Bước | Nội dung |
|------|----------|
| 0 | Nhận ra $p_\theta(x)$ và $p_\theta(z\|x)$ không tính được trực tiếp |
| 1 | Viết $p_\theta(x)$ thành kỳ vọng dưới $q_\phi$ (importance identity) |
| 2 | Dùng Jensen → ELBO (cận dưới khả tính) |
| 3 | Tách ELBO = Reconstruction − KL |
| 4 | Reparameterization để gradient chạy; tối ưu SGD/Adam |

> **Nhớ một dòng:** Khớp dữ liệu trong không gian quan sát, điều chuẩn cấu trúc trong không gian tiềm ẩn.

#### Code VAE (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=16):
        super().__init__()
        # Encoder: x → (μ, log σ²)
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder: z → x̂
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # reparameterization trick

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl
```

---

## 4. Generative Adversarial Networks (GAN)

### 4.1 Quan niệm

#### Ý tưởng lớn

GAN hoàn toàn khác VAE về triết lý: **không cần encoder, không cần tính likelihood** — thay vào đó dùng một mạng thứ hai để "chấm điểm".

$$z \sim p(z) = \mathcal{N}(0, I), \quad \tilde{x} = G_\theta(z)$$

**Generator** $G_\theta$: biến đổi noise thành ảnh (synthesis path). Chỉ chạy một forward pass — **rất nhanh khi inference**.

**Tín hiệu training** đến từ **Discriminator** $D_\phi$: một mạng phân loại real/fake. D không phải objective cố định — nó được học cùng G.

**Phép ẩn dụ tuyệt vời:** G như tên làm bạc giả, D như cảnh sát phát hiện bạc giả. Ban đầu G kém, D dễ phát hiện. G học theo phản hồi của D, dần tạo được bạc giả ngày càng khó phân biệt. D cũng phải liên tục nâng cấp. Kết quả cuối cùng: G tạo ra "tiền" không thể phân biệt với thật.

#### Hai mạng, một trò chơi

1. **Generator** $G_\theta$: $z \mapsto \tilde{x}$ — đường tổng hợp.
2. **Discriminator** $D_\phi$: cho điểm "độ thật" cho ảnh đầu vào (0 = giả, 1 = thật).
3. **Cạnh tranh:** D cố phân biệt thật/giả; G cố đánh lừa D.

Khi D tốt hơn → gradient phân biệt rõ hơn → G nhận phản hồi chất lượng cao hơn → G cải thiện.

📸 [Cần ảnh: sơ đồ luồng GAN training (noise → G → fake → D ← real) — slide page 34] — Tìm: "GAN training diagram"

### 4.2 Góc nhìn Tính toán

#### Dùng mô hình sau khi train

- **Sinh:** $z \sim p(z)$, ra $\tilde{x} = G_\theta(z)$ — chỉ cần G, một forward pass.
- **Không encoder:** GAN vanilla không có đường $x \mapsto z$. (Bi-GAN và các biến thể mới có thêm encoder riêng.)
- **Không cần D lúc inference:** D chỉ là "giáo viên" trong quá trình train — sau khi train xong, chỉ dùng G.

#### Code thực sự tối ưu gì

**Train D:**

```python
# D cố gán xác suất cao cho real, thấp cho fake
loss_D_real = F.binary_cross_entropy_with_logits(D(x_real), torch.ones_like(...))
loss_D_fake = F.binary_cross_entropy_with_logits(D(G(z).detach()), torch.zeros_like(...))
loss_D = loss_D_real + loss_D_fake
```

**Train G (non-saturating):**

```python
# G cố làm D nghĩ fake là real
loss_G = F.binary_cross_entropy_with_logits(D(G(z)), torch.ones_like(...))
# Đây là maximize E[log D(G(z))] thay vì minimize E[log(1-D(G(z)))]
```

**Label smoothing:** thay target 1.0 bằng 0.9 cho real → D không quá tự tin, gradient ổn hơn.

**Cân bằng D vs G:**
- Nếu D quá mạnh → gradient cho G gần bằng 0 (G không học được gì).
- Nếu G quá mạnh → D không phân biệt được, G không nhận phản hồi có ích.
- Thực hành: thường train D nhiều bước ($k$ steps) rồi mới train G 1 bước.

### 4.3 Mục tiêu Toán học

#### Bước 0: Minimax objective (bản gốc)

$$\min_\theta \max_\phi \;\mathbb{E}_{x \sim p_{\text{data}}}[\log D_\phi(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D_\phi(G_\theta(z)))]$$

**Giải thích từng phần:**
- $\mathbb{E}[\log D(x)]$: D muốn cho xác suất cao với real data.
- $\mathbb{E}[\log(1 - D(G(z)))]$: D muốn xác suất thấp với fake; G muốn xác suất cao.
- Tại điểm cân bằng Nash: $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)} = \frac{1}{2}$ khi $p_G = p_{\text{data}}$.
- Tương đương với cực tiểu **Jensen-Shannon divergence** $JSD(p_{\text{data}} \| p_G)$.

#### Bước 1: Vấn đề bão hòa và giải pháp

**Vấn đề:** Khi D rất mạnh, $D(G(z)) \approx 0$ → $\log(1 - D(G(z))) \approx \log(1) = 0$ → **gradient gần bằng 0** → G không học.

**Thực hành (non-saturating loss):** Thay vì minimize $\mathbb{E}[\log(1 - D(G(z)))]$, G maximize $\mathbb{E}[\log D(G(z))]$:

$$\max_\theta \; \mathbb{E}_{z \sim p(z)}[\log D_\phi(G_\theta(z))]$$

Khi $D(G(z)) \approx 0$: gradient là $\frac{1}{D(G(z))}$ — rất lớn → G học tốt hơn khi còn đang kém.

#### Lộ trình + Đánh đổi GAN

| | Nội dung |
|-|----------|
| Train | Minimax: D phân biệt thật/giả, G đánh lừa D |
| Inference | Chỉ G — một lần forward |
| **Ưu điểm** | Mẫu thường sắc nét; inference nhanh |
| **Nhược điểm** | Train không ổn định; mode collapse; không có $p_\theta(x)$; nhiều hyperparameter |

**Mode collapse:** G học cách chỉ sinh ra một vài mẫu "an toàn" mà D không phân biệt được — đa dạng kém. Vấn đề cơ bản của GAN.

---

## 5. Denoising Diffusion (DDPM)

### 5.1 Quan niệm

#### Ý tưởng lớn

DDPM lấy cảm hứng từ vật lý thống kê: **khuếch tán nhiệt** — một thứ được "pha loãng" vào background dần dần. Nếu biết quá trình pha loãng, có thể học cách đảo ngược nó.

**3 bước cốt lõi:**
1. **Forward noising:** thêm nhiễu Gaussian dần vào ảnh sạch $x_0$ qua $T$ bước, đến khi gần như chỉ còn nhiễu thuần $x_T \sim \mathcal{N}(0, I)$.
2. **Train:** cho mạng $\varepsilon_\theta(x_t, t)$ học **dự đoán nhiễu đã thêm vào** tại mỗi bước $t$.
3. **Inference:** bắt đầu từ $x_T \sim \mathcal{N}(0, I)$, chạy $T$ bước đảo ngược được học để dần "gỡ nhiễu" ra $x_0$.

**Phép ẩn dụ:** Giống như ảnh bị xé thành nhiều mảnh dần dần (mỗi bước xé thêm một ít), mô hình học cách ghép lại từng mảnh. Nhưng nếu bắt đầu từ "mảnh vụn" hoàn toàn ngẫu nhiên và áp dụng quá trình ghép, ta tạo ra ảnh mới chưa từng thấy.

#### Tại sao Diffusion hấp dẫn

| Tiêu chí | GAN | Diffusion |
|----------|-----|-----------|
| Mục tiêu train | Minimax (đối kháng) | MSE dự đoán nhiễu (ổn định) |
| Độ ổn định train | Khó, dễ sập | **Dễ, ổn định** |
| Chất lượng mẫu | Sắc, nhưng mode collapse | **Cao**, phủ nhiều mode |
| Tốc độ inference | **Nhanh** (1 forward pass) | Chậm (T forward passes) |

> **Nhớ ngắn:** GAN = sinh nhanh, train khó. Diffusion = train dễ, sinh chậm.

### 5.2 Góc nhìn Tính toán

#### Vòng huấn luyện

```python
for x0 in dataloader:
    t = torch.randint(1, T+1, (batch_size,))     # random timestep
    eps = torch.randn_like(x0)                   # true noise
    xt = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * eps  # noisy image
    eps_pred = model(xt, t)                      # predict noise
    loss = F.mse_loss(eps_pred, eps)             # simple L2 loss
    loss.backward(); optimizer.step()
```

#### Vòng sinh (Sampling)

```python
xT = torch.randn(batch_size, C, H, W)  # start from pure noise
xt = xT
for t in reversed(range(1, T+1)):
    eps_pred = model(xt, t)
    # DDPM reverse step (simplified):
    xt = (1/sqrt_alpha[t]) * (xt - beta[t]/sqrt_one_minus_alpha_bar[t] * eps_pred)
    if t > 1:
        xt += sqrt_beta[t] * torch.randn_like(xt)  # add noise for stochasticity
x0 = xt  # final generated image
```

#### DDPM trên MNIST (course demo)

```
samples/diffusion/
├── train_mnist_diffusion.py
├── config.yaml              # T, beta_start, beta_end, U-Net params
└── results/
    ├── samples.png          # generated samples
    ├── loss_curve.png       # MSE noise prediction
    └── best_ddpm.pt         # best checkpoint
```

**Chi tiết implementation:**
- Ảnh scale về $[-1, 1]$ — phù hợp với Gaussian noise.
- **Lịch β tuyến tính:** $\beta_t$ tăng dần từ `beta_start` đến `beta_end` qua $T$ bước.
- **U-Net nhỏ** với: residual blocks + **sinusoidal time embedding** (encode timestep $t$) + downsampling 7×7 + upsampling với skip connections.
- **Chi phí:** sampling cần $T$ lần forward pass qua U-Net → chậm hơn GAN (1 lần).

📸 [Cần ảnh: U-Net architecture cho DDPM — slide page 49] — Tìm: "DDPM U-Net architecture"

### 5.3 Toán học — Quá trình Khuếch tán

#### Bước 1-2: Quá trình xuôi (Forward Process)

**Lịch nhiễu:** Chọn $T$ và variance schedule $\beta_1, \ldots, \beta_T$ (các giá trị dương nhỏ).

**Định nghĩa:**
$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$

**Quá trình xuôi (Markov chain):**

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(\sqrt{1-\beta_t}\, x_{t-1},\; \beta_t I\right)$$

Nghĩa là: mỗi bước "làm mờ" $x_{t-1}$ một chút, thêm nhiễu $\beta_t$.

**Dạng đóng tại bất kỳ bước $t$ nào** (property của chuỗi Markov Gauss):

$$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t) I\right)$$

Tương đương:

$$\boxed{x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)}$$

**Quan trọng:** Có thể tính trực tiếp $x_t$ từ $x_0$ — không cần tính từng bước! Đây là lý do training rất hiệu quả.

Khi $t \to T$: $\bar{\alpha}_T \approx 0$, nên $x_T \approx \varepsilon \sim \mathcal{N}(0, I)$ — gần như thuần nhiễu.

#### Bước 3-4: Quá trình ngược và Loss

**Mục tiêu training:** Học mạng $\varepsilon_\theta(x_t, t)$ để dự đoán nhiễu $\varepsilon$ từ $(x_t, t)$.

**Training loss (simplified DDPM):**

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon}\!\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|_2^2\right]$$

Đây chỉ là **MSE đơn giản** giữa nhiễu thật và nhiễu dự đoán — không cần minimax, không cần likelihood phức tạp.

**Sampling (quá trình ngược):** Mỗi bước ngược dùng công thức:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta(x_t, t)\right) + \sqrt{\beta_t}\, z, \quad z \sim \mathcal{N}(0, I)$$

**Giải thích:** Bước ngược = lấy $x_t$, trừ đi nhiễu dự đoán (scaled), cộng thêm một chút nhiễu ngẫu nhiên để đảm bảo quá trình stochastic.

📸 [Cần ảnh: quá trình forward (x0 → xT) và reverse (xT → x0) — slide page 45-47] — Tìm: "DDPM forward reverse process"

---

## 6. Sinh văn bản

> Phần này chỉ có trong bản tiếng Việt (slide 56-58 của `GenerativeMethods.vne.pdf`). Nội dung đầy đủ ở `slides/10-modern-llms/ModernLLMs.tex`.

### 6.1 Văn bản là chuỗi — Cùng ý sinh, khác miền dữ liệu

Ảnh là vector $\mathbb{R}^{HWC}$ liên tục. Văn bản là **chuỗi rời rạc** — dãy token (từ, subword, ký tự).

**Phân tích tự hồi quy (autoregressive factorization):**

$$p(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T p(w_t \mid w_1, \ldots, w_{t-1}) = \prod_{t=1}^T p(w_t \mid w_{<t})$$

Nghĩa là: xác suất của cả câu = tích xác suất của từng từ **có điều kiện vào tất cả từ trước đó**. Mô hình học sinh từ **trái sang phải** (causal).

### 6.2 LM chỉ-decoder kiểu GPT

- Dùng **causal self-attention** (không nhìn tương lai): token $t$ chỉ attend đến token $1, \ldots, t-1$.
- **Training:** với một câu, dịch nhãn sang phải một bước — dự đoán token tiếp theo.
- **Inference:** sinh token lần lượt từ trái sang phải (autoregressive decoding).

**Các kỹ thuật liên quan** (xem thêm ở slide ModernLLMs):
- Tokenization (BPE, WordPiece)
- Decoding: greedy, beam search, sampling với temperature
- KV-cache để tăng tốc inference

### 6.3 Demo code sinh văn bản

```
samples/
├── text-gpt/          # GPT nhỏ cấp ký tự (tinyshakespeare.txt)
├── lucbat-gpt/        # TinyGPT thơ lục bát
├── lucbat-colab/      # notebook Colab
└── lucbat-lora-colab/ # fine-tune QLoRA + notebook sinh
```

---

## 7. Tổng hợp Tác vụ và Ứng dụng

### 7.1 Sinh có điều kiện (Conditional Generation)

**Trước:** sinh không điều kiện $p(x)$ — không kiểm soát được nội dung.

**Có điều kiện:** $p(x \mid y)$ hoặc $p(x \mid c)$ với $y$ là nhãn lớp, $c$ là prompt/thuộc tính.

**Các ví dụ thực tế:**
- **Conditional GAN:** generator nhận thêm $y$ (one-hot) → sinh ảnh đúng lớp.
- **Text-to-Image** (DALL-E, Stable Diffusion): $c$ = caption tiếng Anh → ảnh.
- **Style Transfer:** giữ content, đổi style — VAE hoặc diffusion với guidance.
- **Image Inpainting:** điền phần bị che dựa trên context xung quanh.

### 7.2 Tổng hợp dữ liệu (Data Synthesis)

Không chỉ ảnh — mô hình sinh có thể tổng hợp **bảng dữ liệu** và **chuỗi thời gian**:

| Miền | Kỹ thuật | Ứng dụng |
|------|----------|----------|
| Bảng (tabular) | CTGAN, VAE biến thể | Khách hàng, giao dịch giả |
| Chuỗi thời gian | AR, RNN, Transformer | Cảm biến IoT, dữ liệu y tế |

**Kịch bản sử dụng:**
- Tăng cường bộ nhỏ (few-shot augmentation).
- Chia sẻ dữ liệu "giả nhưng giống thật" (privacy-preserving).
- Tạo benchmark khi dữ liệu thật hiếm.

### 7.3 Đạo đức và Thực tiễn

**Mặt tích cực:**
- Công cụ sáng tạo (nghệ thuật AI, thiết kế).
- Tiếp cận (text-to-image cho người không biết vẽ).
- Mô phỏng an toàn (thử nghiệm y tế, tự lái xe).
- Tăng cường dữ liệu cho ML.

**Mặt tiêu cực và rủi ro:**
- **Deepfake:** ảnh/video giả mạo người thật — tác hại chính trị, xã hội.
- **Tin sai (misinformation):** sinh văn bản hoặc ảnh sai lệch thuyết phục.
- **Bản quyền:** mô hình sinh từ dữ liệu copyrighted — ai sở hữu output?
- **Khuếch đại thiên kiến:** nếu training data có bias, mô hình sinh sẽ khuếch đại.

**Thách thức kỹ thuật:**
- **Đánh giá chất lượng:** FID (Fréchet Inception Distance), IS (Inception Score) — chưa hoàn hảo.
- **Kiểm soát đầu ra:** tránh sinh nội dung có hại (content filter, RLHF).
- **Chống lạm dụng:** watermarking ảnh AI.

> **Bài học:** Công nghệ sinh mạnh nhưng trách nhiệm triển khai cũng lớn. Kỹ sư cần hiểu cả năng lực lẫn giới hạn.

---

## 8. So sánh VAE / GAN / DDPM

### 8.1 Luồng xử lý

```
VAE:
  x ──[Encoder]──► (μ, σ²) ──[sample z]──► [Decoder] ──► x̂
  Inference: z ∼ N(0,I) ──────────────────► [Decoder] ──► x̂_new

GAN:
  z ∼ N(0,I) ──► [Generator G] ──► x̃
  Training: x̃ vs x_real ──► [Discriminator D] ──► real/fake signal

DDPM:
  Training:  x0 ──[+noise × T]──► xT   ;  ε_θ(xt, t) predicts noise
  Inference: xT ──[denoise × T]──► x0_new
```

### 8.2 So sánh định tính

| Tiêu chí | VAE | GAN | DDPM |
|----------|-----|-----|------|
| **Likelihood** | ELBO (xấp xỉ) | Không có $p(x)$ | Chuỗi bước Gauss |
| **Độ sắc mẫu** | Thường mờ | Thường sắc (nếu ổn định) | Có thể sắc (phụ thuộc $T$) |
| **Train** | Tương đối ổn | Có thể không ổn | Ổn, nhưng chậm |
| **Inference** | Nhanh (1 forward) | Nhanh (1 forward) | **Chậm ($T$ forward)** |
| **Mode coverage** | Tốt | Dễ mode collapse | **Rất tốt** |
| **Latent space** | Có (encoder) | Không có encoder | Không (latent = noisy image) |
| **Điểm mạnh** | Structured latent, stable | Sample quality, speed | Quality + stability |
| **Điểm yếu** | Blur | Instability, mode collapse | Slow sampling |

### 8.3 Kinh nghiệm thực tế (MNIST demos)

| Demo | Files chính | Metrics |
|------|-------------|---------|
| **VAE** | `samples/vae/` | `reconstructions.png`, `samples.png`, val ELBO |
| **GAN** | `samples/gan/` (MLP) + `config_dcgan.yaml` (conv) | `samples.png`, val BCE D |
| **Diffusion** | `samples/diffusion/` | `samples.png`, val MSE noise |

**Quan sát:** So sánh trực quan:
- VAE: samples mờ hơn, nhưng reconstruction ổn định.
- GAN MLP: nhanh nhưng diversity kém; DCGAN sắc hơn.
- Diffusion: chất lượng cao, đa dạng tốt nhưng inference chậm hơn nhiều.

---

## 9. Tóm tắt

### Điểm chính cần nhớ

| Khái niệm | Công thức / Ý chính |
|-----------|---------------------|
| **Mô hình sinh** | Học $p(x)$ hay $p(x,y)$ — cho phép sample, augment, compress |
| **VAE loss** | $\mathcal{L} = -\mathbb{E}[\log p_\theta(x\|z)] + KL(q_\phi\|p)$ |
| **Reparameterization** | $z = \mu + \sigma \odot \varepsilon$, $\varepsilon \sim \mathcal{N}(0,I)$ — gradient qua sample |
| **GAN minimax** | $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ |
| **Non-saturating G** | Maximize $\mathbb{E}[\log D(G(z))]$ — tránh vanishing gradient |
| **DDPM forward** | $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ |
| **DDPM loss** | $\mathcal{L} = \mathbb{E}\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2$ — chỉ là MSE! |
| **DDPM sampling** | Start $x_T \sim \mathcal{N}(0,I)$, denoise $T$ bước |
| **Text LM** | $p(w_1\ldots w_T) = \prod p(w_t\|w_{<t})$ — causal autoregressive |
| **Ethics** | Deepfake, bias, copyright — trách nhiệm kỹ sư DL |

### Lộ trình học

```
Nếu bắt đầu:
  → Nắm AE → VAE conceptual → VAE loss 2 terms → GAN big picture → DDPM 5 steps

Nếu muốn hiểu sâu hơn:
  → Chứng minh ELBO đầy đủ (Steps 1-4) → GAN Nash equilibrium → DDPM closed-form noising
```

---

📸 **Ảnh slide quan trọng cần bổ sung:**

| Slide | Nội dung | Nguồn gợi ý |
|-------|----------|-------------|
| Page 8 (EN) | So sánh 5 họ mô hình sinh | "generative model taxonomy deep learning" |
| Page 15 (EN) | AE vs VAE latent space | "VAE latent space structure" |
| Page 18 (EN) | VAE computational graph | "VAE reparameterization trick diagram" |
| Page 34 (VNE) | GAN training flow diagram | "GAN architecture diagram" |
| Page 45 (VNE) | DDPM forward process | "DDPM forward process visualization" |
| Page 47 (VNE) | DDPM reverse sampling | "DDPM denoising steps" |
| Page 54 (VNE) | So sánh VAE/GAN/DDPM luồng | "VAE GAN diffusion comparison pipeline" |
