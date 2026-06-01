# SOME REVIEW TOPICS FOR FINAL EXAM HK252

Status: Not started

### **1. Mô hình Markov Ẩn (Hidden Markov Models - HMM)**

- **Cấu trúc mô hình:** Tính chất Markov (bậc 1, bậc 2), Ma trận chuyển trạng thái ($A$), Ma trận bức xạ/phát xạ ($B$), Phân bố xác suất ban đầu ($\pi$). Ý nghĩa vật lý của các ma trận trong ứng dụng thực tế (ví dụ: nhận dạng giọng nói, phân tích chuỗi thời gian).
- **Thuật toán Forward & Tính xác suất chuỗi (Evaluation):** Tính xác suất cận biên (Likelihood), cách xử lý khi khuyết điểm dữ liệu quan sát (Marginalization).
- **Thuật toán giải mã (Decoding):** * **Viterbi:** Nguyên lý tối ưu Bellman, quy hoạch động, độ phức tạp thuật toán, kỹ thuật tính toán trong không gian Logarit (Log-space) để tránh lỗi underflow và xử lý xác suất bằng 0.
- **Forward-Backward Decoding:** Tối đa hóa xác suất biên tại từng thời điểm độc lập ($\gamma_t(i)$) và nhược điểm (nguy cơ tạo ra chuỗi trạng thái không hợp lệ).
- **Huấn luyện mô hình (Learning/Training):** Thuật toán Baum-Welch (Expectation-Maximization), cách tính biến kỳ vọng trung gian ($\xi_t(i,j)$), và ảnh hưởng của các điều kiện khởi tạo (ví dụ: khởi tạo bằng ma trận đơn vị).

### **2. Máy học Vector Hỗ trợ (Support Vector Machines - SVM)**

- **Biên phân quyết định (Decision Boundary):** Khả năng phân tách tuyến tính trong các không gian số chiều thấp (1D, 2D) và không gian n-chiều.
- **Lề (Margin):** Công thức toán học và phương pháp tính khoảng cách lề (Margin width) từ phương trình siêu phẳng.
- **Soft-margin SVM:** Siêu tham số $C$ (tỷ lệ đánh đổi giữa việc tối đa hóa lề và giảm thiểu biến nới lỏng - slack variables/lỗi huấn luyện), hiện tượng Overfit/Underfit khi điều chỉnh $C$.
- **Hàm nhân (Kernel Trick):** RBF Kernel (hàm cơ sở xuyên tâm), cấu trúc ma trận Kernel (Gram matrix), và hành vi của mô hình khi siêu tham số $\gamma$ tiến tới các giá trị cực đoan ($\infty$).
- **Hàm mục tiêu:** Đặc điểm của Hinge Loss và sự khác biệt về mặt bản chất hình học so với hàm mục tiêu của Hồi quy Logistic.

### **3. Hồi quy Logistic (Logistic Regression) & Mạng Nơ-ron cơ bản**

- **Tính toán dự đoán:** Biểu thức Logit (Log-odds), tính Tỷ lệ Odds (Odds ratio) từ trọng số và vector đầu vào.
- **Đánh giá mô hình:** Mối quan hệ giữa Ngưỡng quyết định (Decision Threshold) và các độ đo Precision, Recall, F1-Score.
- **Hàm mất mát và Tối ưu hóa:** * Hàm Cross-Entropy (Log-loss): Công thức, đạo hàm theo hàm ẩn (logit $z$), và ưu điểm triệt tiêu đạo hàm của hàm Sigmoid.
- Hàm Mean Squared Error (MSE): Nguyên nhân toán học gây ra hiện tượng triệt tiêu đạo hàm (Gradient Vanishing) khi kết hợp với hàm kích hoạt Sigmoid.
- **Điều chuẩn (Regularization):** Tác dụng của L2-norm trong việc kiểm soát trọng số.

### **4. Naive Bayes**

- **Định lý Bayes cơ bản:** Tính xác suất hậu nghiệm (Posterior) dựa trên xác suất tiên nghiệm (Prior), Likelihood và xác suất biên.
- **Giả định cốt lõi:** Giả định độc lập có điều kiện (Conditional Independence) giữa các đặc trưng và các hệ quả toán học bất lợi trong thực tế (ví dụ: mô hình trở nên quá tự tin - overconfident làm lệch cực đoan xác suất hậu nghiệm).
- **Ứng dụng:** Multinomial Naive Bayes trong phân loại văn bản (Text Classification).

### **5. Phân cụm K-Means (K-Means Clustering)**

- **Hàm mục tiêu:** Tính chất của tổng bình phương khoảng cách (WCSS), tính phi lồi (non-convex) của không gian phân gán cụm dẫn đến cực tiểu cục bộ (local minima).
- **Đặc tính thuật toán lặp (Lloyd's Algorithm):** Cơ sở toán học đảm bảo sự hội tụ sau một số vòng lặp hữu hạn (do tính hữu hạn của không gian tổ hợp).
- **Biến thể khoảng cách:** Sự thay đổi bước cập nhật tâm cụm (Update step) khi thay thế hàm khoảng cách Euclidean bằng L1-Norm / Manhattan (chuyển từ Mean sang Median - K-Medians).

### **6. Thuật toán K-Lân cận gần nhất (K-Nearest Neighbors - KNN)**

- **Cấu trúc dữ liệu tối ưu:** Sử dụng KD-Tree hoặc Ball-Tree để giảm độ phức tạp tìm kiếm không gian từ $\mathcal{O}(N \cdot D)$ xuống mức Logarit, thay vì duyệt tuần tự.

**Hành vi siêu tham số $K$:**

Sự thay đổi của biên phân quyết định và lỗi huấn luyện (Training Error) khi $K$ di chuyển từ $1$ đến $N$ (tổng số mẫu). Hiện tượng Underfitting / High Bias tột độ khi $K=N$ (dự đoán theo lớp đa số - Majority Class).