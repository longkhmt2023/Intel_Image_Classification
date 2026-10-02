# Intel Image Scene Classification

Dự án này xây dựng một pipeline hoàn chỉnh cho bài toán **phân loại ảnh cảnh tự nhiên** trên bộ dữ liệu **Intel Image Classification**. Mục tiêu của hệ thống là dự đoán một ảnh đầu vào thuộc một trong sáu lớp cảnh:

```text
buildings
forest
glacier
mountain
sea
street
```

Dự án so sánh hai hướng tiếp cận chính:

1. **Machine Learning truyền thống với đặc trưng thủ công**
   - Raw pixel / flattened image features
   - RGB color histogram
   - HOG features
   - Kết hợp HOG + RGB histogram
   - Logistic Regression, KNN và SVM

2. **Deep Learning cho phân loại ảnh**
   - Simple CNN huấn luyện từ đầu
   - ResNet18 sử dụng transfer learning
   - ResNet18 fine-tuning
   - Phân tích lỗi, trực quan hóa kết quả và Grad-CAM

Trong các thử nghiệm hiện tại, mô hình tốt nhất là **ResNet18 Fine-tuned**, đạt khoảng **93.33% test accuracy** và **93.44% test Macro F1** trên tập test.

Ngoài phần huấn luyện và đánh giá mô hình, dự án còn có **demo bằng Gradio trong notebook 09**, cho phép upload ảnh và chọn model để dự đoán trực tiếp.

---

## Mục lục

- [Tổng quan dự án](#tổng-quan-dự-án)
- [Bộ dữ liệu](#bộ-dữ-liệu)
- [Mục tiêu dự án](#mục-tiêu-dự-án)
- [Phương pháp thực hiện](#phương-pháp-thực-hiện)
- [Các mô hình đã triển khai](#các-mô-hình-đã-triển-khai)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Demo Gradio](#demo-gradio)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Cách chạy dự án](#cách-chạy-dự-án)
- [Các file đầu ra quan trọng](#các-file-đầu-ra-quan-trọng)
- [Giải thích và phân tích mô hình](#giải-thích-và-phân-tích-mô-hình)
- [Nhận xét chính](#nhận-xét-chính)
- [Hạn chế](#hạn-chế)
- [Hướng phát triển](#hướng-phát-triển)
- [Tác giả](#tác-giả).

---

## Tổng quan dự án

Dự án tập trung vào bài toán **Natural Scene Image Classification**. Với một ảnh đầu vào, hệ thống cần phân loại ảnh đó vào một trong sáu nhóm cảnh: `buildings`, `forest`, `glacier`, `mountain`, `sea`, hoặc `street`.

Quy trình tổng thể của dự án:

```text
Khảo sát dữ liệu
→ Tiền xử lý dữ liệu
→ Tạo train/validation/test split
→ Trích xuất đặc trưng thủ công
→ Huấn luyện mô hình Machine Learning truyền thống
→ Tuning hyperparameter
→ Phân tích lỗi mô hình truyền thống
→ Trực quan hóa đặc trưng
→ Huấn luyện Simple CNN
→ Transfer learning với ResNet18
→ Fine-tuning ResNet18
→ Đánh giá mô hình
→ Grad-CAM visualization
→ Demo dự đoán bằng Gradio
→ So sánh mô hình cuối cùng
```

Mục tiêu của dự án không chỉ là đạt accuracy cao, mà còn phân tích được **vì sao các mô hình hoạt động khác nhau**, **vì sao có những lớp dễ bị nhầm lẫn**, và **mô hình đang tập trung vào vùng nào trong ảnh khi dự đoán**.

---

## Bộ dữ liệu

Dự án sử dụng bộ dữ liệu **Intel Image Classification** trên Kaggle:

```text
https://www.kaggle.com/datasets/puneet6060/intel-image-classification
```

Bộ dữ liệu gồm ảnh cảnh tự nhiên được chia thành sáu lớp:

| Lớp | Ý nghĩa |
|---|---|
| `buildings` | Cảnh đô thị, tòa nhà, kiến trúc |
| `forest` | Rừng cây, thảm thực vật |
| `glacier` | Băng, tuyết, sông băng |
| `mountain` | Núi, cảnh quan địa hình cao |
| `sea` | Biển, đại dương, bờ biển |
| `street` | Đường phố, cảnh giao thông đô thị |

### Tải dữ liệu

Có thể tải dữ liệu trực tiếp từ Kaggle hoặc dùng Kaggle API:

```bash
kaggle datasets download -d puneet6060/intel-image-classification
```

Sau khi giải nén, dữ liệu nên được đặt vào thư mục `data/raw/` theo dạng:

```text
intel_image_project/
└── data/
    └── raw/
        ├── seg_train/
        ├── seg_test/
        └── seg_pred/
```

Một số bản tải từ Kaggle có thể có thêm một cấp thư mục lồng nhau, ví dụ `seg_train/seg_train/`. Khi chạy notebook, cần kiểm tra lại đường dẫn dữ liệu để khớp với cấu trúc thực tế trên máy hoặc trên Google Drive.

### Chia tập dữ liệu

Tất cả các mô hình dùng chung một cách chia dữ liệu để đảm bảo việc so sánh công bằng.

| Split | Số lượng ảnh |
|---|---:|
| Train | 11,212 |
| Validation | 2,804 |
| Test | 3,000 |

Việc dùng chung tập train, validation và test giúp so sánh các mô hình truyền thống và deep learning một cách nhất quán.

---

## Mục tiêu dự án

Dự án có bốn mục tiêu chính:

### 1. Xây dựng pipeline phân loại ảnh hoàn chỉnh

- Đọc và khảo sát dữ liệu ảnh.
- Kiểm tra chất lượng ảnh.
- Tiền xử lý dữ liệu.
- Tạo các tập train, validation và test.
- Lưu metadata để tái sử dụng ở các notebook sau.

### 2. Đánh giá các mô hình Machine Learning truyền thống

- Trích xuất đặc trưng thủ công từ ảnh.
- Huấn luyện Logistic Regression, KNN và SVM.
- So sánh các loại đặc trưng: raw pixel, RGB histogram, HOG và HOG + RGB histogram.
- Tuning hyperparameter để cải thiện hiệu năng.

### 3. Đánh giá các mô hình Deep Learning

- Huấn luyện Simple CNN từ đầu.
- Sử dụng ResNet18 pretrained theo hướng transfer learning.
- Fine-tune ResNet18 để thích nghi tốt hơn với bộ dữ liệu cảnh tự nhiên.

### 4. Phân tích hành vi mô hình

- Sử dụng confusion matrix.
- Phân tích per-class precision, recall và F1-score.
- Tìm các cặp lớp dễ nhầm lẫn.
- Quan sát ảnh bị dự đoán sai.
- Dùng Grad-CAM để giải thích vùng ảnh mà mô hình CNN/ResNet18 tập trung vào.
- Xây dựng demo Gradio để kiểm thử mô hình trực quan.

---

## Phương pháp thực hiện

### 1. Khảo sát dữ liệu

Bước đầu tiên là phân tích dữ liệu để nắm rõ:

- Số lượng lớp.
- Số lượng ảnh của từng lớp.
- Một số ảnh mẫu ở mỗi lớp.
- Chất lượng ảnh và các vấn đề có thể xuất hiện.
- Phân phối dữ liệu giữa các lớp.

Bước này giúp xác định dữ liệu có phù hợp cho cả mô hình truyền thống và mô hình deep learning hay không.

---

### 2. Tiền xử lý dữ liệu

Giai đoạn tiền xử lý gồm:

- Kiểm tra file ảnh hợp lệ.
- Loại bỏ hoặc bỏ qua ảnh lỗi nếu cần.
- Tạo ánh xạ class sang index.
- Tạo train/validation/test split.
- Lưu metadata để dùng lại trong các notebook khác.
- Tính mean và standard deviation khi cần chuẩn hóa ảnh.

Các file metadata quan trọng:

```text
data/metadata/class_to_idx.json
data/metadata/idx_to_class.json
data/metadata/preprocessing_summary.json
```

Các file này giúp pipeline có tính tái lập và tránh sai lệch thứ tự class giữa các notebook.

---

### 3. Trích xuất đặc trưng thủ công

Các mô hình Machine Learning truyền thống không tự học đặc trưng không gian và ngữ nghĩa từ ảnh giống như CNN. Vì vậy, trước khi huấn luyện, ảnh cần được chuyển thành vector đặc trưng.

Dự án sử dụng bốn nhóm đặc trưng chính.

#### Flattened Pixel Features

Ảnh được resize và làm phẳng thành vector một chiều.

**Ưu điểm:**

- Dễ triển khai.
- Giữ lại thông tin pixel gốc.

**Hạn chế:**

- Số chiều rất lớn.
- Nhạy với vị trí vật thể, ánh sáng và nền ảnh.
- Không mô tả rõ shape, texture hoặc cấu trúc cảnh.

---

#### RGB Color Histogram

RGB histogram biểu diễn phân phối màu sắc của ảnh.

**Ưu điểm:**

- Hữu ích với các lớp có màu sắc đặc trưng.
- Có thể hỗ trợ phân biệt `forest`, `sea`, `glacier`, `mountain`.
- Số chiều thấp hơn raw pixel.

**Hạn chế:**

- Không giữ thông tin vị trí.
- Không mô tả tốt hình dạng hoặc bố cục cảnh.
- Có thể nhầm lẫn nếu hai lớp có màu tương tự nhau.

---

#### HOG Features

HOG, viết tắt của **Histogram of Oriented Gradients**, mô tả hướng cạnh và gradient trong ảnh.

**Ưu điểm:**

- Biểu diễn tốt shape, contour và texture.
- Ổn định hơn raw pixel.
- Phù hợp với các cảnh có cấu trúc mạnh như `buildings`, `street`, `mountain`.

**Hạn chế:**

- Mất thông tin màu sắc.
- Không nắm bắt được ngữ nghĩa cấp cao.
- Vẫn yếu hơn đặc trưng được học tự động bởi CNN.

---

#### HOG + RGB Histogram

Trong thí nghiệm SVM, dự án sử dụng đặc trưng kết hợp:

```text
Final feature vector = HOG features + RGB histogram features
```

Lý do kết hợp:

| Loại đặc trưng | Thông tin biểu diễn |
|---|---|
| HOG | Cạnh, hướng gradient, shape, texture |
| RGB Histogram | Phân phối màu sắc toàn cục |

Cách kết hợp này giúp SVM có thêm thông tin cả về **cấu trúc hình học** lẫn **màu sắc cảnh**. Đây là lựa chọn hợp lý vì ảnh scene classification thường phụ thuộc đồng thời vào texture, bố cục và màu sắc.

Ví dụ:

- `forest` thường có nhiều vùng màu xanh lá.
- `sea` thường có vùng xanh dương hoặc màu nước biển.
- `glacier` thường chứa trắng, xám, xanh nhạt.
- `buildings` và `street` thường có nhiều cạnh, đường thẳng và cấu trúc hình học.

---

### 4. Huấn luyện mô hình truyền thống

Các mô hình truyền thống được sử dụng gồm:

| Mô hình | Đặc trưng đầu vào | Vai trò |
|---|---|---|
| Logistic Regression | Handcrafted features | Baseline tuyến tính mạnh, ổn định |
| KNN | Handcrafted features | Baseline dựa trên khoảng cách |
| SVM | HOG + RGB Histogram | Classifier dựa trên margin, dùng đặc trưng kết hợp |

Các mô hình này giúp đánh giá vai trò của feature engineering trong bài toán phân loại ảnh.

---

### 5. Huấn luyện mô hình Deep Learning

Dự án sử dụng hai hướng chính:

1. **Simple CNN**
2. **ResNet18 pretrained**

Khác với mô hình truyền thống, CNN có khả năng học đặc trưng trực tiếp từ ảnh thông qua các convolutional layers. Điều này giúp mô hình học được edge, texture, pattern và đặc trưng cấp cao mà không cần thiết kế thủ công như HOG hoặc RGB histogram.

---

## Các mô hình đã triển khai

### Logistic Regression

Logistic Regression được dùng làm baseline truyền thống mạnh. Dù là mô hình tuyến tính, Logistic Regression vẫn có thể đạt kết quả tốt nếu được kết hợp với đặc trưng phù hợp như HOG.

Vai trò chính của Logistic Regression trong dự án là tạo một mốc so sánh đơn giản, ổn định và dễ giải thích.

---

### K-Nearest Neighbors

KNN là mô hình dựa trên khoảng cách. Một ảnh được phân loại dựa trên nhãn của các vector đặc trưng gần nó nhất.

KNN hữu ích để so sánh, nhưng thường gặp hạn chế với dữ liệu ảnh có số chiều cao vì khoảng cách trong không gian nhiều chiều có thể kém ổn định.

---

### SVM + HOG + RGB Histogram

SVM được huấn luyện với đặc trưng kết hợp:

```text
HOG features + RGB histogram features
```

Trong phần demo Gradio, pipeline của model SVM `.pkl` là:

```text
Ảnh đầu vào
→ Resize 128x128
→ Trích xuất HOG với orientations = 12
→ Trích xuất RGB Histogram với bins = (8, 8, 8)
→ Ghép HOG + RGB Histogram
→ SVM
```

Số chiều đặc trưng:

```text
HOG 128x128: 10800 features
RGB Histogram: 512 features
Tổng cộng: 11312 features
```

Đây là mô hình truyền thống có ý nghĩa tốt trong đồ án vì nó cho thấy cách kết hợp đặc trưng thủ công để cải thiện biểu diễn ảnh.

---

### Simple CNN

Simple CNN là baseline deep learning được huấn luyện từ đầu.

Mô hình học đặc trưng trực tiếp từ ảnh thông qua các convolutional layers. So với mô hình truyền thống, Simple CNN không cần trích xuất thủ công HOG hoặc histogram màu.

Vai trò của Simple CNN là trả lời câu hỏi:

```text
Một CNN tự huấn luyện từ đầu có thể đạt hiệu quả như thế nào trên bộ dữ liệu này?
```

---

### ResNet18 Feature Extraction

Ở giai đoạn feature extraction, ResNet18 pretrained được dùng như một bộ trích xuất đặc trưng. Phần lớn backbone được đóng băng, chỉ huấn luyện lại lớp phân loại cuối cho sáu lớp cảnh.

Cách làm này tận dụng các đặc trưng ImageNet đã học sẵn như cạnh, texture, pattern, object parts và cấu trúc thị giác phổ biến.

---

### ResNet18 Fine-tuned

Ở giai đoạn fine-tuning, một phần mạng ResNet18 được mở khóa và tiếp tục huấn luyện trên bộ dữ liệu Intel Image Classification.

Fine-tuning giúp mô hình thích nghi tốt hơn với dữ liệu cảnh tự nhiên, từ đó cải thiện kết quả so với chỉ dùng feature extraction.

Trong thí nghiệm hiện tại, **ResNet18 Fine-tuned là mô hình có hiệu năng tốt nhất**.

---

## Kết quả thực nghiệm

### Kết quả Deep Learning

| Mô hình | Test Accuracy | Test Macro F1 | Vai trò |
|---|---:|---:|---|
| Simple CNN | 0.8783 | 0.8793 | Baseline deep learning |
| ResNet18 Fine-tuned | 0.9333 | 0.9344 | Mô hình tốt nhất |

### Nhận xét chính

ResNet18 Fine-tuned vượt trội hơn Simple CNN.

So với Simple CNN:

- Macro F1 tăng khoảng **5.5 percentage points**.
- Số ảnh dự đoán sai giảm từ **365** xuống **200** trên tập test 3,000 ảnh.
- Mô hình ổn định hơn ở các lớp có hình ảnh dễ nhầm lẫn.

Điều này cho thấy transfer learning và fine-tuning là hướng tiếp cận rất hiệu quả với bài toán phân loại ảnh khi dữ liệu không quá lớn.

### Kết quả mô hình truyền thống

Các kết quả của Logistic Regression, KNN và SVM được lưu trong thư mục `outputs/metrics/`, đặc biệt là:

```text
outputs/metrics/traditional_ml_baselines_metrics.csv
outputs/metrics/traditional_ml_tuning_results.csv
outputs/metrics/logistic_regression_test_metrics.json
outputs/metrics/svm_hog_rgb_test_metrics.json
outputs/metrics/final_model_comparison.csv
```

Khi trình bày báo cáo hoặc cập nhật README sau cùng, nên lấy số liệu trực tiếp từ `final_model_comparison.csv` để đảm bảo bảng kết quả khớp với lần chạy mới nhất.

---

## Demo Gradio

Dự án có notebook demo bằng Gradio:

```text
notebooks/09_gradio_demo_intel_image_project_SVM_HOG_RGB.ipynb
```

Notebook này cho phép upload một ảnh cảnh tự nhiên và chọn model để dự đoán một trong sáu lớp.

### Các model hỗ trợ trong demo

| Model | Pipeline chính | Ghi chú |
|---|---|---|
| ResNet18 Fine-tuned | Resize 224x224 → Normalize ImageNet → ResNet18 | Model chính, kết quả tốt nhất |
| HOG + Logistic Regression tuned | Resize 64x64 → Grayscale → HOG 1764 features → StandardScaler → Logistic Regression | Model truyền thống |
| Simple CNN | Resize ảnh theo cấu hình lúc train → Simple CNN | Baseline deep learning |
| SVM + HOG + RGB | Resize 128x128 → HOG 10800 + RGB Histogram 512 → SVM | Model `.pkl` bổ sung |

### Chạy demo

Trong Google Colab, chạy notebook 09 theo thứ tự các cell. Cell cuối sẽ tạo giao diện Gradio:

```python
demo.launch(
    share=True,
    debug=True
)
```

Sau khi chạy, Gradio sẽ tạo một đường link public tạm thời để demo trực tiếp.

### Ý nghĩa của demo

Demo giúp kiểm chứng mô hình theo cách trực quan hơn:

- Upload ảnh bất kỳ.
- Chọn model cần thử nghiệm.
- Xem xác suất dự đoán cho cả sáu lớp.
- So sánh trực tiếp sự khác nhau giữa mô hình truyền thống và deep learning.

Đây là phần phù hợp để trình bày trong báo cáo hoặc buổi thuyết trình vì nó biến kết quả huấn luyện thành một ứng dụng có thể tương tác.

---

## Cấu trúc thư mục

Cấu trúc dự án được tổ chức như sau:

```text
intel_image_project/
│
├── data/
│   ├── raw/
│   │   ├── seg_train/
│   │   ├── seg_test/
│   │   └── seg_pred/
│   │
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   └── metadata/
│       ├── class_to_idx.json
│       ├── idx_to_class.json
│       └── preprocessing_summary.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_traditional_ml_baselines.ipynb
│   ├── 04_traditional_ml_tuning.ipynb
│   ├── 05_feature_visualization.ipynb
│   ├── 06_deep_learning_train.ipynb
│   ├── 07_logistic_regression_error_analysis.ipynb
│   ├── 08_svm_hog_rgb_error_analysis.ipynb
│   └── 09_gradio_demo_intel_image_project_SVM_HOG_RGB.ipynb
│
├── models/
│   ├── traditional_ml/
│   │   ├── best_baseline_model.joblib
│   │   ├── best_tuned_model.joblib
│   │   └── svm_hog_rgb_model.joblib
│   │
│   ├── cs114/
│   │   └── best_tuned_model_full.joblib
│   │
│   └── cs231/
│       ├── simple_cnn_best.pth
│       ├── resnet18_feature_extractor_best.pth
│       └── resnet18_finetuned_best.pth
│
├── outputs/
│   ├── features/
│   │   └── traditional_ml/
│   │       ├── train_hog_features.npz
│   │       ├── val_hog_features.npz
│   │       ├── test_hog_features.npz
│   │       ├── train_rgb_hist_features.npz
│   │       ├── val_rgb_hist_features.npz
│   │       ├── test_rgb_hist_features.npz
│   │       ├── train_hog_rgb_features.npz
│   │       ├── val_hog_rgb_features.npz
│   │       └── test_hog_rgb_features.npz
│   │
│   ├── predictions/
│   │   ├── logistic_regression_test_predictions.csv
│   │   ├── svm_hog_rgb_test_predictions.csv
│   │   ├── simple_cnn_test_predictions.csv
│   │   ├── resnet18_feature_extractor_test_predictions.csv
│   │   └── resnet18_finetuned_test_predictions.csv
│   │
│   ├── metrics/
│   │   ├── traditional_ml_baselines_metrics.csv
│   │   ├── traditional_ml_tuning_results.csv
│   │   ├── logistic_regression_test_metrics.json
│   │   ├── svm_hog_rgb_test_metrics.json
│   │   ├── simple_cnn_test_metrics.json
│   │   ├── resnet18_finetuned_test_metrics.json
│   │   ├── cnn_model_comparison.csv
│   │   └── final_model_comparison.csv
│   │
│   └── figures/
│       ├── dataset/
│       ├── features/
│       ├── traditional_ml/
│       ├── deep_learning/
│       ├── gradcam/
│       └── final/
│
├── reports/
│   └── final_report/
│
├── requirements.txt
└── README.md
```

> Lưu ý: Không nên đưa toàn bộ dữ liệu ảnh, file model lớn hoặc checkpoint nặng trực tiếp lên GitHub. Nên dùng `.gitignore` cho các thư mục như `data/raw/`, `models/` hoặc các file checkpoint lớn nếu cần.

---

## Cài đặt môi trường

### 1. Clone repository

```bash
git clone https://github.com/your-username/intel-image-scene-classification.git
cd intel-image-scene-classification
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt môi trường:

```bash
# Windows
venv\Scripts\activate
```

```bash
# Linux / macOS
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Các thư viện chính:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
scikit-image
opencv-python
opencv-python-headless
Pillow
tqdm
joblib
torch
torchvision
gradio
```

Với notebook Gradio trên Google Colab, có thể cài nhanh:

```python
!pip -q install gradio scikit-image joblib opencv-python-headless
```

---

## Cách chạy dự án

Nên chạy các notebook theo thứ tự.

### Bước 1: Khảo sát dữ liệu

```text
notebooks/01_eda.ipynb
```

Notebook này dùng để kiểm tra số lượng ảnh, phân phối class và hiển thị ảnh mẫu.

---

### Bước 2: Tiền xử lý dữ liệu

```text
notebooks/02_preprocessing.ipynb
```

Notebook này tạo các file:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
data/metadata/class_to_idx.json
data/metadata/idx_to_class.json
```

---

### Bước 3: Huấn luyện baseline truyền thống

```text
notebooks/03_traditional_ml_baselines.ipynb
```

Notebook này trích xuất đặc trưng và huấn luyện các mô hình baseline:

- Logistic Regression
- KNN
- SVM

Các loại đặc trưng gồm:

- Flattened pixel features
- RGB histogram
- HOG
- HOG + RGB histogram

---

### Bước 4: Tuning mô hình truyền thống

```text
notebooks/04_traditional_ml_tuning.ipynb
```

Notebook này tuning hyperparameter và lưu mô hình truyền thống tốt nhất.

---

### Bước 5: Trực quan hóa đặc trưng

```text
notebooks/05_feature_visualization.ipynb
```

Notebook này trực quan hóa:

- Ảnh gốc
- Ảnh grayscale
- Các kênh RGB
- Sobel edge maps
- HOG visualization
- RGB color histogram

Bước này giúp giải thích sự khác biệt giữa đặc trưng thiết kế thủ công và đặc trưng học tự động bởi CNN.

---

### Bước 6: Huấn luyện mô hình Deep Learning

```text
notebooks/06_deep_learning_train.ipynb
```

Notebook này huấn luyện:

- Simple CNN
- ResNet18 Feature Extractor
- ResNet18 Fine-tuned

Kết quả được lưu gồm checkpoint, lịch sử huấn luyện, metrics và prediction files.

---

### Bước 7: Phân tích lỗi Logistic Regression

```text
notebooks/07_logistic_regression_error_analysis.ipynb
```

Notebook này thực hiện:

- Tính overall metrics.
- Phân tích per-class metrics.
- Vẽ confusion matrix.
- Tìm top confused pairs.
- Hiển thị ảnh bị dự đoán sai.

---

### Bước 8: Phân tích lỗi SVM + HOG + RGB

```text
notebooks/08_svm_hog_rgb_error_analysis.ipynb
```

Notebook này tập trung vào SVM dùng đặc trưng kết hợp:

```text
HOG + RGB histogram
```

Kết quả gồm:

- SVM test predictions.
- SVM overall metrics.
- SVM per-class metrics.
- SVM confusion matrix.
- SVM top confused pairs.
- SVM misclassified examples.
- So sánh với Logistic Regression.

---

### Bước 9: Chạy demo Gradio

```text
notebooks/09_gradio_demo_intel_image_project_SVM_HOG_RGB.ipynb
```

Notebook này load các model đã huấn luyện và tạo giao diện demo bằng Gradio.

Các model được hỗ trợ:

- ResNet18 Fine-tuned
- HOG + Logistic Regression tuned
- Simple CNN
- SVM + HOG + RGB

Sau khi chạy cell cuối, Colab sẽ tạo link public để mở giao diện demo.

---

## Các file đầu ra quan trọng

### File model

| File | Mô tả |
|---|---|
| `models/traditional_ml/best_tuned_model.joblib` | Mô hình truyền thống tốt nhất sau tuning |
| `models/traditional_ml/svm_hog_rgb_model.joblib` | SVM dùng HOG + RGB histogram |
| `models/cs114/best_tuned_model_full.joblib` | Logistic Regression tuned dùng HOG |
| `models/cs231/simple_cnn_best.pth` | Checkpoint tốt nhất của Simple CNN |
| `models/cs231/resnet18_feature_extractor_best.pth` | Checkpoint ResNet18 feature extraction |
| `models/cs231/resnet18_finetuned_best.pth` | Checkpoint ResNet18 fine-tuned |

---

### File metrics

| File | Mô tả |
|---|---|
| `outputs/metrics/traditional_ml_baselines_metrics.csv` | So sánh baseline truyền thống |
| `outputs/metrics/traditional_ml_tuning_results.csv` | Kết quả tuning mô hình truyền thống |
| `outputs/metrics/logistic_regression_test_metrics.json` | Metrics cuối cùng của Logistic Regression |
| `outputs/metrics/svm_hog_rgb_test_metrics.json` | Metrics cuối cùng của SVM + HOG + RGB |
| `outputs/metrics/simple_cnn_test_metrics.json` | Metrics của Simple CNN |
| `outputs/metrics/resnet18_finetuned_test_metrics.json` | Metrics của ResNet18 Fine-tuned |
| `outputs/metrics/cnn_model_comparison.csv` | So sánh các mô hình CNN-based |
| `outputs/metrics/final_model_comparison.csv` | So sánh tổng hợp các mô hình chính |

---

### File hình ảnh

| File | Mô tả |
|---|---|
| `outputs/figures/features/original_samples_grid.png` | Ảnh mẫu từ dataset |
| `outputs/figures/features/handcrafted_feature_comparison_panel.png` | So sánh ảnh gốc, grayscale, edge và HOG |
| `outputs/figures/features/color_histogram_examples.png` | Ví dụ RGB histogram |
| `outputs/figures/traditional_ml/confusion_matrix_logistic_regression.png` | Confusion matrix của Logistic Regression |
| `outputs/figures/traditional_ml/confusion_matrix_svm_hog_rgb.png` | Confusion matrix của SVM + HOG + RGB |
| `outputs/figures/deep_learning/training_curves_resnet18_finetuned.png` | Training curves của ResNet18 Fine-tuned |
| `outputs/figures/deep_learning/resnet18_finetuned_confusion_matrix.png` | Confusion matrix của ResNet18 Fine-tuned |
| `outputs/figures/deep_learning/per_class_f1_score_comparison.png` | So sánh F1-score theo từng class |
| `outputs/figures/gradcam/resnet18_finetuned_gradcam_correct_examples.png` | Grad-CAM cho ảnh dự đoán đúng |
| `outputs/figures/gradcam/resnet18_finetuned_gradcam_misclassified_examples.png` | Grad-CAM cho ảnh dự đoán sai |

---

## Giải thích và phân tích mô hình

Dự án sử dụng nhiều kỹ thuật để hiểu rõ hành vi của mô hình.

### Confusion Matrix

Confusion matrix giúp xác định các lớp thường bị nhầm lẫn với nhau.

Trong bài toán scene classification, một số lớp có thể chia sẻ đặc trưng thị giác tương tự:

- `mountain` và `glacier` có thể cùng chứa tuyết, đá, vùng sáng hoặc texture tự nhiên.
- `buildings` và `street` đều có yếu tố đô thị, đường thẳng và kiến trúc.
- `sea` và `glacier` có thể có vùng xanh, trắng hoặc ánh sáng mạnh.

---

### Top Confused Pairs

Top confused pairs giúp xác định cặp nhãn thật và nhãn dự đoán sai xuất hiện nhiều nhất.

Phân tích này trả lời các câu hỏi:

```text
Mô hình hay nhầm lớp nào với lớp nào?
Vì sao hai lớp đó dễ nhầm về mặt thị giác?
Lỗi đến từ màu sắc, texture, bố cục, ánh sáng hay đối tượng chính trong ảnh?
```

---

### Misclassified Examples

Quan sát ảnh bị dự đoán sai giúp đánh giá lỗi của mô hình có hợp lý hay không.

Một số nguyên nhân phổ biến:

- Ảnh chứa nhiều yếu tố của nhiều lớp khác nhau.
- Đối tượng chính không nổi bật.
- Ánh sáng hoặc độ tương phản bất thường.
- Hai lớp có màu sắc hoặc texture giống nhau.
- Bản thân ảnh có tính mơ hồ, ngay cả với con người.

---

### Grad-CAM

Grad-CAM được dùng cho các mô hình CNN-based để làm nổi bật vùng ảnh có ảnh hưởng mạnh đến dự đoán.

Grad-CAM giúp trả lời:

```text
Mô hình có nhìn đúng vùng quan trọng trong ảnh không?
Mô hình đang tập trung vào đối tượng chính hay nền ảnh?
Vì sao mô hình dự đoán đúng hoặc sai?
```

Kỹ thuật này giúp phần deep learning trở nên dễ giải thích hơn và cung cấp bằng chứng trực quan khi trình bày kết quả.

---

## Nhận xét chính

1. **Mô hình truyền thống phụ thuộc nhiều vào feature engineering**

   Logistic Regression, KNN và SVM không tự học đặc trưng ảnh như CNN. Hiệu quả của chúng phụ thuộc mạnh vào chất lượng đặc trưng đầu vào.

2. **HOG và RGB histogram bổ sung cho nhau**

   HOG mô tả shape, edge và texture; RGB histogram mô tả phân phối màu. Kết hợp hai loại đặc trưng giúp SVM có biểu diễn ảnh đầy đủ hơn.

3. **Simple CNN vượt qua nhiều baseline thủ công**

   CNN học đặc trưng trực tiếp từ ảnh, do đó phù hợp hơn với bài toán thị giác máy tính so với nhiều mô hình dùng đặc trưng thủ công.

4. **ResNet18 Fine-tuned là mô hình tốt nhất**

   Fine-tuning ResNet18 pretrained giúp tận dụng tri thức từ ImageNet và điều chỉnh đặc trưng cho bộ dữ liệu cảnh tự nhiên.

5. **Phân tích lỗi là phần không thể thiếu**

   Accuracy không đủ để đánh giá toàn diện. Confusion matrix, per-class metrics, top confused pairs, misclassified examples và Grad-CAM giúp hiểu sâu hơn hành vi của mô hình.

6. **Gradio demo giúp dự án có tính ứng dụng**

   Demo cho phép thử mô hình trực tiếp với ảnh mới, giúp kết quả đồ án dễ trình bày và dễ kiểm chứng hơn.

---

## Hạn chế

Dù pipeline đã khá đầy đủ, dự án vẫn còn một số hạn chế:

- Các mô hình truyền thống vẫn phụ thuộc vào đặc trưng thủ công.
- HOG + RGB histogram chưa thể biểu diễn tốt ngữ nghĩa cấp cao trong ảnh.
- Simple CNN có thể cần thêm dữ liệu, regularization hoặc augmentation mạnh hơn để cạnh tranh với pretrained model.
- Một số lớp cảnh có tính mơ hồ cao, đặc biệt khi ảnh chứa nhiều yếu tố pha trộn.
- Chưa thử nhiều kiến trúc hiện đại hơn như EfficientNet, ConvNeXt hoặc MobileNetV3.
- Demo Gradio chủ yếu phục vụ kiểm thử trực quan, chưa được đóng gói thành ứng dụng triển khai lâu dài.

---

## Hướng phát triển

Một số hướng cải thiện trong tương lai:

### 1. Thử nghiệm kiến trúc CNN mạnh hơn

- ResNet50
- EfficientNet
- MobileNetV3
- ConvNeXt

### 2. Cải thiện data augmentation

- Random crop
- Color jitter
- Random perspective
- CutMix
- MixUp

### 3. Nâng cấp demo và deployment

- Hoàn thiện Gradio demo.
- Thêm Streamlit demo.
- Xây dựng REST API cho dự đoán ảnh.
- Đóng gói model inference thành module riêng.

### 4. Cải thiện explainability

- Grad-CAM++
- Score-CAM
- Integrated Gradients

### 5. Phân tích lỗi sâu hơn

- Phân tích lỗi theo độ sáng ảnh.
- Phân tích lỗi theo dominant color.
- Phân tích lỗi theo confidence score.
- Gom nhóm các ảnh lỗi có đặc điểm thị giác tương tự.

---

## Kết luận

Dự án xây dựng một pipeline tương đối hoàn chỉnh cho bài toán phân loại ảnh cảnh tự nhiên trên bộ dữ liệu Intel Image Classification.

Dự án so sánh hai hướng tiếp cận: mô hình Machine Learning truyền thống dùng đặc trưng thủ công và mô hình Deep Learning học đặc trưng trực tiếp từ ảnh. Các thí nghiệm truyền thống cho thấy vai trò quan trọng của feature engineering, đặc biệt là HOG, RGB histogram và sự kết hợp HOG + RGB trong mô hình SVM. Các thí nghiệm Deep Learning cho thấy CNN-based models mạnh hơn trong bài toán nhận dạng ảnh, trong đó ResNet18 Fine-tuned đạt kết quả tốt nhất.

Bên cạnh accuracy, dự án còn có phần phân tích lỗi bằng per-class metrics, confusion matrix, top confused pairs, misclassified examples và Grad-CAM. Ngoài ra, notebook 09 bổ sung Gradio demo, giúp kiểm thử mô hình trực tiếp trên ảnh mới và tăng tính ứng dụng cho đồ án.

Dự án phù hợp để trình bày trong báo cáo học thuật, slide thuyết trình và GitHub portfolio. 

## Tác giả 
Dự án được xây dựng phục vụ mục đích học tập, thực nghiệm xử lý ảnh và so sánh các mô hình phân loại hình ảnh.

- Họ tên: Nguyễn Thành Long
- MSSV: 23520886
