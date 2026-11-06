# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Python: 3.13.16 / PyTorch: 2.11.0+cu130 / Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
  {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  },

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh vì đây là tác vụ phân loại ảnh. Model dự đoán lớp cab cho toàn ảnh, không xác định vị trí hay gán nhãn cho từng chiếc xe riêng lẻ

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách lớp có sẵn trong taxonomy đi kèm checkpoint. Trong JSON này, checkpoint yolo11n-cls.pt sử dụng taxonomy ImageNet-1K; model chọn lớp từ danh sách đó để đưa ra kết quả

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Do class_id là mã định danh lớp, class_name giúp người đọc hiểu tên lớp, còn taxonomy_name cho biết lớp thuộc hệ phân loại nào. Ghi đủ ba trường giúp tránh nhầm lẫn giữa các bộ nhãn

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần phải quy định cách chọn 1 lớp đại diện cho ảnh, chẳng hạn chọn theo chủ thể chính hoặc thứ tự ưu tiên giữa các lớp. Ngoài ra, cần phải hướng dẫn xử lý khi không xác định được chủ thể chính.

6. Vì sao không được lấy `score` của model làm ground truth?
Score chỉ thể hiện mức tin cậy của model, không bảo đảm dự đoán đúng. Ví dụ, score 0.510915 của lớp cab không chứng minh đó là nhãn đúng của ảnh. Ground truth phải được người gán nhãn xác định và kiểm tra theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
  {
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 0,
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  },

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Box có góc trên-trái tại (385.33, 69.24), góc dưới-phải tại (498.92, 348.92), chiều rộng của box là 113.59 pixel và chiều cao của box là 279.68 pixel.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Ngưỡng 0.20 có 17 prediction
- Ngưỡng 0.60 có 6 prediction

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng, reviewer phải kiểm tra thêm các box có score thấp được giữ lại, xác minh có đúng vật thể không, kiểm tra tên lớp và độ sát của box, đồng thời loại các box sai hoặc trùng

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Mỗi vật thể được gán một box hình chữ nhật ôm sát phần nhìn thấy, không vượt ra ngoài ảnh và không tự mở rộng để đoán phần bị che khuất

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Ở mép trái ảnh kitchen, có một người chỉ nhìn thấy một phần cánh tay và bàn tay, phần còn lại nằm ngoài khung ảnh. Cần hỏi Lab Coach để xác nhận phần nhìn thấy như vậy có đủ điều kiện gán nhãn person hay không, vì chưa có quy định về mức độ cắt mép được chấp nhận. Sau khi được xác nhận, cách xử lý nên được ghi vào guideline để áp dụng thống nhất cho các ảnh khác

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
{
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "kitchen-001",
    "class_id": 0,
    "class_name": "person",
    "score": 0.899318,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.45,
      66.44,
      498.02,
      348.58
    ],
    "polygon_point_count": 348,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ],

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon mô tả đường biên và hình dạng của vật thể chi tiết hơn so với box.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để phân biệt từng đối tượng trong ảnh, chẳng hạn kitchen-001 xác định một người cụ thể trong kết quả này. Nó khác class_id, vì nhiều người có thể cùng thuộc lớp person nhưng có instance_id khác nhau. Nó cũng không phải tracking ID vì không bảo đảm nhận diện cùng người đó qua nhiều khung hình.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Vẽ mask bám sát đường biên phần vật thể nhìn thấy, không lấn sang nền hoặc vật bên cạnh và không tự suy đoán phần bị che khuất.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Ở mép trái ảnh kitchen, một người chỉ lộ một phần cánh tay và bàn tay phía trên bàn, phần còn lại nằm ngoài ảnh hoặc bị che. Chọn escalation và hỏi Lab Coach xem phần nhìn thấy có đủ điều kiện gán mask person hay không, vì chưa có quy định về mức độ hiển thị tối thiểu. Sau khi được xác nhận, cần bổ sung cách xử lý vào guideline để áp dụng thống nhất.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn ảnh, được xác định theo guideline | Ảnh traffic có nhiều loại phương tiện; JSON dự đoán cab hạng 1 với score 0.510915, nên khó chọn một lớp đại diện nếu thiếu quy tắc | Chọn một lớp đại diện theo guideline; hỏi Lab Coach khi chưa rõ | Kiểm tra lớp được chọn có phù hợp với ảnh và quy tắc chọn chủ thể chính không |
| Phát hiện vật thể | Một nhãn lớp và box cho từng vật thể đủ điều kiện gán nhãn | Trong ảnh kitchen, người ở mép trái chỉ lộ một phần cánh tay và bàn tay, khó quyết định có gán box person hay không | Gán lớp và vẽ box theo guideline; hỏi Lab Coach về trường hợp cắt mép chưa được quy định | Kiểm tra sai lớp, bỏ sót, box thừa hoặc trùng và độ sát của box |
| Instance segmentation | Một nhãn lớp và mask riêng cho từng đối tượng đủ điều kiện gán nhãn | Người ở mép trái ảnh kitchen chỉ hiện một phần; khó xác định phạm vi mask và điều kiện được gán nhãn | Vẽ mask sát phần nhìn thấy theo guideline; không tự đoán phần bị che | Kiểm tra biên mask, vùng lấn sang nền, bỏ sót phần nhìn thấy và việc tách riêng từng đối tượng |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng dữ liệu được cấp cho bài lab, không đưa họ tên, MSSV, email, số điện thoại hoặc dữ liệu nhạy cảm vào báo cáo và output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi bài lab, tôi sẽ dừng xử lý và báo cho Lab Coach hoặc giảng viên phụ trách để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
