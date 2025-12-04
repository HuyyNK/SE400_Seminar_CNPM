# Module 2 Reports - README

Thư mục này chứa **3 báo cáo toàn diện** về Module 2: Baseline Deep Learning Models.

---

## 📄 Danh sách Báo cáo

### 1. **MODULE2_COMPREHENSIVE_REPORT.md** 📊
**Báo cáo đầy đủ và chi tiết nhất**

**Nội dung:**
- Tổng quan Module 2 (mục tiêu, dataset, yêu cầu đặc tả)
- Kiến trúc CNN và BiLSTM chi tiết (layer-by-layer)
- Quá trình training (11 epochs CNN, 14 epochs BiLSTM)
- Kết quả evaluation đầy đủ (AUC, F1, Precision, Recall)
- So sánh CNN vs BiLSTM (5/6 labels BiLSTM thắng)
- Phân tích optimal thresholds
- Error analysis (false positives/negatives)
- Khuyến nghị deployment (BiLSTM recommended)
- Lessons learned & next steps

**Độ dài:** ~200 pages (nếu in ra)  
**Audience:** Technical team, researchers  
**Use case:** Deep dive vào model performance

---

### 2. **EXECUTIVE_SUMMARY.md** 📋
**Báo cáo tóm tắt cho quản lý và stakeholders**

**Nội dung:**
- Kết quả chính (CNN AUC 0.9796, BiLSTM AUC 0.9832)
- So sánh trực quan CNN vs BiLSTM
- Khuyến nghị production (BiLSTM với optimal thresholds)
- Key metrics at a glance
- Deliverables checklist (100% complete)
- Next steps (Module 3 preparation)

**Độ dài:** ~10 pages  
**Audience:** Managers, stakeholders  
**Use case:** Quick overview, decision making

---

### 3. **CNN_vs_BiLSTM_COMPARISON.md** 📈
**Báo cáo so sánh trực quan với charts và biểu đồ**

**Nội dung:**
- Biểu đồ ASCII so sánh hiệu suất
- Per-label F1-score comparison (visual bars)
- Training efficiency comparison
- Precision vs Recall trade-offs
- Error analysis highlights
- Use case recommendations
- Final verdict (BiLSTM wins 5/6 labels)

**Độ dài:** ~15 pages  
**Audience:** Technical + non-technical  
**Use case:** Visual presentation, quick comparison

---

## 🎯 Cách Sử dụng

### Nếu bạn cần:

**1. Hiểu sâu về models:**
→ Đọc `MODULE2_COMPREHENSIVE_REPORT.md`

**2. Trình bày cho quản lý:**
→ Đọc `EXECUTIVE_SUMMARY.md`

**3. So sánh trực quan CNN vs BiLSTM:**
→ Đọc `CNN_vs_BiLSTM_COMPARISON.md`

**4. Tất cả:**
→ Đọc theo thứ tự: Executive Summary → Comparison → Comprehensive

---

## 📊 Key Findings (TL;DR)

### Module 2 Status: ✅ **HOÀN THÀNH 100%**

**Models trained:**
- ✅ CNN: AUC 0.9796, F1 0.5607
- ✅ BiLSTM: AUC 0.9832, F1 0.5843

**Winner:** 🏆 **BiLSTM**
- +4.2% F1 macro
- +24% identity_hate F1
- -22% validation loss
- Wins 5/6 labels

**Recommendation:** 
- **Production:** BiLSTM with optimal thresholds
- **High-volume:** Hybrid CNN→BiLSTM pipeline
- **Speed-critical:** CNN only

**Next:** Module 3 - Transformer models (BERT/RoBERTa)

---

## 📁 File Structure

```
Module2_DL/reports/
├── README.md                           (this file)
├── MODULE2_COMPREHENSIVE_REPORT.md     (detailed report)
├── EXECUTIVE_SUMMARY.md                (management summary)
└── CNN_vs_BiLSTM_COMPARISON.md         (visual comparison)
```

---

## 🔗 Related Files

**Models:**
```
Module2_DL/artifacts/
├── toxic_cnn_model.h5              (61 MB)
├── toxic_bilstm_model.h5           (59 MB)
├── config_cnn.json
├── config_bilstm.json
├── tokenizer_cnn.json
├── tokenizer_bilstm.json
├── history_cnn.json
├── history_bilstm.json
├── report_baseline_dl_cnn.json     (evaluation metrics)
├── report_baseline_dl_bilstm.json  (evaluation metrics)
├── learning_curves_cnn.png
├── learning_curves_bilstm.png
├── pr_curves_cnn.png
└── pr_curves_bilstm.png
```

**Code:**
```
Module2_DL/src/
├── preprocess.py       (text preprocessing)
├── models.py           (CNN & BiLSTM architectures)
├── train.py            (training pipeline)
├── evaluate.py         (evaluation with optimal thresholds)
└── infer.py            (production inference)
```

---

## 📈 Performance Highlights

### Overall Metrics (Test Set)

| Metric | CNN | BiLSTM | Winner |
|--------|-----|--------|--------|
| **AUC** | 0.9796 | **0.9832** | BiLSTM (+0.36%) |
| **F1 Macro** | 0.5607 | **0.5843** | BiLSTM (+4.21%) |
| **F1 Micro** | 0.6831 | **0.7210** | BiLSTM (+5.55%) |

### Per-Label F1 (Test Set)

| Label | CNN | BiLSTM | Winner |
|-------|-----|--------|--------|
| toxic | 0.7861 | **0.8100** | BiLSTM (+3.0%) |
| severe_toxic | 0.4725 | **0.4963** | BiLSTM (+5.0%) |
| obscene | 0.8062 | **0.8117** | BiLSTM (+0.7%) |
| **threat** | **0.2258** | 0.2071 | CNN (+9.0%) |
| insult | 0.7075 | **0.7269** | BiLSTM (+2.7%) |
| identity_hate | 0.3659 | **0.4536** | BiLSTM (+24.0%) 🎯 |

---

## ✅ Module 2 Checklist

**Requirements từ đặc tả:**

- [x] Keras/TensorFlow implementation
- [x] Embedding layer (GloVe 300d, frozen)
- [x] Lựa chọn A: Conv1D + GlobalMaxPooling ✅
- [x] Lựa chọn B: Bidirectional LSTM ✅
- [x] Classification layers (Dense + Dropout)
- [x] 6 sigmoid outputs (multi-label)
- [x] binary_crossentropy loss + Adam optimizer
- [x] Validation set + EarlyStopping
- [x] Model files (.h5) exported
- [x] Performance reports (AUC, F1, Precision, Recall)

**Completion: 100%** ✅

---

## 🎓 Key Learnings

### Technical:
- ✅ Multi-kernel CNN captures n-gram patterns well
- ✅ Bidirectional LSTM crucial for context (+5-10%)
- ✅ EarlyStopping prevents overfitting effectively
- ✅ Optimal thresholds improve rare class F1 by 10-20%

### Data Science:
- ⚠️ Class imbalance is challenging (threat 0.3% only)
- ✅ Context understanding matters (BiLSTM +24% identity_hate)
- ⚠️ Sarcasm detection still difficult
- ✅ Validation set essential

### Engineering:
- ✅ Clean modular code → Easy to extend
- ✅ Comprehensive logging → Reproducible
- ✅ Artifact management → Production ready
- ✅ Evaluation framework → Fair comparison

---

## 🚀 Next Steps

### Immediate:
1. ✅ Deploy BiLSTM model to production
2. ✅ Use optimal thresholds from report
3. ✅ Monitor inference performance

### Module 3 Preparation:
**Baseline to beat:**
- AUC: 0.9832 (BiLSTM)
- F1 macro: 0.5843
- Identity_hate F1: 0.4536

**Expected Transformer improvements:**
- AUC: +1-2% → 0.985-0.995
- F1: +5-10% → 0.61-0.64
- Identity_hate: +10-15% → 0.50-0.55

**Trade-off:**
- Speed: 15ms → 50-200ms (slower)
- Model size: 59MB → 400-500MB (larger)

---

## 📞 Contact

**Project:** SE405 - Seminar CNPM  
**Module:** 2 - Baseline Deep Learning Models  
**Date:** November 27, 2025  
**Status:** ✅ Complete

---

**Happy Reading! 📖**
