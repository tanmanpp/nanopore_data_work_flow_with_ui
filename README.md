# 🧬 NGS Reads Mapping UI

本專案提供一個前端 HTML + 後端 FastAPI 服務，讓非專業的生物資訊使用者也能輕鬆完成 NGS 讀長（reads）比對到對應基因體的流程，並自動產出覆蓋度圖與共識序列(Consensus sequence)。

---

## 📥 下載專案

```bash
git clone https://github.com/<你的帳號>/<你的專案名>.git
cd <你的專案名>
```

📦 安裝與環境建立
本專案建議使用 conda 建立兩個獨立環境：

  ngs_mapping：主分析流程（後端、UI 分析）

  ngs_rcf：Recentrifuge（避免衝突）

  1️⃣ 建立主分析環境
  
```bash
conda env create -f environment.yml
```

2️⃣ 建立 RCF 專用環境
```
conda env create -f environment_rcf.yml
```

⚙️ 流程功能
1. **Dorado Basecalling**
   - 支援 GPU / CPU
   - 輸出 FASTQ

2. **Demultiplex（demux）**
   - 按 barcode 分樣

3. **Reads 修剪與品質檢查**
   - Porechop 修剪
   - NanoPlot 視覺化品質報告

4. **分類分析**
   - Kraken2 分類
   - Recentrifuge 視覺化

5. **比對與共識序列生成**
   - Minimap2 比對到參考基因體
   - samtools / bcftools 分析
   - 自動繪製 genome coverage 圖


🚀 使用方式
啟動後端 API

```bash
conda activate ngs_mapping
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 開啟前端 UI

直接點開：

- 中文介面：`v1.3_20250812_zh.html`
- 英文介面：`v1.3_20250812_en.html`

> 前端會呼叫後端 API 進行分析，請確保後端服務已啟動。



