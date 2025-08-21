# NGS(ONT) Mapping UI

A user-friendly web interface for non-specialists to run **NGS(ONT) read mapping** against reference genomes.  
The system combines a **FastAPI backend** with an **HTML/JS frontend**, allowing users to launch pipelines easily on Windows (via WSL).

---

## Features
- Web-based UI (HTML) for simple workflow control.
- Backend powered by **FastAPI** (`main.py`).
- Supports read mapping workflows ([minimap2](https://github.com/lh3/minimap2), [samtools](https://github.com/samtools/samtools), [bcftools](https://github.com/samtools/bcftools), etc.).
- Organized output directories for each processing step.
- Ready-to-use script (`run_ui.py`) to start the backend and access the UI.

---

## Requirements
- **Windows + WSL** (tested with Ubuntu on WSL2).
- **Conda environment** with required bioinformatics tools installed.
- Python ≥ 3.9

## Installation

### 1. Clone this repository
```bash
git clone https://github.com/tanmanpp/nanopore_data_work_flow_with_ui.git
cd ngs-mapping-ui
```

### 2. Conda environments
Two environment files are provided:
- `environment.yml` → standard environment for UI and backend.
- `environment_rcf.yml` → Environment for [rcf](https://github.com/khyox/recentrifuge).

Create the environment:
```bash
conda env create -f environment.yml
conda env create -f environment_rcf.yml
conda activate ngs_ui
```

## 3. Usage
### Launch the backend and UI

From WSL:
```bash
conda activate ngs_ui
python run_ui.py
```
This will:

Start the FastAPI backend using uvicorn.

Detect your WSL IP using hostname -I.

Print two URLs (English / Chinese) for direct access.

Example output:
```java
✅ Service is running
🔎 Detected IP: 172.**.**.**

— UI entry (English / default):
   http://172.**.**.**:8000/ui/v1.3_20250812_en.html

— UI entry (Chinese):
   http://172.**.**.**:8000/ui/v1.3_20250812_zh.html
```
## Then, just copy one of the URLs and paste it into your Windows browser. Enjoy :)


