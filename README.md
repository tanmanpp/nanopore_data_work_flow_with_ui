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

## Installation

### I. Clone this repository
```bash
git clone https://github.com/tanmanpp/nanopore_data_work_flow_with_ui.git
cd ngs-mapping-ui
```

### II. Conda environments
Two environment files are provided:
- `environment.yml` → standard environment for UI and backend.
- `environment_rcf.yml` → Environment for [rcf](https://github.com/khyox/recentrifuge).

Create the environment:
```bash
conda env create -f environment.yml
conda env create -f environment_rcf.yml
```
### III. Download Kraken2 (K2) database
To run classification, you must download a pre-built K2 database.
We recommend the official AWS-hosted indexes:

👉 [AWS Kraken2 Indexes Database](https://benlangmead.github.io/aws-indexes/k2)

### IV. Install Dorado
Dorado is required for basecalling and demultiplexing.
Download the latest release (example: 1.1.1):
```bash
wget https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.1.1-linux-x64.tar.gz
tar -xvzf dorado-1.1.1-linux-x64.tar.gz
```
Add Dorado binary to your PATH (adjust version if different):

```bash
export PATH=$PATH:$HOME/dorado-1.1.1-linux-x64/bin
```
To make this permanent, add the line above to your ~/.bashrc or ~/.zshrc.
Verify installation:

```bash
dorado --version
```


## Usage
### Launch the backend and UI

From WSL:
```bash
conda activate ngs_ui
python run_serve.py
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


