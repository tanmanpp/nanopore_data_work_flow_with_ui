
import pysam
import argparse
import os
import re
from colorama import Fore, Back, Style

# ----------------------------
# 解析 mapping_sorted.coverage → TSV
# ----------------------------
RE_HEADER = re.compile(r"^(\S+)\s+\(([^)]+)\)\s*$")  # e.g. PV989858.1_VP4 (2.4Kbp)
RE_NUM_READS = re.compile(r"Number of reads:\s*([0-9,]+)\s*$")
RE_COVERED_BASES = re.compile(r"Covered bases:\s*([0-9.]+[KMG]?bp)\s*$", re.IGNORECASE)
RE_PERCENT = re.compile(r"Percent covered:\s*([0-9.]+)\s*%\s*$", re.IGNORECASE)
RE_MEAN_COV = re.compile(r"Mean coverage:\s*([0-9.+\-eE]+)\s*x\s*$", re.IGNORECASE)

def export_coverage_tsv(coverage_fp: str, out_tsv_fp: str):
    """
    讀取 samtools coverage -m 產生的文字報表（ASCII 報表格式），
    匯出每個 segment 的 coverage_percent 與 mean_depth_x 到 TSV。
    會同時輸出 n_reads / covered_bases / length_str 方便 QC。
    """
    if not os.path.exists(coverage_fp):
        raise FileNotFoundError(f"coverage file not found: {coverage_fp}")

    stats = {}
    current = None

    with open(coverage_fp, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # 1) 新段落 header（chrom/segment）
            m = RE_HEADER.match(line)
            if m:
                chrom = m.group(1)
                length_str = m.group(2)
                current = chrom
                stats.setdefault(chrom, {})
                stats[chrom]["length_str"] = length_str
                continue

            if current is None:
                continue

            # 2) 段落內資訊
            m = RE_NUM_READS.search(line)
            if m:
                stats[current]["n_reads"] = int(m.group(1).replace(",", ""))
                continue

            m = RE_COVERED_BASES.search(line)
            if m:
                stats[current]["covered_bases"] = m.group(1)
                continue

            m = RE_PERCENT.search(line)
            if m:
                stats[current]["coverage_percent"] = float(m.group(1))
                continue

            m = RE_MEAN_COV.search(line)
            if m:
                stats[current]["mean_depth_x"] = float(m.group(1))
                continue

    # 輸出 TSV
    cols = ["chrom", "coverage_percent", "mean_depth_x", "n_reads", "covered_bases", "length_str"]
    with open(out_tsv_fp, "w", encoding="utf-8", newline="\n") as out:
        out.write("\t".join(cols) + "\n")
        for chrom in sorted(stats.keys()):
            rec = stats[chrom]
            row = [
                chrom,
                "" if "coverage_percent" not in rec else str(rec["coverage_percent"]),
                "" if "mean_depth_x" not in rec else str(rec["mean_depth_x"]),
                "" if "n_reads" not in rec else str(rec["n_reads"]),
                "" if "covered_bases" not in rec else str(rec["covered_bases"]),
                "" if "length_str" not in rec else str(rec["length_str"]),
            ]
            out.write("\t".join(row) + "\n")

    return len(stats)

# create an argument parser
parser = argparse.ArgumentParser(description='Entry fastaq files to mapping virus genome')

# add arguments for the input files and parameters
parser.add_argument('-f1', help='fastaq R1 file')
#parser.add_argument('-f2', help='fastaq R2 file')
parser.add_argument('-VG', help='Virus genome file.')
parser.add_argument("-mt", help="Mapping method, minimap2.", default="minimap2", type=str)
parser.add_argument("-t", help="Number of Threads", default=20, type=int)
parser.add_argument("-sid", help="Sample id for depth plot", type=str)

# add an argument for the output directory
parser.add_argument('-oD', help='Output directory for results')

# parse the arguments
args = parser.parse_args()
sample_id = args.sid
fastaq_1 = os.path.abspath(args.f1)
virus_genome = os.path.abspath(args.VG)
output_directory = args.oD
threads = args.t
method = args.mt

# check if the input files exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_directory_ab = os.path.abspath(output_directory)

# start mapping process
print(Back.GREEN + f"Mapping method is {method}." + Style.RESET_ALL)
if method == "minimap2":
    os.system(f"minimap2 -ax map-ont -t {threads} {virus_genome} {fastaq_1} > {output_directory_ab}/mapping.sam")
else:
    print(Fore.RED + f"Unsupported mapping method: {method}" + Style.RESET_ALL)
    exit(1)

# sort the SAM file and generate coverage and depth statistics
print(Back.GREEN + "Doing samtools sort..." + Style.RESET_ALL)
os.system(f"samtools sort {output_directory_ab}/mapping.sam > {output_directory_ab}/mapping_sorted.bam")
os.system(f"rm {output_directory_ab}/mapping.sam")

print(Back.GREEN + "Doing samtools coverage and depth..." + Style.RESET_ALL)
os.system(f"samtools depth {output_directory_ab}/mapping_sorted.bam > {output_directory_ab}/mapping_sorted.depth")
os.system(f"samtools index {output_directory_ab}/mapping_sorted.bam")
os.system(f"samtools coverage -m {output_directory_ab}/mapping_sorted.bam > {output_directory_ab}/mapping_sorted.coverage")

# ✅ 新增：把 mapping_sorted.coverage 匯出成 TSV（不加 args）
try:
    cov_fp = f"{output_directory_ab}/mapping_sorted.coverage"
    cov_tsv_fp = f"{output_directory_ab}/mapping_sorted.coverage.tsv"
    n_seg = export_coverage_tsv(cov_fp, cov_tsv_fp)
    print(Back.GREEN + f"[OK] Exported coverage TSV ({n_seg} segments): {cov_tsv_fp}" + Style.RESET_ALL)
except Exception as e:
    print(Fore.YELLOW + f"[WARN] Export coverage TSV failed: {e}" + Style.RESET_ALL)
    
script_dir = os.path.dirname(os.path.abspath(__file__))
plot_script = os.path.join(script_dir, "plot_genome_coverage.py")
os.system(f"python {plot_script} -i {output_directory_ab}/mapping_sorted.depth -r {virus_genome} -o  {output_directory_ab}/{sample_id}_mapping_sorted.depth.html --style histo --title {sample_id}")

# generate consensus sequence using bcftools
print(Back.GREEN + "Doing bcftools consensus series..." + Style.RESET_ALL)
os.system(f"bcftools mpileup -O v -x --threads {threads} -d 200000 -L 200000 -f {virus_genome} {output_directory_ab}/mapping_sorted.bam -o {output_directory_ab}/sort.raw.vcf")
os.system(f"bcftools call {output_directory_ab}/sort.raw.vcf -m --ploidy 1 -o {output_directory_ab}/sort.vcf")
os.system(f"bcftools norm -f {virus_genome} {output_directory_ab}/sort.vcf -O z > {output_directory_ab}/sort.vcf.gz")
os.system(f"bcftools index {output_directory_ab}/sort.vcf.gz")
os.system(f"bcftools consensus -a N --mark-del - -i 'QUAL>20' -f {virus_genome} {output_directory_ab}/sort.vcf.gz > {output_directory_ab}/{sample_id}_consensus.fasta")
os.system(f"awk -v id={sample_id}_consensus_from '/^>/ {{split($0,a,\" \"); print \">\"id\"_\"substr(a[1],2); next}} {{print}}' {output_directory_ab}/{sample_id}_consensus.fasta > {output_directory_ab}/tmp.fasta && mv {output_directory_ab}/tmp.fasta {output_directory_ab}/{sample_id}_consensus.fasta")
print(Back.GREEN + "Done" + Style.RESET_ALL)

