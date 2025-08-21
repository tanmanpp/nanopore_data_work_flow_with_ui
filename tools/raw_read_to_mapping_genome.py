
import pysam
import argparse
import os
from colorama import Fore, Back, Style

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
