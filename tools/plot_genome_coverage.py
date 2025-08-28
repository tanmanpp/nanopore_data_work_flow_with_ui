import pandas as pd
import argparse
from Bio import SeqIO
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

def load_fasta_lengths(fasta_file):
    chrom_lengths = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        chrom_lengths[record.id] = len(record.seq)
    return chrom_lengths

def contiguous_zero_regions(full_df):
    """Return list of (start, end) intervals where depth == 0"""
    zero = (full_df["depth"] == 0).values
    pos = full_df["pos"].values
    regions = []
    start = None
    for i, is_zero in enumerate(zero):
        if is_zero and start is None:
            start = pos[i]
        elif (not is_zero) and (start is not None):
            regions.append((start, pos[i-1]))
            start = None
    if start is not None:
        regions.append((start, pos[-1]))
    return regions

def plot_coverage(input_file, fasta_file, output_file, title, style,
                  row_height_px=120, vspace_px=16, lock_y=False):
    # 讀取 samtools depth 輸出
    df_cov = pd.read_csv(input_file, sep="\t", header=None, names=["chrom", "pos", "depth"])

    # 取得 chr 長度
    chrom_lengths = load_fasta_lengths(fasta_file)
    chroms = list(chrom_lengths.keys())
    rows = len(chroms)
    if rows == 0:
        raise ValueError("No chromosomes found in FASTA.")

    # 子圖標題與每條 chr 的完整 coverage 表
    subplot_titles = []
    chr_dfs = {}
    per_chr_max = []

    for chrom in chroms:
        chr_len = chrom_lengths[chrom]
        chr_cov = df_cov[df_cov["chrom"] == chrom]

        full = pd.DataFrame({"pos": range(1, chr_len + 1)})
        full["chrom"] = chrom
        full = full.merge(chr_cov, on=["chrom", "pos"], how="left")
        full["depth"] = full["depth"].fillna(0)

        cov_ratio = (full["depth"] > 0).sum() / chr_len * 100
        mean_depth = full["depth"].mean()
        subplot_titles.append(f"{chrom} | Coverage: {cov_ratio:.1f}% | Mean depth: {mean_depth:.1f}")

        chr_dfs[chrom] = full
        per_chr_max.append(full["depth"].max())

    # 計算統一的 figure 高度（像素）與 vertical_spacing（相對）
    # 讓每個 row 皆為等高（相對高度 = 1/rows），真正像素高度由 layout.height 決定
    top_margin = 70
    bottom_margin = 50
    # 子圖總高度 + 間距總高度 + 上下邊界
    fig_height = rows * row_height_px + (rows - 1) * vspace_px + top_margin + bottom_margin
    # Plotly 的 vertical_spacing 是相對於可用繪圖高度（不含邊界），這裡做近似換算
    plot_area_height = max(1, fig_height - top_margin - bottom_margin)
    vertical_spacing = vspace_px / plot_area_height

    # 建立子圖
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        row_heights=[1.0 / rows] * rows,
    )

    # y 軸上限
    if lock_y:
        global_ymax = max(per_chr_max) if per_chr_max else 1
        # 避免全零資料的情況
        global_ymax = max(global_ymax, 1)
        global_ymax *= 1.2  # 留些頭部空間

    # 繪圖
    for i, chrom in enumerate(chroms, start=1):
        full = chr_dfs[chrom]
        zero_mask = full["depth"] == 0

        # 背景零點（不影響 hover）
        background = go.Scatter(
            x=full["pos"][zero_mask],
            y=[0] * int(zero_mask.sum()),
            mode="markers",
            marker=dict(color="black", size=1),
            showlegend=False,
            hoverinfo="skip",
        )

        # coverage 主 trace
        if style == "line":
            trace = go.Scatter(
                x=full["pos"],
                y=full["depth"],
                mode="lines",
                name=chrom,
                line=dict(width=1, color="#003366"),
                hovertemplate="pos=%{x}<br>depth=%{y}<extra></extra>",
            )
        elif style == "histo":
            trace = go.Bar(
                x=full["pos"],
                y=full["depth"],
                name=chrom,
                marker=dict(color="#003366"),
                hovertemplate="pos=%{x}<br>depth=%{y}<extra></extra>",
            )
        else:
            raise ValueError("❌ Invalid style. Use 'line' or 'histo'.")

        fig.add_trace(background, row=i, col=1)
        fig.add_trace(trace, row=i, col=1)

        # y 軸範圍：鎖定或各自最大
        if lock_y:
            ymax = global_ymax
        else:
            ymax = max(full["depth"].max(), 1) * 1.2

        fig.update_yaxes(range=[0, ymax], title_text="Depth", row=i, col=1)

        # 標記 depth=0 的連續區段（灰色半透明）
        for (z0, z1) in contiguous_zero_regions(full):
            fig.add_shape(
                type="rect",
                xref=f"x{i}",
                yref=f"y{i}",
                x0=z0, y0=0,
                x1=z1, y1=ymax,
                fillcolor="gray",
                opacity=0.35,
                layer="below",
                line_width=0,
            )

    # 版面設定
    fig.update_layout(
        autosize=False,
        height=fig_height,
        title_text=title,
        showlegend=False,
        template="plotly_white",
        margin=dict(t=top_margin, b=bottom_margin, l=60, r=20),
        bargap=0.0 
    )

    pio.write_html(
        fig,
        file=output_file,
        full_html=True,
        auto_open=False,
        config={"responsive": True},
    )
    print(f"✅ Coverage plot saved to: {output_file}")

# 主程式
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot genome-wide coverage from samtools depth with missing regions in gray, with fixed row height and spacing."
    )
    parser.add_argument("-i", "--input", required=True, help="Input coverage file (from samtools depth)")
    parser.add_argument("-r", "--ref", required=True, help="Reference genome FASTA file")
    parser.add_argument("-o", "--output", default="genome_coverage.html", help="Output HTML file")
    parser.add_argument("--title", default="Genome-wide Coverage", help="Plot title")
    parser.add_argument("--style", choices=["line", "histo"], default="line", help="Plot style: 'line' or 'histo'")
    parser.add_argument("--row-height-px", type=int, default=120, help="Per-trace (subplot) height in pixels (default: 120)")
    parser.add_argument("--vspace-px", type=int, default=50, help="Spacing between subplots in pixels (default: 16)")
    parser.add_argument("--lock-y", action="store_true", help="Use a shared y-axis max across all subplots")

    args = parser.parse_args()
    plot_coverage(
        args.input, args.ref, args.output, args.title, args.style,
        row_height_px=args.row_height_px,
        vspace_px=args.vspace_px,
        lock_y=args.lock_y
    )
