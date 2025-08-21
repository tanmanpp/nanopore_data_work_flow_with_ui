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

def plot_coverage(input_file, fasta_file, output_file, title, style):
    # read coverage data
    df_cov = pd.read_csv(input_file, sep="\t", header=None, names=["chrom", "pos", "depth"])

    # read chromosome lengths from FASTA
    chrom_lengths = load_fasta_lengths(fasta_file)
    chroms = list(chrom_lengths.keys())
    rows = len(chroms)

    # subplot titles and dataframes
    subplot_titles = []
    chr_dfs = {}  # store full coverage data for each chromosome
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

    # create subplots
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
    )

    # plot each chromosome
    for i, chrom in enumerate(chroms):
        full = chr_dfs[chrom]
        zero_mask = full["depth"] == 0

        # black background for depth=0
        background = go.Scatter(
            x=full["pos"][zero_mask],
            y=[0] * zero_mask.sum(),
            mode="markers",
            marker=dict(color="black", size=1),
            showlegend=False,
        )

        # plot coverage
        if style == "line":
            trace = go.Scatter(
                x=full["pos"],
                y=full["depth"],
                mode="lines",
                name=chrom,
                line=dict(width=1)
            )
        elif style == "histo":
            trace = go.Bar(
                x=full["pos"],
                y=full["depth"],
                name=chrom
            )
        else:
            raise ValueError("❌ Invalid style. Use 'line' or 'histo'.")

        fig.add_trace(background, row=i + 1, col=1)
        fig.add_trace(trace, row=i + 1, col=1)
        fig.update_yaxes(title_text="Depth", row=i + 1, col=1)

        zero_regions = []
        start = None
        for idx, val in enumerate(full["depth"] == 0):
            pos = full["pos"].iloc[idx]
            if val and start is None:
                start = pos
            elif not val and start is not None:
                zero_regions.append((start, pos - 1))
                start = None
        if start is not None:
            zero_regions.append((start, full["pos"].iloc[-1]))

        for (start, end) in zero_regions:
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=start,
                y0=0,
                x1=end,
                y1=full["depth"].max() * 1.1,
                fillcolor="gray",
                opacity=0.4,
                layer="below",
                line_width=0,
                row=i + 1,
                col=1
            )

    # Layout settings
    fig.update_layout(
        autosize=True
        title_text=title,
        showlegend=False,
        template="plotly_white"
    )

    pio.write_html(
        fig,
        file=output_file,
        full_html=True,
        auto_open=False,
        config={"responsive": True},
        default_width="98vw",
        default_height="95vh"
    )
    print(f"✅ Coverage plot saved to: {output_file}")

# main function to handle command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot genome-wide coverage from samtools depth with missing regions in black.")
    parser.add_argument("-i", "--input", required=True, help="Input coverage file (from samtools depth)")
    parser.add_argument("-r", "--ref", required=True, help="Reference genome FASTA file")
    parser.add_argument("-o", "--output", default="genome_coverage.html", help="Output HTML file")
    parser.add_argument("--title", default="Genome-wide Coverage", help="Plot title")
    parser.add_argument("--style", choices=["line", "histo"], default="line", help="Plot style: 'line' or 'histo'")

    args = parser.parse_args()
    plot_coverage(args.input, args.ref, args.output, args.title, args.style)
