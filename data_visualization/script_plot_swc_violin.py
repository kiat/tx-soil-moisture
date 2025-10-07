'''python data_visualization/script_plot_swc_violin.py --param SWC_10 --station 1 --start_year 2015 --end_year 2020'''

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Violin plots of yearly SWC distributions.")
    parser.add_argument("--param",      "-p", type=str, required=True, help="SWC param, e.g. SWC_10")
    parser.add_argument("--station",    "-s", type=int, required=True, help="Station number (1-6)")
    parser.add_argument("--start_year", "-y1", type=int, required=True, help="Start year")
    parser.add_argument("--end_year",   "-y2", type=int, required=True, help="End year")
    args = parser.parse_args()

    # Read csv file
    csv_path = Path("datasets") / "Revised_Final_Data" / f"Station{args.station}_Revised_Final_Data.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    # Check if param exists
    if args.param not in df.columns:
        print(f"Column '{args.param}' not found in {csv_path.name}.")
        return

    # Create output folder
    out_dir = Path("data_visualization") / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create PDF
    pdf_name = f"{args.param}_Station{args.station}_{args.start_year}_{args.end_year}_violin.pdf"
    pdf_path = out_dir / pdf_name

    pdf = PdfPages(pdf_path)

    for year in range(args.start_year, args.end_year + 1):
        df_year = df[df.index.year == year]
        if df_year.empty:
            print(f"Skipping {year} - no data for {args.param}.")
            continue

        # Drop NaN to avoid empty violin
        data_year = df_year[args.param].dropna()
        if data_year.empty:
            print(f"Skipping {year} - all {args.param} are NaN.")
            continue

        # Create violin plot
        fig, ax = plt.subplots(figsize=(4, 7))
        ax.violinplot(dataset=[data_year.values], showmeans=False, showmedians=True)

        # Remove x ticks 
        ax.set_xticks([])
        ax.set_xlabel(f"Year {year}")
        ax.set_title(f"{args.param}", fontsize=10)

        # Save to PDF
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"Saved PDF to: {pdf_path}")

if __name__ == "__main__":
    main()
