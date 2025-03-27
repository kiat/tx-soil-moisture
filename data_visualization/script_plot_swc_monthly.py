'''python data_visualization/script_plot_swc_monthly.py --param SWC_10 --station 1 --start_year 2015 --end_year 2020'''

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot average monthly SWC in a bar chart.")
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
        print("Available columns:", df.columns.tolist())
        return

    # Create output directory
    out_dir = Path("data_visualization") / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create PDF
    pdf_name = f"{args.param}_Station{args.station}_{args.start_year}_{args.end_year}_monthly.pdf"
    pdf = PdfPages(out_dir / pdf_name)

    # compute monthly average for each year
    for year in range(args.start_year, args.end_year + 1):
        df_year = df[df.index.year == year]
        if df_year.empty:
            print(f"No data for year {year}, skipping...")
            continue

        # Group by month and compute mean
        df_monthly = df_year.groupby(df_year.index.month)[args.param].mean().dropna()

        # barplot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(df_monthly.index, df_monthly.values, width=0.8, color="skyblue")
        ax.set_xticks(range(1, 13))
        ax.set_xlim(0.5, 12.5)
        ax.set_xlabel("Month")
        ax.set_ylabel(args.param)
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"Saved PDF: {pdf_name}")

if __name__ == "__main__":
    main()