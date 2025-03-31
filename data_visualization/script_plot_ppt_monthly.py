'''python data_visualization/script_plot_ppt_monthly.py --station 1 --start_year 2015 --end_year 2020'''

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot monthly total precipitation (Ppt) per year.")
    parser.add_argument("--station",    "-s", type=int, required=True, help="Station number (1-6).")
    parser.add_argument("--start_year", "-y1", type=int, required=True, help="Start year.")
    parser.add_argument("--end_year",   "-y2", type=int, required=True, help="End year.")
    args = parser.parse_args()

    # Read csv file
    csv_path = Path("datasets") / "Revised_Final_Data" / f"Station{args.station}_Revised_Final_Data.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    # Ensure 'Ppt' exists
    if "Ppt" not in df.columns:
        print(f"Error: 'Ppt' column not found in {csv_path.name}.")
        return

    # Create output folder
    out_dir = Path("data_visualization") / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create PDF
    pdf_name = f"Ppt_Station{args.station}_{args.start_year}_{args.end_year}_monthly_totals.pdf"
    pdf_path = out_dir / pdf_name
    
    pdf = PdfPages(pdf_path)

    for year in range(args.start_year, args.end_year + 1):
        df_year = df[df.index.year == year]
        if df_year.empty:
            print(f"No data for year {year}, skipping...")
            continue

        # Group by month, sum ppt
        monthly_sums = df_year.groupby(df_year.index.month)["Ppt"].sum().dropna()
        if monthly_sums.empty:
            print(f"All Ppt values are NaN or zero in {year}, skipping...")
            continue

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(monthly_sums.index, monthly_sums.values, width=0.8, color="skyblue")

        ax.set_xticks(range(1, 13))
        ax.set_xlim(0.5, 12.5)
        ax.set_xlabel(f"Year {year}")
        ax.set_ylabel("Total Ppt (mm)")

        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"Saved PDF to: {pdf_path}")

if __name__ == "__main__":
    main()