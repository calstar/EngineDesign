from pathlib import Path
import pandas as pd

def create_plot_list():
    """Create a numbered list of all plots with metadata"""
    plots_dir = Path('data_plots')
    plot_files = sorted(list(plots_dir.glob('*_plot.png')))
    
    # Match with original CSV files to get row counts
    data_dir = Path('data')
    
    print("=" * 100)
    print("ALL PLOTS - SELECT WHICH TO KEEP")
    print("=" * 100)
    print(f"\n{'#':<4} {'Filename':<65} {'Rows':<8} {'Type':<20}")
    print("-" * 100)
    
    plot_info = []
    
    for i, plot_file in enumerate(plot_files, 1):
        # Extract CSV filename from plot filename
        csv_name = plot_file.name.replace('_plot.png', '.csv')
        csv_path = data_dir / csv_name
        
        # Get row count
        rows = 0
        try:
            df = pd.read_csv(csv_path)
            rows = len(df)
        except:
            pass
        
        # Categorize
        if 'pressure_data_filtered' in csv_name:
            file_type = 'Pressure (Filtered)'
        elif 'pressure_data_raw' in csv_name:
            file_type = 'Pressure (Raw)'
        elif 'pressure_data' in csv_name:
            file_type = 'Pressure'
        elif 'system_data' in csv_name:
            file_type = 'System'
        else:
            file_type = 'Other'
        
        print(f"{i:<4} {plot_file.name:<65} {rows:<8} {file_type:<20}")
        
        plot_info.append({
            'number': i,
            'filename': plot_file.name,
            'path': plot_file,
            'rows': rows,
            'type': file_type
        })
    
    print("\n" + "=" * 100)
    print("\nHOW TO SELECT:")
    print("  - Single numbers: 1, 5, 10")
    print("  - Ranges: 1-5, 20-25")
    print("  - Combined: 1, 3, 5-10, 15")
    print("  - Or type 'all' to keep all")
    print("=" * 100)
    
    # Save to file for reference
    with open('plot_list.txt', 'w') as f:
        f.write("PLOT LIST\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'#':<4} {'Filename':<65} {'Rows':<8} {'Type':<20}\n")
        f.write("-" * 100 + "\n")
        for info in plot_info:
            f.write(f"{info['number']:<4} {info['filename']:<65} {info['rows']:<8} {info['type']:<20}\n")
    
    print("\nList saved to 'plot_list.txt' for reference")
    
    return plot_info

if __name__ == "__main__":
    create_plot_list()






