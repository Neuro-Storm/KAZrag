import os
import json
import pandas as pd
from pathlib import Path

def combine_evaluation_results(results_dir="../evaluation_results", output_csv="../evaluation_results_final.csv", output_excel="../evaluation_results_final.xlsx"):
    """
    Combines JSON files from evaluation_results directory into CSV and Excel files
    with clearly formatted columns.
    """
    
    # Get all JSON files in the directory
    results_path = Path(results_dir)
    json_files = list(results_path.glob("*.json"))
    
    # Initialize list to store all data
    combined_data = []
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract basic information
            row = {
                'Model Name': data.get('model_name', 'Unknown'),
                'Indexing Time (s)': round(data.get('indexing_time', 0), 2),
                'Search Time (s)': round(data.get('search_time', 0), 2)
            }
            
            # Extract search results for each query
            search_results = data.get('search_results', [])
            
            # Define query labels
            query_labels = [
                "Query 1: Электростатические поля и теплоотдача",
                "Query 2: Плотность масла МС-20",
                "Query 3: Осадкообразование в системах смазки"
            ]
            
            # Process each query and its results
            for i, query_result in enumerate(search_results[:3]):  # Limit to 3 queries
                query_label = query_labels[i]
                results = query_result.get('results', [])
                
                # Add content for each of the top 3 results
                for j, result in enumerate(results[:3]):  # Limit to 3 results per query
                    content = result.get('content', '')
                    # Сохраняем полное содержимое без обрезки
                    row[f'{query_label} - Result {j+1}'] = content
            
            combined_data.append(row)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(combined_data)
    
    # Save to CSV
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Combined results saved to {output_csv}")
    
    # Save to Excel with better formatting
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Evaluation Results', index=False)
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Evaluation Results']
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Set column width and apply header format
        for idx, col in enumerate(df.columns):
            # Apply header format
            worksheet.write(0, idx, col, header_format)
            
            # Set column width based on content
            max_len = max(
                df[col].astype(str).map(len).max(),  # len of largest item
                len(str(col))  # len of column name/header
            ) + 2  # adding a little extra space
            
            # Limit max width to accommodate full content
            if max_len > 100:
                max_len = 100
                
            worksheet.set_column(idx, idx, max_len)
    
    print(f"Combined results saved to {output_excel}")
    
    return df

if __name__ == "__main__":
    df = combine_evaluation_results()
    print(f"Processed {len(df)} evaluation files")
    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")