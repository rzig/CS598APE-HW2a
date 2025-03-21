import numpy as np
import csv
import os
from tqdm import tqdm

def generate_large_function_csv(filename, target_size_gb=1, batch_size=10000):
    """
    Generate a CSV file of approximately the target size in GB with random inputs
    and outputs of the function:
    f(x1,x2,x3,x4,x5,x6,x7,x8) = (1/x1) + 2*x2*x3 + 4*(x4/x5 + x6/x5)/sqrt(x7 + x8)
    
    Parameters:
    filename (str): Output CSV filename
    target_size_gb (float): Target file size in gigabytes
    batch_size (int): Number of rows to generate at once for efficiency
    """
    # Function to calculate
    def calculate_f(x1, x2, x3, x4, x5, x6, x7, x8):
        try:
            return (1/x1) + 2*x2*x3 + 4*(x4/x5 + x6/x5)/np.sqrt(x7 + x8)
        except (ZeroDivisionError, ValueError):
            # Handle potential division by zero or negative sqrt
            return np.nan
    
    # Calculate approximate bytes per line based on a test batch
    test_batch_x = np.random.uniform(0.1, 10, (100, 8))
    test_batch_y = np.array([calculate_f(*row) for row in test_batch_x])
    
    test_data = np.column_stack((test_batch_x, test_batch_y))
    test_str = '\n'.join([','.join(map(str, row)) for row in test_data])
    bytes_per_line = len(test_str) / len(test_data)
    
    # Calculate total lines needed
    target_size_bytes = target_size_gb * 1024**3
    total_lines = int(target_size_bytes / bytes_per_line)
    
    # Write header
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'output'])
    
    # Generate and write data in batches
    lines_written = 0
    
    with tqdm(total=total_lines, desc="Generating CSV") as pbar:
        while lines_written < total_lines:
            # Generate batch of random values (ensuring no zeros for x1 and x5)
            batch_size = min(batch_size, total_lines - lines_written)
            
            x1 = np.random.uniform(0.1, 10, batch_size)  # Avoid division by zero
            x2 = np.random.uniform(0.1, 10, batch_size)
            x3 = np.random.uniform(0.1, 10, batch_size)
            x4 = np.random.uniform(0.1, 10, batch_size)
            x5 = np.random.uniform(0.1, 10, batch_size)  # Avoid division by zero
            x6 = np.random.uniform(0.1, 10, batch_size)
            x7 = np.random.uniform(0, 10, batch_size)    # Ensure x7+x8 > 0 for sqrt
            x8 = np.random.uniform(0.1, 10, batch_size)
            
            # Calculate function output
            outputs = []
            for i in range(batch_size):
                output = calculate_f(x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i], x8[i])
                outputs.append(output)
            
            # Combine inputs and outputs
            batch_data = np.column_stack((x1, x2, x3, x4, x5, x6, x7, x8, outputs))
            
            # Append to file
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(batch_data)
            
            lines_written += batch_size
            pbar.update(batch_size)
    
    # Verify file size
    actual_size_gb = os.path.getsize(filename) / (1024**3)
    print(f"CSV generation complete. File size: {actual_size_gb:.2f} GB")
    print(f"Total rows generated: {lines_written}")

# Example usage
if __name__ == "__main__":
    generate_large_function_csv("function_data_1gb.csv")
