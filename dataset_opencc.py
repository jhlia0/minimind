import argparse
from opencc import OpenCC

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Convert JSONL file using OpenCC.")
parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")
parser.add_argument("output_file", type=str, nargs="?", help="Path to the output JSONL file. If not provided, '_tw' will be appended to the input file name.")
args = parser.parse_args()

# Determine output file path
if not args.output_file:
    input_file_base, input_file_ext = args.input_file.rsplit('.', 1)
    args.output_file = f"{input_file_base}_tw.{input_file_ext}"

converter = OpenCC("s2twp")

# Open the input file and create a new output file
with open(args.input_file, "r") as infile, open(args.output_file, "w") as outfile:
    total_lines = sum(1 for _ in infile)  # Count total lines in the file
    infile.seek(0)  # Reset file pointer to the beginning

    for idx, line in enumerate(infile, start=1):
        converted_line = converter.convert(line.strip())  # Convert the line using OpenCC
        outfile.write(converted_line + "\n")  # Write the converted line to the new file

        # Print progress
        if idx % 100 == 0 or idx == total_lines:
            print(f"Processed {idx}/{total_lines} lines ({(idx / total_lines) * 100:.2f}%)")