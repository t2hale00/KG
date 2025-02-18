import fitz  # PyMuPDF
import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        if text:
            print("Text extracted successfully!")
        else:
            print("Warning: No text extracted from the PDF.")
        return text
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return ""

def save_text_chunks(text, chunk_size=500):
    """Splits text into smaller chunks for retrieval"""
    if not text:
        print("No text available to chunk.")
        return []
    
    # Split text into chunks, trying to break at sentence boundaries
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # Get the next chunk_size characters
        chunk_end = min(current_pos + chunk_size, len(text))
        chunk = text[current_pos:chunk_end]
        
        # If we're not at the end, try to find a sentence boundary
        if chunk_end < len(text):
            # Look for sentence endings (., !, ?)
            last_period = max(chunk.rfind('. '), chunk.rfind('! '), chunk.rfind('? '))
            if last_period != -1:
                chunk = chunk[:last_period + 1]
                chunk_end = current_pos + last_period + 1
        
        chunks.append(chunk.strip())
        current_pos = chunk_end
    
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

def read_pdfs_from_directory(directory: str = "data", pattern: str = "*.pdf") -> List[Dict[str, str]]:
    """Read all PDFs from a directory and return their contents.
    
    Args:
        directory (str): Directory containing PDF files
        pattern (str): Glob pattern for PDF files
        
    Returns:
        List[Dict[str, str]]: List of documents with their metadata and content
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"Directory not found: {directory}")
        return []
    
    documents = []
    pdf_files = list(directory_path.glob(pattern))
    
    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files")
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(str(pdf_file))
            if text:
                doc_info = {
                    'file_path': str(pdf_file),
                    'filename': pdf_file.name,
                    'content': text
                }
                documents.append(doc_info)
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue
    
    return documents

def save_chunks_to_file(chunks, output_dir="processed_data"):
    """Save chunks to both JSON and text files for easy viewing"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save as JSON for structured storage
    json_output = os.path.join(output_dir, "text_chunks.json")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump({"chunks": chunks}, f, indent=2, ensure_ascii=False)
    
    # Save as text file for easy reading
    text_output = os.path.join(output_dir, "text_chunks.txt")
    with open(text_output, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"=== Chunk {i} ===\n")
            f.write(chunk)
            f.write("\n\n" + "="*50 + "\n\n")
    
    print(f"Chunks saved to:")
    print(f"- JSON format: {json_output}")
    print(f"- Text format: {text_output}")
    return json_output, text_output

if __name__ == "__main__":
    # Test the functions
    pdf_file = os.path.join(os.path.dirname(__file__), ".", "data", "TS 24.501.pdf")
    if os.path.exists(pdf_file):
        print(f"Processing file: {pdf_file}")
        raw_text = extract_text_from_pdf(pdf_file)
        if raw_text:
            chunks = save_text_chunks(raw_text)
            if chunks:
                # Save chunks to files
                json_file, text_file = save_chunks_to_file(chunks)
                # Print sample for verification
                print(f"\nSample from first chunk: {chunks[0][:100]}...")
                print(f"\nYou can view all chunks in:")
                print(f"1. {text_file} (human-readable format)")
                print(f"2. {json_file} (JSON format)")
    else:
        print(f"PDF file not found: {pdf_file}")