import fitz  # PyMuPDF
import os
import json
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

def clean_text(text: str) -> str:
    """Cleans extracted PDF text by removing TOC, headers, footers and unnecessary whitespace.
    
    Args:
        text (str): Raw text extracted from PDF
        
    Returns:
        str: Cleaned text
    """
    # Remove common PDF artifacts and clean text
    cleaned = text
    
    # Remove ETSI/3GPP document headers and footers
    cleaned = re.sub(r'ETSI\s+ETSI TS \d+\s+\d+\s+V\d+\.\d+\.\d+\s+\(\d{4}-\d{2}\)', '', cleaned)
    cleaned = re.sub(r'3GPP TS \d+\.\d+ version \d+\.\d+\.\d+ Release \d+', '', cleaned)
    
    # Remove ETSI address and legal information
    cleaned = re.sub(r'ETSI\s+\d+ Route des Lucioles.*?non lucratif.*?Grasse.*?Important notice.*?authorization of ETSI\.', '', cleaned, flags=re.DOTALL)
    
    # Remove page numbers and headers/footers
    cleaned = re.sub(r'\n\s*\d+\s*\n', '\n', cleaned)
    cleaned = re.sub(r'\f', ' ', cleaned)  # Form feed characters
    
    # Remove Table of Contents section
    toc_patterns = [
        r'Table of Contents.*?(?=\d+\s+Scope)', # From TOC until Scope section
        r'Contents.*?(?=\d+\s+Scope)',          # Alternative TOC header
        r'(?:\n\d+\.[\d\.]*\s+.*?(?=\n)){3,}'  # Consecutive numbered entries
    ]
    
    for pattern in toc_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove section numbers at start of lines (common in 3GPP docs)
    cleaned = re.sub(r'^\s*\d+\.[\d\.]*\s+', '', cleaned, flags=re.MULTILINE)
    
    # Remove page numbers and section references
    cleaned = re.sub(r'(?m)^\s*\d+\s*$', '', cleaned)  # Standalone page numbers
    cleaned = re.sub(r'\s*\.\.\.\.*\s*\d+', '', cleaned)  # Section references with dots
    
    # Remove repeated document references
    cleaned = re.sub(r'Reference RTS/TSGC-\d+.*?Keywords.*?\n', '', cleaned, flags=re.DOTALL)
    
    # Clean up excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
    
    # Remove any remaining dots-only lines (common in TOC)
    cleaned = re.sub(r'^\.*\s*$', '', cleaned, flags=re.MULTILINE)
    
    return cleaned.strip()

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        
        if text:
            print("Text extracted successfully!")
            # Clean the extracted text
            text = clean_text(text)
            if not text:
                print("Warning: Text was empty after cleaning.")
        else:
            print("Warning: No text extracted from the PDF.")
            
        return text
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return ""

def save_text_chunks(text, chunk_size=1000):
    """Splits text into smaller chunks for retrieval, respecting natural text boundaries.
    
    Args:
        text (str): Text to split into chunks
        chunk_size (int): Target size for each chunk
        
    Returns:
        List[str]: List of text chunks
    """
    if not text:
        print("No text available to chunk.")
        return []
    
    # First split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # If a single paragraph is longer than chunk_size, split it into sentences
        if len(paragraph) > chunk_size:
            # Split into sentences (handling common abbreviations)
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', paragraph)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # If adding this sentence exceeds chunk_size and we have content,
                # save current chunk and start new one
                if current_size + len(sentence) > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(sentence)
                current_size += len(sentence) + 1  # +1 for space
                
        else:
            # If adding this paragraph exceeds chunk_size and we have content,
            # save current chunk and start new one
            if current_size + len(paragraph) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(paragraph)
            current_size += len(paragraph) + 2  # +2 for paragraph break
    
    # Add any remaining content
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    print(f"Text split into {len(chunks)} chunks.")
    
    # Verify no empty chunks and no extremely short chunks
    chunks = [chunk for chunk in chunks if len(chunk) > 50]  # Filter out very short chunks
    
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