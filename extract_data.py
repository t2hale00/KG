import os
import json
from datetime import datetime
import fitz  # PyMuPDF
from typing import Dict, Any

class PDFMetadataExtractor:
    def __init__(self, base_dir: str = "data_store"):
        self.base_dir = base_dir
        self.metadata = {
            "collection_info": {
                "name": "3GPP Technical Specifications",
                "series": "TS 24",
                "last_updated": datetime.now().isoformat(),
                "total_documents": 0,
                "total_pages": 0
            },
            "documents": []
        }

    def extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from a PDF file"""
        try:
            doc = fitz.open(pdf_path)
            
            # Basic PDF metadata
            pdf_metadata = {
                "file_name": os.path.basename(pdf_path),
                "path": pdf_path,
                "pages": len(doc),
                "file_size_kb": round(os.path.getsize(pdf_path) / 1024, 2),
                "last_modified": datetime.fromtimestamp(
                    os.path.getmtime(pdf_path)
                ).isoformat()
            }

            # Extract document info
            doc_info = doc.metadata
            if doc_info:
                pdf_metadata.update({
                    "title": doc_info.get("title", ""),
                    "author": doc_info.get("author", ""),
                    "creation_date": doc_info.get("creationDate", ""),
                    "modification_date": doc_info.get("modDate", ""),
                    "producer": doc_info.get("producer", ""),
                })

            # Extract spec number and version from filename
            filename = os.path.basename(pdf_path)
            if filename.startswith("TS_"):
                parts = filename.split("_")
                if len(parts) >= 3:
                    pdf_metadata.update({
                        "spec_number": parts[0] + " " + parts[1],
                        "version": parts[2].split(".")[0] if len(parts) > 2 else "",
                        "release": parts[3].split(".")[0] if len(parts) > 3 else ""
                    })

            # Extract first page text for additional context
            if len(doc) > 0:
                first_page = doc[0]
                text = first_page.get_text().strip()
                pdf_metadata["abstract"] = text[:500] if text else ""  # First 500 chars

            doc.close()
            return pdf_metadata

        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {}

    def process_all_documents(self):
        """Process all PDF documents in the directory structure"""
        total_pages = 0
        total_docs = 0

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    print(f"Processing: {pdf_path}")
                    
                    metadata = self.extract_pdf_metadata(pdf_path)
                    if metadata:
                        self.metadata["documents"].append(metadata)
                        total_pages += metadata.get("pages", 0)
                        total_docs += 1

        # Update collection info
        self.metadata["collection_info"].update({
            "total_documents": total_docs,
            "total_pages": total_pages,
            "last_updated": datetime.now().isoformat()
        })

    def save_metadata(self):
        """Save metadata to JSON file"""
        output_path = os.path.join(self.base_dir, "detailed_metadata.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"\nMetadata saved to: {output_path}")

def main():
    extractor = PDFMetadataExtractor()
    print("Starting metadata extraction...")
    extractor.process_all_documents()
    extractor.save_metadata()
    print("Metadata extraction completed!")

if __name__ == "__main__":
    main() 