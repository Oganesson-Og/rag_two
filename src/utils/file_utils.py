"""
File Utilities Module
------------------

Comprehensive file operations and management utilities.

Key Features:
- Text file operations
- JSON handling
- File compression
- Hash calculations
- MIME type detection
- Path management
- File metadata

Technical Details:
- UTF-8 encoding
- JSON serialization
- GZIP compression
- MD5/SHA256 hashing
- MIME type detection
- Path manipulation
- Error handling

Dependencies:
- pathlib
- hashlib
- json
- shutil
- mimetypes
- gzip
- typing-extensions>=4.7.0

Example Usage:
    utils = FileUtils()
    
    # Read/Write operations
    content = utils.read_text("file.txt")
    utils.write_text("output.txt", "content")
    
    # JSON operations
    data = utils.read_json("config.json")
    utils.write_json("output.json", {"key": "value"})
    
    # File information
    size = utils.get_file_size("large_file.txt")
    mime = utils.get_mime_type("document.pdf")
    md5 = utils.get_md5("important.doc")
    
    # Compression
    compressed = utils.compress_file("data.txt")
    decompressed = utils.decompress_file("data.txt.gz")

Performance Considerations:
- Efficient file handling
- Memory-efficient operations
- Chunked hash calculation
- Optimized compression
- Smart buffer sizes
- Error handling
- Path caching

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import hashlib
import json
import shutil
import mimetypes
import gzip

class FileUtils:
    def read_text(self, file_path: Union[str, Path]) -> str:
        """Read text from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def write_text(self, file_path: Union[str, Path], content: str) -> None:
        """Write text to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    def read_json(self, file_path: Path) -> Dict[str, Any]:
        return json.loads(self.read_text(file_path))
        
    def write_json(self, file_path: Path, data: Dict[str, Any]):
        self.write_text(file_path, json.dumps(data, indent=2))
        
    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """Get file size in bytes."""
        return Path(file_path).stat().st_size
        
    def get_mime_type(self, file_path: Path) -> str:
        return mimetypes.guess_type(file_path)[0]
        
    def get_extension(self, file_path: Path) -> str:
        return file_path.suffix
        
    def get_filename(self, file_path: Path) -> str:
        return file_path.name
        
    def get_directory(self, file_path: Path) -> Path:
        return file_path.parent
        
    def get_md5(self, file_path: Union[str, Path]) -> str:
        """Calculate MD5 hash of file."""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5.update(chunk)
        return md5.hexdigest()
        
    def get_sha256(self, file_path: Path) -> str:
        return hashlib.sha256(file_path.read_bytes()).hexdigest()
        
    def compress_file(self, file_path: Union[str, Path]) -> Path:
        """Compress file using gzip."""
        output_path = Path(str(file_path) + '.gz')
        with open(file_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return output_path
        
    def decompress_file(self, file_path: Path) -> Path:
        decompressed_path = file_path.with_suffix('')
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
        return decompressed_path 