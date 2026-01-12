#!/usr/bin/env python3
"""
Convert CV.md to PDF with proper formatting and styling.
"""
import markdown
import subprocess
from pathlib import Path

def convert_md_to_pdf(md_file: str, pdf_file: str):
    """Convert Markdown CV to PDF with professional styling."""
    
    # Read the markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['extra', 'nl2br'])
    
    # Create a styled HTML document
    styled_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mahmoud Hussien - CV</title>
    <style>
        @page {{
            size: A4;
            margin: 1.5cm;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-size: 11pt;
        }}
        
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 18pt;
        }}
        
        h3 {{
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 14pt;
        }}
        
        p {{
            margin: 8px 0;
        }}
        
        ul {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        
        li {{
            margin: 5px 0;
        }}
        
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        
        strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        
        hr {{
            border: none;
            border-top: 1px solid #ddd;
            margin: 15px 0;
        }}
        
        /* Contact information styling */
        body > p:first-of-type {{
            font-size: 10pt;
            line-height: 1.8;
        }}
        
        /* Section spacing */
        h2 + p, h2 + ul {{
            margin-top: 10px;
        }}
        
        /* Professional experience emphasis */
        em {{
            color: #7f8c8d;
            font-style: italic;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    # Save the HTML to a temporary file
    html_file = md_file.replace('.md', '_temp.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    # Convert HTML to PDF using wkhtmltopdf
    try:
        subprocess.run([
            'wkhtmltopdf',
            '--enable-local-file-access',
            '--page-size', 'A4',
            '--margin-top', '15mm',
            '--margin-bottom', '15mm',
            '--margin-left', '15mm',
            '--margin-right', '15mm',
            '--encoding', 'UTF-8',
            html_file,
            pdf_file
        ], check=True, capture_output=True, text=True)
        
        print(f"✅ Successfully converted {md_file} to {pdf_file}")
        
        # Clean up temporary HTML file
        Path(html_file).unlink()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting to PDF: {e}")
        print(f"Error output: {e.stderr}")
        # Keep the HTML file for debugging
        print(f"HTML file saved at: {html_file}")
        raise

if __name__ == "__main__":
    md_file = "CV.md"
    pdf_file = "CV_KCS_IT.pdf"
    
    convert_md_to_pdf(md_file, pdf_file)
