import shutil
from pathlib import Path

folder_to_zip = Path("/home/serverai/ltdoanh/LayoutGeneration")      # thư mục cần nén
output_zip = Path("/home/serverai/ltdoanh/LayoutGeneration/layout")  # KHÔNG cần .zip, shutil sẽ tự thêm

shutil.make_archive(
    base_name=str(output_zip),  # đường dẫn + tên file zip (không đuôi)
    format="zip",               # định dạng nén
    root_dir=str(folder_to_zip) # thư mục gốc cần nén
)

print("Done!")

"""
huggingface-cli upload \
  --repo-type dataset \
  doanh25032004/layout \
  /home/serverai/ltdoanh/LayoutGeneration/layout.zip
"""