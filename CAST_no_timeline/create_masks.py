import cv2  
import numpy as np  
import os  
  
def create_simple_mask(image_path, output_path):  
    """  
    Tạo mask đen trắng đơn giản - toàn bộ ảnh là foreground (trắng)  
    """  
    # Đọc ảnh  
    img = cv2.imread(image_path)  
      
    # Tạo mask trắng toàn bộ (255 = foreground)  
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255  
      
    # Lưu mask  
    cv2.imwrite(output_path, mask)  
    print(f"Đã tạo mask: {output_path}")  
  
def create_center_mask(image_path, output_path, margin_percent=0.1):  
    """  
    Tạo mask với vùng trắng ở giữa (80% diện tích), viền đen  
    Tương tự logic mặc định của Colla khi không có foreground  
    """  
    img = cv2.imread(image_path)  
    h, w = img.shape[:2]  
      
    # Tạo mask đen  
    mask = np.zeros((h, w), dtype=np.uint8)  
      
    # Tính vùng foreground (80% diện tích như trong extract_foreground)  
    x1 = int(w * margin_percent)  
    x2 = int(w * (1 - margin_percent))  
    y1 = int(h * margin_percent)  
    y2 = int(h * (1 - margin_percent))  
      
    # Vẽ hình chữ nhật trắng ở giữa  
    mask[y1:y2, x1:x2] = 255  
      
    cv2.imwrite(output_path, mask)  
    print(f"Đã tạo mask: {output_path}")  
  
def batch_create_masks(image_folder, mask_folder, mask_type='simple'):  
    """  
    Tạo mask hàng loạt cho tất cả ảnh trong thư mục  
    """  
    os.makedirs(mask_folder, exist_ok=True)  
      
    # Lấy danh sách file ảnh (.png, .jpg, .jpeg)  
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]  
      
    for img_file in image_files:  
        # Tên file mask (đổi extension thành .png)  
        mask_file = os.path.splitext(img_file)[0] + '.png'
          
        image_path = os.path.join(image_folder, img_file)  
        mask_path = os.path.join(mask_folder, mask_file)  
          
        if mask_type == 'simple':  
            create_simple_mask(image_path, mask_path)  
        else:  
            create_center_mask(image_path, mask_path)  
      
    print(f"Đã tạo {len(image_files)} mask files")  
  
# Sử dụng  
if __name__ == '__main__':  
    # Ví dụ 1: Tạo mask cho 1 ảnh  
    create_simple_mask('my_images/image1.jpg', 'my_masks/image1.png')  
      
    # Ví dụ 2: Tạo mask hàng loạt  
    batch_create_masks('my_images', 'my_masks', mask_type='center')