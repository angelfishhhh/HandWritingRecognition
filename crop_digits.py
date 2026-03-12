import os
from PIL import Image

def crop_grid_image(image_path, output_folder, rows=8, cols=14, max_images=100):
    """
    将包含网格的图片裁剪成单个小图片
    """
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载图片
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"找不到图片文件: {image_path}，请检查路径。")
        return

    img_width, img_height = img.size

    # 计算每个小网格的宽度和高度
    cell_width = img_width // cols
    cell_height = img_height // rows

    count = 0
    
    # 遍历行和列进行裁剪
    for row in range(rows):
        for col in range(cols):
            # 达到所需的图片数量后停止
            if count >= max_images:
                return

            # 计算当前小图的边界 (左, 上, 右, 下)
            left = col * cell_width
            upper = row * cell_height
            right = (col + 1) * cell_width
            lower = (row + 1) * cell_height

            # 裁剪并保存
            cropped_img = img.crop((left, upper, right, lower))
            
           
            file_name = f"img_9_{count+100+1:03d}.png"
            save_path = os.path.join(output_folder, file_name)
            
            cropped_img.save(save_path, "PNG")
            count += 1

    print(f"处理完成！共成功裁剪出 {count} 张图片，已保存至 '{output_folder}' 文件夹。")

# 运行程序
if __name__ == "__main__":
    # 输入图片路径
    INPUT_IMAGE = "1.png"  
    # 输出文件夹名称
    OUTPUT_DIR = "my_digits\\9"       
    
    crop_grid_image(INPUT_IMAGE, OUTPUT_DIR)