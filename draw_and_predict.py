import pygame
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import sys
import os

# 导入项目已有的模块
from model import SimpleCNN
from predict import load_trained_model

# ==============================================================================
# 交互式手写数字识别画板
# 功能：
# 1. 提供一个画布窗口，用户可以用鼠标在上面书写数字
# 2. 实时调用训练好的 CNN 模型进行预测
# 3. 在窗口右侧显示预测结果和各数字的置信度条形图
#
# 使用方法：
#   python draw_and_predict.py
#
# 操作：
#   - 鼠标左键拖拽：在画板上书写
#   - 按 C 键 或 鼠标右键：清除画板
#   - 关闭窗口或按 ESC：退出
# ==============================================================================

# 窗口布局
CANVAS_SIZE = 400           # 画布区域 400x400 像素
PANEL_WIDTH = 280           # 右侧预测面板宽度
WINDOW_WIDTH = CANVAS_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = CANVAS_SIZE

# 颜色定义 (RGB)
COLOR_BG_CANVAS = (0, 0, 0)           # 画布背景：黑色
COLOR_BRUSH = (255, 255, 255)         # 笔刷颜色：白色
COLOR_BG_PANEL = (30, 30, 40)         # 面板背景：深灰蓝
COLOR_TEXT = (220, 220, 230)          # 普通文字：浅灰
COLOR_TEXT_DIM = (120, 120, 140)      # 暗淡文字
COLOR_ACCENT = (80, 200, 120)         # 强调色/高亮：翠绿
COLOR_BAR_BG = (50, 50, 65)          # 条形图背景
COLOR_BAR_FILL = (70, 160, 230)      # 条形图填充：蓝色
COLOR_BAR_TOP = (80, 200, 120)       # 最高置信度条形图：绿色
COLOR_DIVIDER = (60, 60, 80)         # 分割线颜色

# 笔刷
BRUSH_RADIUS = 12                     # 笔刷半径

# 预测触发
PREDICT_DELAY_MS = 400                # 停止绘制后多少毫秒触发预测


def get_script_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def preprocess_canvas(canvas_surface):
    """
    将 Pygame 画布转换为模型输入张量。
    
    处理流程：
    1. 从 Pygame Surface 提取像素数组
    2. 转换为 PIL 灰度图
    3. 缩放至 28x28（与训练数据一致）
    4. 归一化：Normalize(0.5, 0.5)
    
    注意：画布已经是黑底白字，与训练时经过 RandomInvert 后的效果一致，
    因此这里不需要再做颜色反转。也不需要 RandomRotation。
    """
    # 从 canvas surface 上获取像素数组 (H, W, 3) - RGB
    pixel_array = pygame.surfarray.array3d(canvas_surface)
    # Pygame surfarray 返回的是 (W, H, 3)，需要转置为 (H, W, 3)
    pixel_array = pixel_array.transpose((1, 0, 2))
    
    # 转为灰度图：取 RGB 平均值（因为画的是纯白/纯黑，简单取 R 通道即可）
    gray_array = pixel_array[:, :, 0].astype(np.uint8)
    
    # 转为 PIL Image
    pil_image = Image.fromarray(gray_array, mode='L')
    
    # 预处理 transform（与训练保持一致，但不含数据增强）
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),               # 缩放至 28x28
        transforms.ToTensor(),                     # 转为张量，值域 [0, 1]
        transforms.Normalize((0.5,), (0.5,))       # 归一化到 [-1, 1]
    ])
    
    tensor = preprocess(pil_image)          # shape: [1, 28, 28]
    tensor = tensor.unsqueeze(0)            # shape: [1, 1, 28, 28]，加上 batch 维度
    return tensor


def predict_from_canvas(model, canvas_surface, device):
    """
    对画布内容进行预测。
    
    返回:
        predicted_digit (int): 预测的数字 0-9
        probabilities (list[float]): 各数字的概率列表（长度为 10）
    """
    input_tensor = preprocess_canvas(canvas_surface).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output[0], dim=0)
        predicted_digit = torch.argmax(probs).item()
        probabilities = probs.cpu().tolist()
    
    return predicted_digit, probabilities


def is_canvas_blank(canvas_surface):
    """检查画布是否为空白（全黑）"""
    pixel_array = pygame.surfarray.array3d(canvas_surface)
    return pixel_array.max() == 0


def draw_panel(screen, font_large, font_medium, font_small,
               predicted_digit, probabilities, canvas_is_blank):
    """
    绘制右侧的预测结果面板。
    包含：标题、预测数字、置信度条形图、操作提示。
    """
    panel_x = CANVAS_SIZE
    panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT)
    
    # 面板背景
    pygame.draw.rect(screen, COLOR_BG_PANEL, panel_rect)
    # 左侧分割线
    pygame.draw.line(screen, COLOR_DIVIDER, (panel_x, 0), (panel_x, WINDOW_HEIGHT), 2)
    
    # ---- 标题 ----
    title_surf = font_medium.render("预测结果", True, COLOR_TEXT)
    title_rect = title_surf.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=15)
    screen.blit(title_surf, title_rect)
    
    # ---- 分割线 ----
    pygame.draw.line(screen, COLOR_DIVIDER,
                     (panel_x + 20, 50), (panel_x + PANEL_WIDTH - 20, 50), 1)
    
    if canvas_is_blank:
        # 画布为空时显示提示
        hint_surf = font_small.render("请在左侧画板书写数字", True, COLOR_TEXT_DIM)
        hint_rect = hint_surf.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=80)
        screen.blit(hint_surf, hint_rect)
        return
    
    if predicted_digit is not None and probabilities is not None:
        # ---- 预测数字（大号显示）----
        digit_str = str(predicted_digit)
        digit_surf = font_large.render(digit_str, True, COLOR_ACCENT)
        digit_rect = digit_surf.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=60)
        screen.blit(digit_surf, digit_rect)
        
        # 在数字下方显示置信度百分比
        confidence = probabilities[predicted_digit] * 100
        conf_str = f"置信度: {confidence:.1f}%"
        conf_surf = font_small.render(conf_str, True, COLOR_TEXT)
        conf_rect = conf_surf.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=130)
        screen.blit(conf_surf, conf_rect)
        
        # ---- 分割线 ----
        pygame.draw.line(screen, COLOR_DIVIDER,
                         (panel_x + 20, 158), (panel_x + PANEL_WIDTH - 20, 158), 1)
        
        # ---- 各数字置信度条形图 ----
        bar_area_top = 168
        bar_height = 18
        bar_gap = 4
        bar_max_width = 140
        label_x = panel_x + 15
        bar_x = panel_x + 50
        
        for i in range(10):
            y = bar_area_top + i * (bar_height + bar_gap)
            prob = probabilities[i]
            
            # 数字标签
            label_surf = font_small.render(str(i), True, COLOR_TEXT)
            screen.blit(label_surf, (label_x, y))
            
            # 条形图背景
            bg_rect = pygame.Rect(bar_x, y + 2, bar_max_width, bar_height - 4)
            pygame.draw.rect(screen, COLOR_BAR_BG, bg_rect, border_radius=3)
            
            # 条形图填充
            fill_width = int(prob * bar_max_width)
            if fill_width > 0:
                fill_color = COLOR_BAR_TOP if i == predicted_digit else COLOR_BAR_FILL
                fill_rect = pygame.Rect(bar_x, y + 2, fill_width, bar_height - 4)
                pygame.draw.rect(screen, fill_color, fill_rect, border_radius=3)
            
            # 概率百分比文字
            pct_str = f"{prob * 100:.1f}%"
            pct_surf = font_small.render(pct_str, True, COLOR_TEXT_DIM)
            screen.blit(pct_surf, (bar_x + bar_max_width + 8, y))
    
    # ---- 底部操作提示 ----
    tips = ["C / 右键 - 清除画板", "ESC - 退出"]
    for idx, tip in enumerate(tips):
        tip_surf = font_small.render(tip, True, COLOR_TEXT_DIM)
        tip_rect = tip_surf.get_rect(
            centerx=panel_x + PANEL_WIDTH // 2,
            bottom=WINDOW_HEIGHT - 10 - (len(tips) - 1 - idx) * 22
        )
        screen.blit(tip_surf, tip_rect)


def interpolate_points(p1, p2, radius):
    """
    在两点之间插值，避免鼠标快速移动时笔画出现断裂。
    返回需要绘制圆形的所有中间点列表。
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    distance = max(abs(dx), abs(dy))
    
    points = []
    if distance == 0:
        return [p1]
    
    # 步长为笔刷半径的一半，确保连续覆盖
    steps = max(int(distance / (radius * 0.5)), 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(x1 + dx * t)
        y = int(y1 + dy * t)
        points.append((x, y))
    
    return points


def main():
    # ---- 初始化 ----
    script_dir = get_script_dir()
    os.chdir(script_dir)  # 确保能正确找到 best_model.pth
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在加载模型... (设备: {device})")
    
    try:
        model = load_trained_model("best_model.pth", device=device)
        print("模型加载成功！")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 train.py 训练模型。")
        return
    
    # 初始化 Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("手写数字识别 - 画板")
    clock = pygame.time.Clock()
    
    # 创建画布 Surface（仅左侧画板区域）
    canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
    canvas.fill(COLOR_BG_CANVAS)
    
    # 加载字体（使用系统中文字体）
    # Windows 通常有微软雅黑
    chinese_font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
    ]
    
    font_path = None
    for fp in chinese_font_paths:
        if os.path.exists(fp):
            font_path = fp
            break
    
    if font_path:
        font_large = pygame.font.Font(font_path, 64)
        font_medium = pygame.font.Font(font_path, 22)
        font_small = pygame.font.Font(font_path, 15)
    else:
        # 回退到系统默认字体
        font_large = pygame.font.SysFont("arial", 64)
        font_medium = pygame.font.SysFont("arial", 22)
        font_small = pygame.font.SysFont("arial", 15)
    
    # ---- 状态变量 ----
    drawing = False                     # 当前是否正在绘制
    last_pos = None                     # 上一帧鼠标位置（用于插值连线）
    predicted_digit = None              # 当前预测数字
    probabilities = None                # 当前各数字概率
    last_draw_time = 0                  # 上次绘制的时间戳
    need_predict = False                # 是否需要触发预测
    canvas_dirty = False                # 画布是否有内容更新
    
    running = True
    
    # ---- 主循环 ----
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # 使用 scancode 检测物理 C 键，避免中文输入法拦截
                elif event.scancode == 6 or event.key == pygame.K_c:
                    # 清除画板
                    canvas.fill(COLOR_BG_CANVAS)
                    predicted_digit = None
                    probabilities = None
                    canvas_dirty = False
                    need_predict = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键：绘制
                    mx, my = event.pos
                    if mx < CANVAS_SIZE:  # 只在画布区域内绘制
                        drawing = True
                        last_pos = (mx, my)
                        pygame.draw.circle(canvas, COLOR_BRUSH, (mx, my), BRUSH_RADIUS)
                        last_draw_time = current_time
                        canvas_dirty = True
                        need_predict = True
                elif event.button == 3:  # 右键：清除画板
                    canvas.fill(COLOR_BG_CANVAS)
                    predicted_digit = None
                    probabilities = None
                    canvas_dirty = False
                    need_predict = False
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
                    last_pos = None
            
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    mx, my = event.pos
                    # 将坐标限制在画布范围内
                    mx = max(0, min(mx, CANVAS_SIZE - 1))
                    my = max(0, min(my, CANVAS_SIZE - 1))
                    
                    if last_pos:
                        # 在上一点和当前点之间插值绘制，防止笔画断裂
                        points = interpolate_points(last_pos, (mx, my), BRUSH_RADIUS)
                        for pt in points:
                            pygame.draw.circle(canvas, COLOR_BRUSH, pt, BRUSH_RADIUS)
                    else:
                        pygame.draw.circle(canvas, COLOR_BRUSH, (mx, my), BRUSH_RADIUS)
                    
                    last_pos = (mx, my)
                    last_draw_time = current_time
                    canvas_dirty = True
                    need_predict = True
        
        # ---- 延迟预测：停止绘制一段时间后触发 ----
        if need_predict and canvas_dirty and (current_time - last_draw_time > PREDICT_DELAY_MS):
            if not is_canvas_blank(canvas):
                predicted_digit, probabilities = predict_from_canvas(model, canvas, device)
            else:
                predicted_digit = None
                probabilities = None
            need_predict = False
        
        # ---- 渲染 ----
        # 将画布绘制到窗口左侧
        screen.blit(canvas, (0, 0))
        
        # 绘制右侧预测面板
        draw_panel(screen, font_large, font_medium, font_small,
                   predicted_digit, probabilities, not canvas_dirty)
        
        pygame.display.flip()
        clock.tick(60)  # 限制帧率为 60 FPS
    
    pygame.quit()
    print("程序已退出。")


if __name__ == "__main__":
    main()
