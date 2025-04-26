from PIL import Image, ImageDraw, ImageFont
import time
from datetime import datetime
import matplotlib.pyplot as plt

# ------------------ 配置参数 ------------------
char_h = 11
pixels_size = (128, 64)
max_x, max_y = 22, 5
rpi_font_path = "DejaVuSans.ttf"  # 或者使用系统字体路径
# 创建全局 figure 句柄
fig, ax = None, None

# ------------------ 初始化字体 ------------------
try:
    font = ImageFont.truetype(rpi_font_path, char_h)
except IOError:
    font = ImageFont.load_default()
    print("字体文件未找到，已使用默认字体")

# ------------------ 模拟 oled 初始化 ------------------
try:
    import board
    import adafruit_ssd1306
    i2c = board.I2C()
    oled = adafruit_ssd1306.SSD1306_I2C(pixels_size[0], pixels_size[1], i2c)
    oled_found = True
except Exception as e:
    print("未检测到OLED设备，使用PC模拟显示")
    oled = None
    oled_found = False

# ------------------ 显示缓冲区 ------------------
display_lines = [""]

# ------------------ 屏幕刷新函数 ------------------
def _display_update():
    """在OLED或PC上显示内容"""
    global oled
    image = Image.new("1", pixels_size)
    draw = ImageDraw.Draw(image)
    for y, line in enumerate(display_lines):
        draw.text((0, y * char_h), line, font=font, fill=255)

    if oled_found:
        oled.fill(0)
        oled.image(image)
        oled.show()
    else:
        image.show()  # 使用 PIL 显示图像窗口

# ------------------ 添加新行 ------------------
def add_display_line(text: str):
    global display_lines
    text_chunks = [text[i: i+max_x] for i in range(0, len(text), max_x)]
    for text in text_chunks:
        for line in text.split("\n"):
            display_lines.append(line)
            display_lines = display_lines[-max_y:]
    _display_update()

# ------------------ 添加 token ------------------
def add_display_tokens(text: str):
    global display_lines
    last_line = display_lines.pop()
    new_line = last_line + text
    add_display_line(new_line)

# ------------------ 测试滚动 ------------------
if __name__ == "__main__":
    for p in range(20):
        add_display_line(f"{datetime.now().strftime('%H:%M:%S')}: Line-{p}")
        time.sleep(0.5)
