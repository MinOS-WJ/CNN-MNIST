import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from torchvision import transforms
from PIL import Image
import os

# 定义与训练代码相同的CNN模型
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(256, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 全局变量
model = None
model_path = ""

# 加载模型函数
def load_model(path=None):
    global model, model_path
    
    if not path:
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
        )
        if not path:
            return
    
    try:
        # 创建新模型实例
        new_model = CNN()
        # 加载权重
        new_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        new_model.eval()
        
        # 更新全局模型和路径
        model = new_model
        model_path = path
        
        # 更新UI
        model_status.config(text=f"模型已加载: {os.path.basename(path)}")
        btn_load_model.config(text="更换模型")
        messagebox.showinfo("成功", f"模型加载成功!\n路径: {path}")
        
        # 启用预测功能
        btn_load_image.config(state=tk.NORMAL)
        label_result.config(text="请上传图片进行预测")
        
    except Exception as e:
        messagebox.showerror("错误", f"加载模型失败: {e}")
        model_status.config(text="模型未加载")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 预测函数
def predict_image(image_path):
    global model
    if model is None:
        messagebox.showerror("错误", "请先加载模型")
        return -1
    
    try:
        img = Image.open(image_path)
        img = transform(img).unsqueeze(0)  # 加入batch维度
        with torch.no_grad():
            output = model(img)
        _, predicted = torch.max(output, 1)
        return predicted.item()
    except Exception as e:
        messagebox.showerror("错误", f"无法处理图像: {e}")
        return -1

# 加载图像函数
def load_image():
    if model is None:
        messagebox.showerror("错误", "请先加载模型")
        return
        
    file_path = filedialog.askopenfilename(
        title="选择图片",
        filetypes=[("图片文件", "*.png *.jpg *.jpeg"), ("所有文件", "*.*")]
    )
    if file_path:
        label_result.config(text="预测中...")
        root.update()  # 立即更新界面
        
        try:
            prediction = predict_image(file_path)
            if prediction >= 0:
                label_result.config(text=f"预测结果: {prediction}")
                # 显示缩略图
                img = Image.open(file_path)
                img.thumbnail((150, 150))  # 调整大小
                img_preview = ImageTk.PhotoImage(img)
                preview_label.config(image=img_preview)
                preview_label.image = img_preview  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {e}")

# 创建窗口
root = tk.Tk()
root.title("手写数字识别")
root.geometry("500x500")  # 增大窗口尺寸

# 设置样式
style = ttk.Style()
style.configure("TButton", font=("Arial", 12))
style.configure("TLabel", font=("Arial", 12))

# 标题
label_title = ttk.Label(root, text="手写数字识别系统", font=("Arial", 20, "bold"))
label_title.pack(pady=15)

# 模型区域框架
model_frame = ttk.Frame(root)
model_frame.pack(fill=tk.X, padx=20, pady=10)

# 模型加载按钮
btn_load_model = ttk.Button(model_frame, text="加载模型", command=load_model)
btn_load_model.pack(side=tk.LEFT, padx=5)

# 模型状态标签
model_status = ttk.Label(model_frame, text="模型未加载")
model_status.pack(side=tk.LEFT, padx=10)

# 图片区域框架
image_frame = ttk.Frame(root)
image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

# 图片预览标签 (初始占位符)
from PIL import ImageTk
dummy_img = Image.new('RGB', (150, 150), color='gray')
dummy_preview = ImageTk.PhotoImage(dummy_img)
preview_label = ttk.Label(image_frame, image=dummy_preview)
preview_label.image = dummy_preview  # 保持引用
preview_label.pack(pady=10)

# 上传图片按钮
btn_load_image = ttk.Button(root, text="上传图片", command=load_image, state=tk.DISABLED)
btn_load_image.pack(pady=10)

# 预测结果标签
label_result = ttk.Label(root, text="请先加载模型", font=("Arial", 16))
label_result.pack(pady=20)

# 状态栏
status_bar = ttk.Label(root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# 尝试加载默认模型
default_model_path = "results/run_20250731-124210/best_model.pth"
if os.path.exists(default_model_path):
    load_model(default_model_path)

root.mainloop()