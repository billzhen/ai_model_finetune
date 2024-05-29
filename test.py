import torch

def run_on_cuda():
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA device is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA device is not available. Using CPU.")

    # 创建一个张量
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    print(f"Original tensor on {device}: {x}")

    # 在 CUDA 设备上进行计算
    y = x ** 2
    print(f"Computed tensor on {device}: {y}")

    # 将结果移回到 CPU 上
    y_cpu = y.to('cpu')
    print(f"Result tensor on CPU: {y_cpu}")

    return y_cpu

# 调用函数并打印返回值
result = run_on_cuda()
print(f"Returned value: {result}")




# # 检查是否支持BFloat16数据类型
# if torch.backends.cuda.bfloat16:
#     print("PyTorch支持BFloat16数据类型")
# else:
#     print("PyTorch不支持BFloat16数据类型")










# import torch

# 获取当前 GPU 的属性信息
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
props = torch.cuda.get_device_properties(device)

# 检查是否支持BFloat16数据类型
if props.bfloat16:
    print("GPU支持BFloat16数据类型")
else:
    print("GPU不支持BFloat16数据类型")
