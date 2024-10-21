import matplotlib.pyplot as plt
import numpy as np
from TimemixerHBB.configClass import config
import os
from datetime import datetime

def plot_two_arrays(MSEtrain, MSEtest):
    # Check if the input arrays are of the same length
    if len(MSEtrain) != len(MSEtest):
        raise ValueError("Input arrays must be of the same length")

    # Generate the index for the x-axis
    index = np.arange(len(MSEtrain))

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(index, MSEtrain, label='MSE_train', marker='o')
    plt.plot(index, MSEtest, label='MSE_test', marker='s')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Line Plot of Two Arrays')

    # Add legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()


def write_metrics_to_txt(file_path, mse, mae, config):
    # 将时间和字符串组合起来
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 获取 config 对象的所有属性
    config_attributes = "\n".join([f"{attr}: {value}" for attr, value in config.__dict__.items()])

    # 组合输出字符串
    line = f"{timestamp} Test MSE: {mse:.6f}, Test MAE: {mae:.6f}\n{config_attributes}\n\n"

    # 将字符串写入到txt文件的新一行
    with open(file_path, 'a') as file:
        file.write(line)


def write_string_to_file(content: str, CONFIG, mse: float, mae: float):
    # 获取当前时间，并格式化为字符串
    timestamp = datetime.now().strftime("%Y'%m'%d-%H'%M'%S")

    # 创建文件名
    filename = f"{CONFIG.model_name}_{timestamp}_{CONFIG.input_length}_{CONFIG.output_length}_{mse:.4f}.txt"

    config_attributes = "\n".join([f"{attr}: {value}" for attr, value in CONFIG.__dict__.items()])
    # 组合输出字符串
    line = f"{timestamp} Test MSE: {mse:.6f}, Test MAE: {mae:.6f}\n{config_attributes}\n\n"
    # 构建保存路径
    folder_path = os.path.join(os.getcwd(), "all_logger")
    file_path = os.path.join(folder_path, filename)

    # 创建文件夹（如果不存在）
    os.makedirs(folder_path, exist_ok=True)

    # 将内容写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(line)
        f.write(content)

    print(f"日志文件已保存到: {file_path}")
# 使用示例
# write_metrics_to_txt('metrics.txt', 0.123456, 0.234567)


if __name__ == "__main__":

    CONFIG = config()
    write_string_to_file("这是测试内容", CONFIG, mse=0.1234, mae=0.3456)
    # write_metrics_to_txt('../Experimental_Logger.txt', 0.123456, 0.234567, CONFIG)