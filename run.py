import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# import torch.nn.functional as F
from dataLoader import load_data
from testGRU import GRUModel
from MSEshower import plot_two_arrays

filepath = "ETTh1.csv"
# Load the data
input_window = 720  # Number of time steps for the input (for long-term forecasting)
output_window = 192  # Number of time steps for the output (for long-term forecasting)
seg_len = 48

batch_size = 64

train_loader, test_loader = load_data(filepath, input_window, output_window, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = GRUModel(input_size=input_window, output_size=output_window, seg_len=seg_len, enc_in=7).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# scheduler = StepLR(optimizer, step_size=3, gamma=0.8)

globalMSE_train = []
globalMSE_test = []

num_epochs = 30  # 训练轮数

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    # mse_loss_whileTrain = 0
    # total_samples_whileTrain = 0
    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device)
        # print(X_batch.shape)
        Y_batch = Y_batch.to(device)
        # print(X_batch.shape)
        # 前向传播
        outputs = model(X_batch)

        # 计算损失
        loss = criterion(outputs, Y_batch)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # mse_loss_whileTrain += nn.functional.mse_loss(outputs, Y_batch, reduction='sum').item()
        # total_samples_whileTrain += Y_batch.numel()
    # scheduler.step()
    avg_loss = total_loss / len(train_loader)
    globalMSE_train.append(avg_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    # mse_whileTrain = mse_loss_whileTrain / total_samples_whileTrain
    # print(f'Epoch [{epoch + 1}/{num_epochs}], MSE: {mse_whileTrain:.4f}')

    model.eval()  # 将模型设置为评估模式
    mse_loss = 0
    mae_loss = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            outputs = model(X_batch)

            # 计算 MSE 和 MAE，使用 'sum' 来累加每个样本的误差
            mse_loss += nn.functional.mse_loss(outputs, Y_batch, reduction='sum').item()
            mae_loss += nn.functional.l1_loss(outputs, Y_batch, reduction='sum').item()
            total_samples += Y_batch.numel()  # 统计总的样本数

    # 计算平均 MSE 和 MAE
    mse = mse_loss / total_samples
    mae = mae_loss / total_samples
    globalMSE_test.append(mse)
    print(f'Test MSE: {mse:.6f}, Test MAE: {mae:.6f}')


print(globalMSE_test)
print(globalMSE_train)
plot_two_arrays(globalMSE_train, globalMSE_test)