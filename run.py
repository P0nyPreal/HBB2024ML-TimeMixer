import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR

# import torch.nn.functional as F
# from dataLoader import load_data
from TimemixerHBB.dataSets.dataLoader import data_provider
from TimemixerHBB.models.timeMixer import Model
from TimemixerHBB.utils.MSEshower import plot_two_arrays, write_metrics_to_txt, write_string_to_file
from torch.optim import lr_scheduler
from configClass import config

CONFIG = config()

filepath = CONFIG.filepath
# Load the data
input_window = CONFIG.input_length  # Number of time steps for the input (for long-term forecasting)
# input_window = 96  # Number of time steps for the input (for long-term forecasting)

output_window = CONFIG.output_length
    # Number of time steps for the output (for long-term forecasting)
# output_window = 24  # Number of time steps for the output (for long-term forecasting)

seg_len = CONFIG.seg_length
# seg_len = 24


batch_size = CONFIG.batch_size
num_epochs = CONFIG.num_epochs  # 训练轮数
lr = CONFIG.learning_rate

train_dataset, train_loader = data_provider(embed='timeF', batch_size=batch_size, freq='h', root_path='./',
                                            data_path=CONFIG.filepath, seq_len=CONFIG.input_length, label_len=0,
                                            pred_len=CONFIG.output_length, features='M', target='OT', num_workers=0, flag='train')
test_dataset, test_loader = data_provider(embed='timeF', batch_size=batch_size, freq='h', root_path='./',
                                          data_path=CONFIG.filepath, seq_len=CONFIG.input_length, label_len=0,
                                          pred_len=CONFIG.output_length, features='M', target='OT', num_workers=0, flag='test')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = GRUModel(CONFIG).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
# criterion = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = StepLR(optimizer, step_size=3, gamma=0.8)
scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                    steps_per_epoch=len(train_loader),
                                    pct_start=0.3,
                                    epochs=num_epochs,
                                    max_lr=lr)

globalMSE_train = []
globalMSE_test = []
globalMAE_test = []
str_to_log = ""

for epoch in range(num_epochs):
    model.train()
    total_loss = []
    # mse_loss_whileTrain = 0
    # total_samples_whileTrain = 0
    for X_batch, Y_batch, _, _ in train_loader:
        X_batch = X_batch.float().to(device)
        # print(X_batch.shape)
        Y_batch = Y_batch.float().to(device)
        # print(X_batch.shape)
        optimizer.zero_grad()
        # 前向传播
        outputs = model(X_batch)

        # 计算损失
        loss = criterion(outputs, Y_batch)
        total_loss.append(loss.item())

        # 反向传播和优化

        loss.backward()
        optimizer.step()

        # mse_loss_whileTrain += nn.functional.mse_loss(outputs, Y_batch, reduction='sum').item()
        # total_samples_whileTrain += Y_batch.numel()
    scheduler.step()
    avg_loss = np.average(total_loss)
    globalMSE_train.append(avg_loss)

    str_to_print = f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}'
    print(str_to_print)
    str_to_log += str_to_print + "\n"
    # mse_whileTrain = mse_loss_whileTrain / total_samples_whileTrain
    # print(f'Epoch [{epoch + 1}/{num_epochs}], MSE: {mse_whileTrain:.4f}')

    if epoch % 1 == 0:
        model.eval()  # 将模型设置为评估模式
        eval_tot_loss = []
        mse_loss = 0
        mae_loss = 0
        total_samples = 0

        # with torch.no_grad():
        # for X_batch, Y_batch in test_loader:
        for X_batch, Y_batch, _, _ in test_loader:
            X_batch = X_batch.float().to(device)
            Y_batch = Y_batch.float().to(device)

            outputs = model(X_batch)

            # 计算 MSE 和 MAE，使用 'sum' 来累加每个样本的误差
            # mse_loss += nn.functional.mse_loss(outputs, Y_batch, reduction='sum').item()
            mae_loss += nn.functional.l1_loss(outputs, Y_batch, reduction='sum').item()
            total_samples += Y_batch.numel()  # 统计总的样本数

            eval_tot_loss.append(criterion(outputs, Y_batch).item())


        # 计算平均 MSE 和 MAE
        # mse = mse_loss / total_samples
        mse = np.average(eval_tot_loss)
        mae = mae_loss / total_samples
        globalMSE_test.append(mse)
        globalMAE_test.append(mae)
        # globalMSE_test.append(eval_tot_loss/ total_samples)

        str_to_print = f'Test MSE: {mse:.6f}, Test MAE: {mae:.6f}'
        print(str_to_print)
        str_to_log += str_to_print + "\n\n"


    # print(globalMSE_test)
# print(globalMSE_train)
plot_two_arrays(globalMSE_train, globalMSE_test)
write_string_to_file(str_to_log, CONFIG, min(globalMSE_test), min(globalMAE_test))
write_metrics_to_txt(CONFIG.Global_exp_logger_path, min(globalMSE_test), min(globalMAE_test), CONFIG)