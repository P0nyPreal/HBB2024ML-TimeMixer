class config:
    def __init__(self):
        self.seq_len = 100
        #  输入序列长度
        self.down_sampling_window = 100
        #  下采样窗口大小
        self.pred_len = 100
#         输出窗口长度
        self.d_model = 512
        self.d_ff = 1024
        self.enc_in = 7
        self.dropout= 0.25
        self.channel_independence = 2
        self.top_k = 4
        self.moving_avg = 3
        self.num_class = 7