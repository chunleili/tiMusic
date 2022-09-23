from scipy.io import wavfile
import numpy as np

# 读取wav文件。得到的sample_rate是每秒钟采样多少次，
# data是采样后的结果，为二维numpy数组。每维分别是左右声道的数据。
# 16-bit音频就是int16的整型数据。范围是-32768 到 +32767
sample_rate, data = wavfile.read('music.wav')
print(f"data shape = {data.shape}")
print(f"sample_rate = {sample_rate}")

# 音频的时间长度就是数据长度除以采样率
length = data.shape[0] / sample_rate
print(f"音频总长度 {length}s")
print(f"数据总长度 {data.shape[0]}")

# 音频太长，我们就截取一段
length1 = 20.0 # 截取前20s
num_len = int(length1 * sample_rate) 
data1 = data[:num_len]
# np.savetxt("data1.txt",data1,fmt="%d")
print(f"截取前{length1} s")
print(f"截取后数据长度 {data1.shape[0]}")

def wave_plot():
    # 绘制波形图
    import matplotlib.pyplot as plt
    import numpy as np
    time = np.linspace(0., length1, data1.shape[0])
    plt.plot(time, data1[:, 0], label="Left channel")
    # plt.plot(time, data1[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

# 最终转递的音频数据，我们只传递左声道的
music_data = data1[:,0]

# 模拟是1e-4s 一个substep，seg代表1e-4 s中有多少数据
seg = int(1e-4 * sample_rate)
print(f"seg={seg}")
# 然后每隔seg个数据，进行一小段一小段的求和，存到music_data里面
music_data = np.add.reduceat(music_data, np.arange(0, len(music_data), seg))

print("max=",max(music_data))
print("min=",min(music_data))
np.savetxt("music_data.txt",music_data,fmt="%d")
print(f"最终音频数据shape {music_data.shape}")