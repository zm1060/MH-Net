1.首先是PCAP到NPZ的转换 (pcap2npy.py):
输入：从./datasets/iscx_vpn/各个子目录(chat/email/file等)读取pcap文件
输出：在./processed/iscx_vpn/对应子目录下生成npz文件
处理过程：
读取每个pcap文件中的所有数据包
提取每个包的header和payload信息
保存包的各种特征(header, payload, IP, 端口等)到npz文件


2.然后是预处理阶段 (preprocess.py):
构建数据集 (construct_dataset_from_bytes_ISCX):
读取上一步生成的npz文件
使用split_flow_ISCX函数将数据包分割成流
按9:1比例分割训练集和测试集
保存到以下文件：
载荷数据：./res_iscx_github/train.npz和test.npz
包头数据：./res_iscx_github/header_train.npz和header_test.npz
b) 比特转换 (train2bit):
将字节数据转换为指定长度(2/4/10)的比特表示
保存到以下文件(以4bit为例)：
./res_iscx4bit_github/train.npz
./res_iscx4bit_github/header_train.npz
./res_iscx4bit_github/test.npz
./res_iscx4bit_github/header_test.npz
c) 构建图结构 (construct_graph_format_data):
将包头和载荷数据构建成图结构
使用PMI窗口大小(默认5)构建节点关系
保存到以下文件：
原始数据图：./res_ISCX_8bit/train_graph_combine.dgl和test_graph_combine.dgl
4bit数据图：./res_ISCX_4bit/train_graph_combine.dgl和test_graph_combine.dgl


PCAP文件 (原始数据包)
↓ pcap2npy.py
NPZ文件 (提取的包特征)
↓ preprocess.py
训练/测试集 NPZ (分割的流数据)
↓ train2bit()
比特形式 NPZ (2/4/10比特表示)
↓ construct_graph_format_data()
DGL图文件 (最终的图结构数据)