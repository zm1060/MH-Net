'''
Training Configuration
'''

# subgraph
class Config:
    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0

    EMBEDDING_SIZE = 64
    H_FEATS = 128
    NUM_CLASSES = 14

    PMI_WINDOW_SIZE = 5
    PAD_TRUNC_DIGIT = 256
    FLOW_PAD_TRUNC_LENGTH = 50
    BYTE_PAD_TRUNC_LENGTH = 150
    HEADER_BYTE_PAD_TRUNC_LENGTH = 50
    ANOMALOUS_FLOW_THRESHOLD = 10000

'''
ISCX-VPN Dataset Configuration
'''
class ISCXVPNConfig(Config):
    # Base directory for ISCX-VPN dataset
    DATASET_DIR = r'./datasets/iscx_vpn'
    PROCESSED_DIR = r'./processed/iscx_vpn'
    RESULT_DIR = r'./results/iscx_vpn'
    
    # NPZ data files
    TRAIN_DATA = RESULT_DIR + r'/train.npz'
    HEADER_TRAIN_DATA = RESULT_DIR + r'/header_train.npz'
    TEST_DATA = RESULT_DIR + r'/test.npz'
    HEADER_TEST_DATA = RESULT_DIR + r'/header_test.npz'
    TEST_DATA_FLOW = RESULT_DIR + r'/test_flow.npz'
    HEADER_TEST_DATA_FLOW = RESULT_DIR + r'/header_test_flow.npz'
    
    # Graph data files - 8bit
    TRAIN_GRAPH_COMBINE = RESULT_DIR + r'/8bit/train_graph_combine.dgl'
    TEST_GRAPH_COMBINE = RESULT_DIR + r'/8bit/test_graph_combine.dgl'
    
    # Graph data files - 4bit
    TRAIN_GRAPH_COMBINE_4bit = RESULT_DIR + r'/4bit/train_graph_combine.dgl'
    TEST_GRAPH_COMBINE_4bit = RESULT_DIR + r'/4bit/test_graph_combine.dgl'
    
    # Regular graph data
    TRAIN_GRAPH_DATA = RESULT_DIR + r'/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = RESULT_DIR + r'/header_train_graph.dgl'
    TEST_GRAPH_DATA = RESULT_DIR + r'/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = RESULT_DIR + r'/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW = RESULT_DIR + r'/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW = RESULT_DIR + r'/header_test_graph_flow.dgl'
    
    # 4-bit data files
    TRAIN_DATA_4bit = RESULT_DIR + r'/4bit/train.npz'
    HEADER_TRAIN_DATA_4bit = RESULT_DIR + r'/4bit/header_train.npz'
    TEST_DATA_4bit = RESULT_DIR + r'/4bit/test.npz'
    HEADER_TEST_DATA_4bit = RESULT_DIR + r'/4bit/header_test.npz'
    TEST_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/test_flow.npz'
    HEADER_TEST_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/header_test_flow.npz'
    
    # 4-bit graph data
    TRAIN_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/header_train_graph.dgl'
    TEST_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/header_test_graph_flow.dgl'
    
    # Model checkpoint
    MIX_MODEL_CHECKPOINT = RESULT_DIR + r'/checkpoints/bit_and_normal_iscx_multimodal_Hetero_1_4_batch.pth'
    
    # Other configurations
    NUM_CLASSES = 6
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 24
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.0
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    # Raw data and processed data directories
    DIR_PATH_DICT = {
        0: DATASET_DIR + r'/chat',
        1: DATASET_DIR + r'/email', 
        2: DATASET_DIR + r'/file',
        3: DATASET_DIR + r'/p2p',
        4: DATASET_DIR + r'/streaming',
        5: DATASET_DIR + r'/voip',
    }
    
    DIR_SAVE_DICT = {
        0: PROCESSED_DIR + r'/chat',
        1: PROCESSED_DIR + r'/email',
        2: PROCESSED_DIR + r'/file',
        3: PROCESSED_DIR + r'/p2p',
        4: PROCESSED_DIR + r'/streaming',
        5: PROCESSED_DIR + r'/voip'
    }

'''
ISCX-NonVPN Dataset Configuration
'''
class ISCXNonVPNConfig(Config):
    # Base directory for ISCX-NonVPN dataset
    DATASET_DIR = r'./datasets/iscx_nonvpn'
    PROCESSED_DIR = r'./processed/iscx_nonvpn'
    RESULT_DIR = r'./results/iscx_nonvpn'
    
    # NPZ data files
    TRAIN_DATA = RESULT_DIR + r'/train.npz'
    HEADER_TRAIN_DATA = RESULT_DIR + r'/header_train.npz'
    TEST_DATA = RESULT_DIR + r'/test.npz'
    HEADER_TEST_DATA = RESULT_DIR + r'/header_test.npz'
    TEST_DATA_FLOW = RESULT_DIR + r'/test_flow.npz'
    HEADER_TEST_DATA_FLOW = RESULT_DIR + r'/header_test_flow.npz'
    
    # Graph data files - 8bit
    TRAIN_GRAPH_COMBINE = RESULT_DIR + r'/8bit/train_graph_combine.dgl'
    TEST_GRAPH_COMBINE = RESULT_DIR + r'/8bit/test_graph_combine.dgl'
    
    # Graph data files - 4bit
    TRAIN_GRAPH_COMBINE_4bit = RESULT_DIR + r'/4bit/train_graph_combine.dgl'
    TEST_GRAPH_COMBINE_4bit = RESULT_DIR + r'/4bit/test_graph_combine.dgl'
    
    # Regular graph data
    TRAIN_GRAPH_DATA = RESULT_DIR + r'/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = RESULT_DIR + r'/header_train_graph.dgl'
    TEST_GRAPH_DATA = RESULT_DIR + r'/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = RESULT_DIR + r'/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW = RESULT_DIR + r'/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW = RESULT_DIR + r'/header_test_graph_flow.dgl'
    
    # 4-bit data files
    TRAIN_DATA_4bit = RESULT_DIR + r'/4bit/train.npz'
    HEADER_TRAIN_DATA_4bit = RESULT_DIR + r'/4bit/header_train.npz'
    TEST_DATA_4bit = RESULT_DIR + r'/4bit/test.npz'
    HEADER_TEST_DATA_4bit = RESULT_DIR + r'/4bit/header_test.npz'
    TEST_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/test_flow.npz'
    HEADER_TEST_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/header_test_flow.npz'
    
    # 4-bit graph data
    TRAIN_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/header_train_graph.dgl'
    TEST_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/header_test_graph_flow.dgl'
    
    # Model checkpoint
    MIX_MODEL_CHECKPOINT = RESULT_DIR + r'/checkpoints/bit_and_normal_iscx_multimodal_Hetero.pth'

    NUM_CLASSES = 6
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 40
    LR = 1e-2
    LR_MIN = 1e-5
    LABEL_SMOOTHING = 0.01
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.1
    DOWNSTREAM_DROPOUT = 0.15
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {
        0: DATASET_DIR + r'/chat',
        1: DATASET_DIR + r'/email',
        2: DATASET_DIR + r'/streaming',
        3: DATASET_DIR + r'/video',
        4: DATASET_DIR + r'/voip',
    }
    
    DIR_SAVE_DICT = {
        0: PROCESSED_DIR + r'/chat',
        1: PROCESSED_DIR + r'/email',
        2: PROCESSED_DIR + r'/streaming',
        3: PROCESSED_DIR + r'/video',
        4: PROCESSED_DIR + r'/voip',
    }

'''
ISCX-Tor Dataset Configuration
'''
class ISCXTorConfig(Config):
    # Base directory for ISCX-Tor dataset
    DATASET_DIR = r'./datasets/iscx_tor'
    PROCESSED_DIR = r'./processed/iscx_tor'
    RESULT_DIR = r'./results/iscx_tor'
    
    # NPZ data files
    TRAIN_DATA = RESULT_DIR + r'/train.npz'
    HEADER_TRAIN_DATA = RESULT_DIR + r'/header_train.npz'
    TEST_DATA = RESULT_DIR + r'/test.npz'
    HEADER_TEST_DATA = RESULT_DIR + r'/header_test.npz'
    TEST_DATA_FLOW = RESULT_DIR + r'/test_flow.npz'
    HEADER_TEST_DATA_FLOW = RESULT_DIR + r'/header_test_flow.npz'
    
    # Graph data files - 8bit
    TRAIN_GRAPH_COMBINE = RESULT_DIR + r'/8bit/train_graph_combine.dgl'
    TEST_GRAPH_COMBINE = RESULT_DIR + r'/8bit/test_graph_combine.dgl'
    
    # Graph data files - 4bit
    TRAIN_GRAPH_COMBINE_4bit = RESULT_DIR + r'/4bit/train_graph_combine.dgl'
    TEST_GRAPH_COMBINE_4bit = RESULT_DIR + r'/4bit/test_graph_combine.dgl'
    
    # Regular graph data
    TRAIN_GRAPH_DATA = RESULT_DIR + r'/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = RESULT_DIR + r'/header_train_graph.dgl'
    TEST_GRAPH_DATA = RESULT_DIR + r'/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = RESULT_DIR + r'/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW = RESULT_DIR + r'/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW = RESULT_DIR + r'/header_test_graph_flow.dgl'
    
    # 4-bit data files
    TRAIN_DATA_4bit = RESULT_DIR + r'/4bit/train.npz'
    HEADER_TRAIN_DATA_4bit = RESULT_DIR + r'/4bit/header_train.npz'
    TEST_DATA_4bit = RESULT_DIR + r'/4bit/test.npz'
    HEADER_TEST_DATA_4bit = RESULT_DIR + r'/4bit/header_test.npz'
    TEST_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/test_flow.npz'
    HEADER_TEST_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/header_test_flow.npz'
    
    # 4-bit graph data
    TRAIN_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/header_train_graph.dgl'
    TEST_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/header_test_graph_flow.dgl'
    
    # Model checkpoint
    MIX_MODEL_CHECKPOINT = RESULT_DIR + r'/checkpoints/bit_and_normal_iscx_multimodal_Hetero.pth'
    
    NUM_CLASSES = 8
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 1

    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.0
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {
        0: DATASET_DIR + r'/audio-streaming',
        1: DATASET_DIR + r'/browsing',
        2: DATASET_DIR + r'/chat',
        3: DATASET_DIR + r'/file',
        4: DATASET_DIR + r'/mail',
        5: DATASET_DIR + r'/p2p',
        6: DATASET_DIR + r'/video-streaming',
        7: DATASET_DIR + r'/voip'
    }

    DIR_SAVE_DICT = {
        0: PROCESSED_DIR + r'/audio-streaming',
        1: PROCESSED_DIR + r'/browsing',
        2: PROCESSED_DIR + r'/chat',
        3: PROCESSED_DIR + r'/file',
        4: PROCESSED_DIR + r'/mail',
        5: PROCESSED_DIR + r'/p2p',
        6: PROCESSED_DIR + r'/video-streaming',
        7: PROCESSED_DIR + r'/voip'
    }

'''
ISCX-NonTor Dataset Configuration
'''
class ISCXNonTorConfig(Config):
    # Base directory for ISCX-NonTor dataset
    DATASET_DIR = r'./datasets/iscx_nontor'
    PROCESSED_DIR = r'./processed/iscx_nontor'
    RESULT_DIR = r'./results/iscx_nontor'
    
    # NPZ data files
    TRAIN_DATA = RESULT_DIR + r'/train.npz'
    HEADER_TRAIN_DATA = RESULT_DIR + r'/header_train.npz'
    TEST_DATA = RESULT_DIR + r'/test.npz'
    HEADER_TEST_DATA = RESULT_DIR + r'/header_test.npz'
    TEST_DATA_FLOW = RESULT_DIR + r'/test_flow.npz'
    HEADER_TEST_DATA_FLOW = RESULT_DIR + r'/header_test_flow.npz'
    
    # Graph data files - 8bit
    TRAIN_GRAPH_COMBINE = RESULT_DIR + r'/8bit/train_graph_combine.dgl'
    TEST_GRAPH_COMBINE = RESULT_DIR + r'/8bit/test_graph_combine.dgl'
    
    # Graph data files - 4bit
    TRAIN_GRAPH_COMBINE_4bit = RESULT_DIR + r'/4bit/train_graph_combine.dgl'
    TEST_GRAPH_COMBINE_4bit = RESULT_DIR + r'/4bit/test_graph_combine.dgl'
    
    # Regular graph data
    TRAIN_GRAPH_DATA = RESULT_DIR + r'/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = RESULT_DIR + r'/header_train_graph.dgl'
    TEST_GRAPH_DATA = RESULT_DIR + r'/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = RESULT_DIR + r'/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW = RESULT_DIR + r'/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW = RESULT_DIR + r'/header_test_graph_flow.dgl'
    
    # 4-bit data files
    TRAIN_DATA_4bit = RESULT_DIR + r'/4bit/train.npz'
    HEADER_TRAIN_DATA_4bit = RESULT_DIR + r'/4bit/header_train.npz'
    TEST_DATA_4bit = RESULT_DIR + r'/4bit/test.npz'
    HEADER_TEST_DATA_4bit = RESULT_DIR + r'/4bit/header_test.npz'
    TEST_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/test_flow.npz'
    HEADER_TEST_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/header_test_flow.npz'
    
    # 4-bit graph data
    TRAIN_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/header_train_graph.dgl'
    TEST_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/header_test_graph_flow.dgl'
    
    # Model checkpoint
    MIX_MODEL_CHECKPOINT = RESULT_DIR + r'/checkpoints/bit_and_normal_iscx_multimodal_Hetero.pth'

    NUM_CLASSES = 8
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 10
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 10
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.1
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {
        0: DATASET_DIR + r'/audio',
        1: DATASET_DIR + r'/browsing',
        2: DATASET_DIR + r'/chat',
        3: DATASET_DIR + r'/email',
        4: DATASET_DIR + r'/ftp',
        5: DATASET_DIR + r'/p2p',
        6: DATASET_DIR + r'/video',
        7: DATASET_DIR + r'/voip',
    }
    
    DIR_SAVE_DICT = {
        0: PROCESSED_DIR + r'/audio',
        1: PROCESSED_DIR + r'/browsing',
        2: PROCESSED_DIR + r'/chat',
        3: PROCESSED_DIR + r'/email',
        4: PROCESSED_DIR + r'/ftp',
        5: PROCESSED_DIR + r'/p2p',
        6: PROCESSED_DIR + r'/video',
        7: PROCESSED_DIR + r'/voip',
    }

class CICIoTConfig(Config):
    # Base directory for CICIoT dataset
    DATASET_DIR = r'./datasets/ciciot'
    PROCESSED_DIR = r'./processed/ciciot'
    RESULT_DIR = r'./results/ciciot'
    
    # NPZ data files
    TRAIN_DATA = RESULT_DIR + r'/train.npz'
    HEADER_TRAIN_DATA = RESULT_DIR + r'/header_train.npz'
    TEST_DATA = RESULT_DIR + r'/test.npz'
    HEADER_TEST_DATA = RESULT_DIR + r'/header_test.npz'
    TEST_DATA_FLOW = RESULT_DIR + r'/test_flow.npz'
    HEADER_TEST_DATA_FLOW = RESULT_DIR + r'/header_test_flow.npz'
    
    # Graph data files - 8bit
    TRAIN_GRAPH_COMBINE = RESULT_DIR + r'/8bit/train_graph_combine.dgl'
    TEST_GRAPH_COMBINE = RESULT_DIR + r'/8bit/test_graph_combine.dgl'
    
    # Graph data files - 4bit
    TRAIN_GRAPH_COMBINE_4bit = RESULT_DIR + r'/4bit/train_graph_combine.dgl'
    TEST_GRAPH_COMBINE_4bit = RESULT_DIR + r'/4bit/test_graph_combine.dgl'
    
    # Regular graph data
    TRAIN_GRAPH_DATA = RESULT_DIR + r'/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = RESULT_DIR + r'/header_train_graph.dgl'
    TEST_GRAPH_DATA = RESULT_DIR + r'/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = RESULT_DIR + r'/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW = RESULT_DIR + r'/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW = RESULT_DIR + r'/header_test_graph_flow.dgl'
    
    # 4-bit data files
    TRAIN_DATA_4bit = RESULT_DIR + r'/4bit/train.npz'
    HEADER_TRAIN_DATA_4bit = RESULT_DIR + r'/4bit/header_train.npz'
    TEST_DATA_4bit = RESULT_DIR + r'/4bit/test.npz'
    HEADER_TEST_DATA_4bit = RESULT_DIR + r'/4bit/header_test.npz'
    TEST_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/test_flow.npz'
    HEADER_TEST_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/header_test_flow.npz'
    
    # 4-bit graph data
    TRAIN_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/header_train_graph.dgl'
    TEST_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA_4bit = RESULT_DIR + r'/4bit/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW_4bit = RESULT_DIR + r'/4bit/header_test_graph_flow.dgl'

    # Additional bit versions for CICIoT
    TRAIN_DATA_2bit = RESULT_DIR + r'/2bit/train.npz'
    HEADER_TRAIN_DATA_2bit = RESULT_DIR + r'/2bit/header_train.npz'
    TEST_DATA_2bit = RESULT_DIR + r'/2bit/test.npz'
    HEADER_TEST_DATA_2bit = RESULT_DIR + r'/2bit/header_test.npz'
    TEST_DATA_FLOW_2bit = RESULT_DIR + r'/2bit/test_flow.npz'
    HEADER_TEST_DATA_FLOW_2bit = RESULT_DIR + r'/2bit/header_test_flow.npz'
    
    TRAIN_GRAPH_DATA_2bit = RESULT_DIR + r'/2bit/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA_2bit = RESULT_DIR + r'/2bit/header_train_graph.dgl'
    TEST_GRAPH_DATA_2bit = RESULT_DIR + r'/2bit/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA_2bit = RESULT_DIR + r'/2bit/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW_2bit = RESULT_DIR + r'/2bit/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW_2bit = RESULT_DIR + r'/2bit/header_test_graph_flow.dgl'
    
    TRAIN_DATA_6bit = RESULT_DIR + r'/6bit/train.npz'
    HEADER_TRAIN_DATA_6bit = RESULT_DIR + r'/6bit/header_train.npz'
    TEST_DATA_6bit = RESULT_DIR + r'/6bit/test.npz'
    HEADER_TEST_DATA_6bit = RESULT_DIR + r'/6bit/header_test.npz'
    TEST_DATA_FLOW_6bit = RESULT_DIR + r'/6bit/test_flow.npz'
    HEADER_TEST_DATA_FLOW_6bit = RESULT_DIR + r'/6bit/header_test_flow.npz'
    
    TRAIN_GRAPH_DATA_6bit = RESULT_DIR + r'/6bit/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA_6bit = RESULT_DIR + r'/6bit/header_train_graph.dgl'
    TEST_GRAPH_DATA_6bit = RESULT_DIR + r'/6bit/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA_6bit = RESULT_DIR + r'/6bit/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW_6bit = RESULT_DIR + r'/6bit/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW_6bit = RESULT_DIR + r'/6bit/header_test_graph_flow.dgl'
    
    TRAIN_DATA_10bit = RESULT_DIR + r'/10bit/train.npz'
    HEADER_TRAIN_DATA_10bit = RESULT_DIR + r'/10bit/header_train.npz'
    TEST_DATA_10bit = RESULT_DIR + r'/10bit/test.npz'
    HEADER_TEST_DATA_10bit = RESULT_DIR + r'/10bit/header_test.npz'
    TEST_DATA_FLOW_10bit = RESULT_DIR + r'/10bit/test_flow.npz'
    HEADER_TEST_DATA_FLOW_10bit = RESULT_DIR + r'/10bit/header_test_flow.npz'
    
    TRAIN_GRAPH_DATA_10bit = RESULT_DIR + r'/10bit/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA_10bit = RESULT_DIR + r'/10bit/header_train_graph.dgl'
    TEST_GRAPH_DATA_10bit = RESULT_DIR + r'/10bit/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA_10bit = RESULT_DIR + r'/10bit/header_test_graph.dgl'
    TEST_GRAPH_DATA_FLOW_10bit = RESULT_DIR + r'/10bit/test_graph_flow.dgl'
    TEST_HEADER_DATA_FLOW_10bit = RESULT_DIR + r'/10bit/header_test_graph_flow.dgl'
    
    # Model checkpoint
    MIX_MODEL_CHECKPOINT = RESULT_DIR + r'/checkpoints/bit_and_normal_iscx_multimodal_wo_.pth'

    NUM_CLASSES = 6
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 1

    BATCH_SIZE = 24
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 10
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 5e-4
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.0
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {
        0: DATASET_DIR + r'/chat',
        1: DATASET_DIR + r'/email',
        2: DATASET_DIR + r'/file',
        3: DATASET_DIR + r'/p2p',
        4: DATASET_DIR + r'/streaming',
        5: DATASET_DIR + r'/voip',
    }
    
    DIR_SAVE_DICT = {
        0: PROCESSED_DIR + r'/chat',
        1: PROCESSED_DIR + r'/email',
        2: PROCESSED_DIR + r'/file',
        3: PROCESSED_DIR + r'/p2p',
        4: PROCESSED_DIR + r'/streaming',
        5: PROCESSED_DIR + r'/voip'
    }

if __name__ == '__main__':
    config = Config()
