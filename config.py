
class Config:

    WIDTH = 128
    HEIGHT = 128
    STATE_CHANNEL = 2
    ENTROPY_BETA = 0.01
    SEED = 100
    USE_SEG = True

    # train
    current_global_ep = 0
    current_max_score = 0.7356
    MAX_GLOBAL_EP = 1000001
    MAX_EP_STEPS = 10
    EP_BOOTSTRAP = 100
    UPDATE_GLOBAL_ITER = 8  # 目前最优模型参数为8 优化为2
    SAMPLE_BATCH_SIZE = 16
    FREQUENCY_PLANNER = 2
    FREQUENCY_VAE = 1

    GPU_ID = 4
    NF = 16
    BOTTLE = 64
    SAMPLE_THRESHOLD = 0.80
    SCORE_THRESHOLD = 0.86
    WARMUP_EPS = 100
    # MEMORY_SIZE = 800 这里设置太大 会导致内存爆满 进而导致卡顿
    MEMORY_SIZE = 400
    TAU = 0.005
    FIXED_ALPHA = 0.01
    GAMMA = 0.99
    INIT_TEMPERATURE = 0.1
    BSPLINE_AUG = False

    PRE_TRAINED = True

    OPTIMIZE = False

    IS_OPTIMIZER = False
    FSAM = True


    # main parameters
    LEARNING_RATE = 4e-5

    # F-SAM 参数
    FSAM_RHO = 0.35  # 开始消融 之前是0.25最优的 现在改成0.2 0.15 0.10 0.05 0.3 0.35 来进行消融
    FSAM_SIGMA = 1
    FSAM_LDA = 0.95
    FSAM_weight_decay = 1e-4



    # server liver
    # IMAGE_TYPE = 'liver' # liver, brain
    # DATA_TYPE = 'lits' # liver, brain
    # TRAIN_DATA = '/datasets/affined_3d/train_3d.h5'
    # TEST_DATA = '/datasets/liver/lspig_affine_test_3d.h5'
    # # TEST_DATA = '/datasets/liver/sliver_affine_test_3d.h5'
    # TRAIN_SEG_DATA = '/datasets/affined_3d/affine_test_3d.h5'
    # ATLAS = '/datasets/affined_3d/atlas.npz'

    # # server brain
    IMAGE_TYPE = 'heart'  # liver, brain, heart
    DATA_TYPE = 1  # 0:all 1:train  2:test 3:calculateDice
    MODE_TYPE_NAME = 'heart_18'  # 目前在跑heart_3
    PRE_TRAIN_NAME = 'heart_18'
    TRAIN_DATA = '/datasets/affined_3d/train_3d.h5' #没有用到
    TEST_DATA = '/datasets/lpba_val.h5' #没有用到 这在test.py
    # TEST_DATA = '/datasets/liver/sliver_affine_test_3d.h5'
    TRAIN_SEG_DATA = '/datasets/affined_3d/affine_test_3d.h5'
    ATLAS = '/datasets/affined_3d/atlas.npz'

    # LOG_DIR = './log/' + MODE_TYPE_NAME + '/'
    LOG_DIR = '/hdd/nas/b103/xianhongjiang/project/SPAC_acdc/log/' + MODE_TYPE_NAME + '/'

    # MODEL_PATH = './model/'
    MODEL_PATH = '/hdd/nas/b103/xianhongjiang/project/SPAC_acdc/model/' + MODE_TYPE_NAME + '/'
    OPT_MODEL_PATH = './opt_model/'
    PROCESS_PATH = 'process'
    TEST_PATH = 'result'
    DICE_PATH = 'diceResult'
    # ACTOR_MODEL = MODEL_PATH + 'actor.ckpt'
    ENCODER_MODEL = MODEL_PATH + 'encoder.ckpt'


    # for evaluation
    # idx = "max_step" heart_5 92000 是目前最优的
    idx = 30500
    ACTOR_MODEL = MODEL_PATH + 'actor_{}_{}.ckpt'.format(PRE_TRAIN_NAME, idx)
    PLANNER_MODEL = MODEL_PATH + 'planner_{}_{}.ckpt'.format(PRE_TRAIN_NAME, idx)
    CRITIC1_MODEL = MODEL_PATH + 'critic1_{}_{}.ckpt'.format(PRE_TRAIN_NAME, idx)
    CRITIC2_MODEL = MODEL_PATH + 'critic2_{}_{}.ckpt'.format(PRE_TRAIN_NAME, idx)
    # ACTOR_MODEL_RL = MODEL_PATH + 'actor_{}.ckpt'.format(idx)

    ALPHA_OPTIM = MODEL_PATH + 'alpha_optim_{}_{}.ckpt'.format(PRE_TRAIN_NAME, idx)
    ACTOR_MODEL_OPTIM = MODEL_PATH + 'optim_actor_{}_{}.ckpt'.format(PRE_TRAIN_NAME, idx)
    PLANNER_MODEL_OPTIM = MODEL_PATH + 'optim_planner_{}_{}.ckpt'.format(PRE_TRAIN_NAME, idx)
    CRITIC1_MODEL_OPTIM = MODEL_PATH + 'optim_critic1_{}_{}.ckpt'.format(PRE_TRAIN_NAME, idx)
    CRITIC2_MODEL_OPTIM = MODEL_PATH + 'optim_critic2_{}_{}.ckpt'.format(PRE_TRAIN_NAME, idx)




