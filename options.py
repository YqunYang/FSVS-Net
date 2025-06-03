from easydict import EasyDict

OPTION = EasyDict()

# ------------------------------------------ data configuration ---------------------------------------------
OPTION.trainset = 'hepatic_vessel'
OPTION.valset = 'hepatic_vessel'
# OPTION.trainset = 'pulmonary_vessel'
# OPTION.valset = 'pulmonary_vessel'
# OPTION.trainset = 'ranel_artery'
# OPTION.valset = 'ranel_artery'
OPTION.datafreq = [5, 1]
OPTION.input_size = (512, 512)  # input image size
OPTION.sampled_frames = 12  # min sampled time length while trianing
OPTION.max_skip = 0  # max skip time length while trianing
OPTION.samples_per_video = 2  # sample numbers per video

# ----------------------------------------- model configuration ---------------------------------------------
OPTION.keydim = 128
OPTION.valdim = 512
OPTION.save_freq = 1
OPTION.epochs_per_increment = 20

# ---------------------------------------- training configuration -------------------------------------------
OPTION.epochs = 50
OPTION.train_batch = 1
OPTION.learning_rate = 0.00001
OPTION.gamma = 0.1
OPTION.momentum = (0.8, 0.9)
OPTION.solver = 'adam'
OPTION.weight_decay = 5e-4
OPTION.iter_size = 1
OPTION.milestone = []  # epochs to degrades the learning rate
OPTION.loss = 'both'  # 'ce' or 'iou' or 'both'
OPTION.mode = 'recurrent'  # 'mask' or 'recurrent' or 'threshold'
OPTION.iou_threshold = 0.65  # used only for 'threshold' training
OPTION.epoch_per_save = 1
# ---------------------------------------- testing configuration --------------------------------------------
OPTION.epoch_per_test = 1

# ------------------------------------------- other configuration -------------------------------------------
OPTION.checkpoint = 'STCN_checkpoint'
OPTION.initial = ''  # path to initialize the backbone
OPTION.resume = False
OPTION.gpu_id = '0'  # defualt gpu-id (if not specified in cmd)
OPTION.workers = 4
OPTION.save_indexed_format = True  # set True to save indexed format png file, otherwise segmentation with original image
#--------------------------------------------------save path--------------------------------------------------
OPTION.output_dir = 'output/hv_alpha1_beta-1'
OPTION.refine_time = 0
OPTION.round_name = 'hv_alpha1_beta-1'
OPTION.refine_round_name = 'hv_alpha1_beta-1'
OPTION.trainset_image_folder = 'training_image'
OPTION.mask_folder = 'training_mask'
OPTION.project_name = 'hepatic_vessel'
OPTION.txt_path = 'hepatic_vessel'
OPTION.GSFMcheckpoint = ''
