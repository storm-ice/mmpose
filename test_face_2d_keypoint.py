from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model

from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

model_cfg = 'configs/face_2d_keypoint/rtmpose/face6/rtmpose-m_8xb256-120e_face6-256x256.py'

ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth'

device = 'cuda'

# 使用初始化接口构建模型
model = init_model(model_cfg, ckpt, device=device)

img_path = 'tests/data/cofw/001766.jpg'

# 单张图片推理
batch_results = inference_topdown(model, img_path)

pred_instances = batch_results[0].pred_instances

print(pred_instances.keypoints)
# # array([[[365.83333333,  87.50000477],
# #         [372.08333333,  79.16667175],
# #         [361.66666667,  81.25000501],
# #         [384.58333333,  85.41667151],
# #         [357.5       ,  85.41667151],
# #         [407.5       , 112.50000381],
# #         [363.75      , 125.00000334],
# #         [438.75      , 150.00000238],
# #         [347.08333333, 158.3333354 ],
# #         [451.25      , 170.83333492],
# #         [305.41666667, 177.08333468],
# #         [432.5       , 214.58333325],
# #         [401.25      , 218.74999976],
# #         [430.41666667, 285.41666389],
# #         [370.        , 274.99999762],
# #         [470.        , 356.24999452],
# #         [403.33333333, 343.74999499]]])
#
# 将推理结果打包
results = merge_data_samples(batch_results)

# 初始化可视化器
visualizer = VISUALIZERS.build(model.cfg.visualizer)

# 设置数据集元信息
visualizer.set_dataset_meta(model.dataset_meta)

img = imread(img_path, channel_order='rgb')

# 可视化
visualizer.add_datasample(
    'result',
    img,
    data_sample=results,
    show=True)

