# TCIG
A generic model for text classification assisted by image generation




通过下述几步可以直接将该模型应用到下游任务中：

1.从“https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main”下载stable_diffusion模型，并将其全部存放到TCIG\stable_diffusion中

2.将数据集按照我们给的样例的形式存储到 CLIP_fusion_mul\text_data中。

3.运行两次Image_generation_model\main.py来根据训练数据集和测试数据集生成对应的图像。（需要手动修改main.py文件中的数据集的存放地址）

4.将CLIP_fusion_mul\classifier.py文件中的text_linear和img_linear的输出维度按照具体任务的标签数量设置。默认为2。

5.CLIP_fusion_mul\Model_parameter_setting.py中可以设置学习路，batch_size和epoch等超参数。

6.运行run.py即可完成模型的训练和测试。
