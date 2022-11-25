import torch
from diffusers import StableDiffusionPipeline


pipe = StableDiffusionPipeline.from_pretrained('../stable_diffusion', use_auth_token=True)
# cpu会很慢所以最好在GPU上运行，大概需要10G显存
pipe.to('cuda:0')


#读取数据/将文件目录设置成自己的训练数据集和测试数据集即可.
all_data=[]
with open("CLIP_fusion_mul/text_data/train_data.txt", "r", encoding="utf-8") as file:
    for line in file.readlines():
        all_data.append(line.replace('\n',''))

#将数据分批次输入进模型中
batch_size=5
item_data=[]
batch_data=[]
for i in range(len(all_data)):
    item_data.append(all_data[i])
    if (i+1)%batch_size==0:
        batch_data.append(item_data)
        item_data=[]
    elif i==len(all_data)-1:
        batch_data.append(item_data)

#使用diffusion model获得生成的图像
image_number=0 #统计生成了多少张图片
#迭代50次生成的图片
for j in range(len(batch_data)):
    print("第{}轮生成开始".format(j))
    seed_number=1024 #随机种子的大小
    prompt=batch_data[j]
    #设置随机种子
    generator = torch.Generator("cuda").manual_seed(seed_number)
    #获得生成的图像
    images_nsfw = pipe(prompt, guidance_scale=7.5,generator=generator)
    images=images_nsfw['sample']
    nsfws=images_nsfw['nsfw_content_detected']
    for i, (img, nsfw) in enumerate(zip(images,nsfws)):
        #防止出现nsfw的情况
        while nsfw==True:
            print('重新生成')
            seed_number+=10
            img_nsfw = pipe(prompt[i],guidance_scale=7.5,generator=generator)
            img=img_nsfw["sample"][0]
            nsfw=img_nsfw["nsfw_content_detected"][0]
        #保存生成的图片
        img.save("CLIP_fusion_mul/image_data/train_image/{}.png".format(image_number))
        image_number+=1
            
# #迭代100次生成的图片
# image_number=0 #统计生成了多少张图片
# for j in range(len(batch_data)):
#     print("第{}轮生成开始".format(j))
#     seed_number=1024 #随机种子的大小
#     prompt=batch_data[j]
#     #设置随机种子
#     generator = torch.Generator("cuda").manual_seed(seed_number)
#     #获得生成的图像
#     images_nsfw = pipe(prompt, guidance_scale=7.5,num_inference_steps=100,generator=generator)
#     images=images_nsfw.images
#     nsfws=images_nsfw.nsfw_content_detected
#     for i, (img, nsfw) in enumerate(zip(images,nsfws)):
#         #防止出现nsfw的情况
#         while nsfw==True:
#             print('重新生成')
#             seed_number+=10
#             img_nsfw = pipe(prompt[i],guidance_scale=7.5,num_inference_steps=100,generator=generator)
#             img=img_nsfw["sample"][0]
#             nsfw=img_nsfw["nsfw_content_detected"][0]
#         img.save("/hy-tmp/stable_diffusion_model/image_100/{}.png".format(image_number))
#         image_number+=1