import torch
from transformers import AutoModel, AutoTokenizer
from resp_gen_ai.models.rmllms.skywork_utils.utils import load_image, split_model
import argparse
from .base import MLLMBaseModel


class Skywork_R1V(MLLMBaseModel):
    def __init__(self, model_path, device="cuda"):
        super().__init__(model_path, device)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True, use_fast=False)

    def generate(self,question,images):
        if type(images) != list:
            images = [images]
        pixel_values = [load_image(img_path, max_num=12).to(torch.bfloat16).cuda() for img_path in images]

        if len(pixel_values) > 1:
            num_patches_list = [img.size(0) for img in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = pixel_values[0]
            num_patches_list = None
        pixel_shapes = pixel_values.shape
        print(f"pixel_shapes:{pixel_shapes}")
        prompt = "<image>\n"*len(images) + question
        generation_config = dict(max_new_tokens=64000, do_sample=True, temperature=0.6, top_p=0.95, repetition_penalty=1.05)
        response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config, num_patches_list=num_patches_list)
        return response

    def generate_batch(self, questions, images_batch):
        """
        批量处理方法。每条输入包括一个问题 + 多张图。
        :param questions: list[str]
        :param images_batch: list[list[str]]  # 每一项是多张图的路径
        """
        assert isinstance(questions, list) and isinstance(images_batch, list)
        assert len(questions) == len(images_batch)

        pixel_values_list = []
        num_patches_list = []
        prompt_list = []
        for images, question in zip(images_batch,questions):
            if not isinstance(images, list):
                images = [images]

            pixel_values = [load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                            for img_path in images]

            # 拼接每条样本的图像
            if len(pixel_values) > 1:
                num_patches = len(pixel_values)
                pixel_values = torch.cat(pixel_values, dim=0)
            else:
                num_patches = 1
                pixel_values = pixel_values[0] 
                # pixel_values = pixel_values[0].unsqueeze(0)
            pixel_values_list.append(pixel_values) 
            num_patches_list.append(num_patches)
            prompt = "<image>\n"*len(images) + question
            prompt_list.append(prompt)
        # 检查所有样本拼接后的 pixel shape 是否一致
        pixel_shapes = [p.shape for p in pixel_values_list]
        print(f"pixel_shapes:{pixel_shapes}") # 这里应该是4维
        if all(shape == pixel_shapes[0] for shape in pixel_shapes):
            pass
        else:
            raise ValueError(f"图像 shape 不一致，无法 batch：{pixel_shapes}")

        # 配置生成参数
        generation_config = dict(
            max_new_tokens=64000,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
        )
        # 调用 batch_chat
        responses = self.model.batch_chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values_list,
            questions=prompt_list,
            generation_config=generation_config,
            num_patches_list=num_patches_list
        )

        return responses



