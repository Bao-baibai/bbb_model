import torch
from torch import nn
from .basic_layers import Transformer, CrossTransformer, GradientReversalLayer
from .bert import BertTextEncoder
import torch.nn.functional as F
from einops import rearrange, repeat


class MyMultimodal(nn.Module):
    def __init__(self, args):
        super(MyMultimodal, self).__init__()

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args['model']['feature_extractor']['bert_pretrained'])

        # 语言、视觉、音频特征的投影
        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0], args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][0], 
                        dim=args['model']['feature_extractor']['hidden_dims'][0], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2], args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][2], 
                        dim=args['model']['feature_extractor']['hidden_dims'][2], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1], args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][1], 
                        dim=args['model']['feature_extractor']['hidden_dims'][1], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )

        # 自适应对齐层中的线性变换
        d_model = args['model']['feature_extractor']['hidden_dims'][0]
        d_k = d_model // args['model']['feature_extractor']['heads']
        nhead = args['model']['feature_extractor']['heads']

        self.W_Q_audio = nn.Linear(d_model, d_k * nhead)  # 音频对齐的查询
        self.W_K_audio = nn.Linear(d_model, d_k * nhead)  # 音频对齐的键
        self.W_Q_visual = nn.Linear(d_model, d_k * nhead)  # 视觉对齐的查询
        self.W_K_visual = nn.Linear(d_model, d_k * nhead)  # 视觉对齐的键
        self.W_V = nn.Linear(d_model, d_k * nhead)  # 对齐后的融合值投影
        
        self.GRL = GradientReversalLayer(alpha=1.0)

        self.effective_discriminator = nn.Sequential(
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['input_dim'],
                      args['model']['dmc']['effectiveness_discriminator']['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['hidden_dim'],
                      args['model']['dmc']['effectiveness_discriminator']['out_dim']),
        )

        self.completeness_check = nn.ModuleList([
            Transformer(num_frames=args['model']['dmc']['completeness_check']['input_length'], 
                        save_hidden=False, 
                        token_len=args['model']['dmc']['completeness_check']['token_length'], 
                        dim=args['model']['dmc']['completeness_check']['input_dim'], 
                        depth=args['model']['dmc']['completeness_check']['depth'], 
                        heads=args['model']['dmc']['completeness_check']['heads'], 
                        mlp_dim=args['model']['dmc']['completeness_check']['hidden_dim']),

            nn.Sequential(
                nn.Linear(args['model']['dmc']['completeness_check']['hidden_dim'], int(args['model']['dmc']['completeness_check']['hidden_dim']/2)),
                nn.LeakyReLU(0.1),
                nn.Linear(int(args['model']['dmc']['completeness_check']['hidden_dim']/2), 1),
                nn.Sigmoid()),
        ])

        # 使用 CrossTransformer 进行多模态融合
        self.cross_transformer = CrossTransformer(
            source_num_frames=args['model']['dmml']['fuison_transformer']['source_length'],
            tgt_num_frames=args['model']['dmml']['fuison_transformer']['tgt_length'],
            dim=args['model']['dmml']['fuison_transformer']['input_dim'],
            depth=args['model']['dmml']['fuison_transformer']['depth'],
            heads=args['model']['dmml']['fuison_transformer']['heads'],
            mlp_dim=args['model']['dmml']['fuison_transformer']['hidden_dim']
        )
        self.output_layer = nn.Linear(128, 1)

        self.proxy_dominate_modality_generator = Transformer(
            num_frames=args['model']['dmc']['proxy_dominant_feature_generator']['input_length'],
            save_hidden=False,
            token_len=args['model']['dmc']['proxy_dominant_feature_generator']['token_length'],
            dim=args['model']['dmc']['proxy_dominant_feature_generator']['input_dim'],
            depth=args['model']['dmc']['proxy_dominant_feature_generator']['depth'],
            heads=args['model']['dmc']['proxy_dominant_feature_generator']['heads'],
            mlp_dim=args['model']['dmc']['proxy_dominant_feature_generator']['hidden_dim'])
        self.h_p = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))


    def adaptive_alignment(self, H_l_2, H_l_3, H_a_1, H_v_1):
        # 假设我们希望实现 8 个头的多头注意力
        num_heads = 8
        head_dim = H_l_2.size(-1) // num_heads  # 每个头的维度

        # 1. 多头查询、键和值
        query_audio = self.W_Q_audio(H_l_2).view(H_l_2.size(0), -1, num_heads, head_dim).transpose(1, 2)
        key_audio = self.W_K_audio(H_a_1).view(H_a_1.size(0), -1, num_heads, head_dim).transpose(1, 2)
        value_audio = self.W_V(H_a_1).view(H_a_1.size(0), -1, num_heads, head_dim).transpose(1, 2)

        # 2. 对每个头独立计算注意力
        alpha = torch.softmax((query_audio @ key_audio.transpose(-2, -1)) / (head_dim ** 0.5), dim=-1)
        aligned_audio = (alpha @ value_audio).transpose(1, 2).contiguous().view(H_l_2.size(0), -1, num_heads * head_dim)

        # 3. 重复以上步骤对视觉模态执行同样的操作
        query_visual = self.W_Q_visual(H_l_3).view(H_l_3.size(0), -1, num_heads, head_dim).transpose(1, 2)
        key_visual = self.W_K_visual(H_v_1).view(H_v_1.size(0), -1, num_heads, head_dim).transpose(1, 2)
        value_visual = self.W_V(H_v_1).view(H_v_1.size(0), -1, num_heads, head_dim).transpose(1, 2)

        beta = torch.softmax((query_visual @ key_visual.transpose(-2, -1)) / (head_dim ** 0.5), dim=-1)
        aligned_visual = (beta @ value_visual).transpose(1, 2).contiguous().view(H_l_3.size(0), -1,
                                                                                 num_heads * head_dim)

        # 4. 拼接音频和视觉的对齐结果
        H_aligned = aligned_audio + aligned_visual

        return H_aligned, alpha, beta

    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        b = vision_m.size(0)

        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        feat_tmp = self.completeness_check[0](h_1_l)[:, :1].squeeze()
        w = self.completeness_check[1](feat_tmp)  # completeness scores

        h_0_p = repeat(self.h_p, '1 n d -> b n d', b=b)
        h_1_p = self.proxy_dominate_modality_generator(torch.cat([h_0_p, h_1_a, h_1_v], dim=1))[:, :8]
        h_1_p = self.GRL(h_1_p)
        h_1_d = h_1_p * (1 - w.unsqueeze(-1)) + h_1_l * w.unsqueeze(-1)

        # 自适应对齐模块
        H_aligned, alpha, beta = self.adaptive_alignment(h_1_d, h_1_d, h_1_a, h_1_v)

        # 跨模态融合（使用 CrossTransformer）
        # 将 h_1_l 和 H_aligned 在最后一个维度上拼接为单一张量
        fusion_input = torch.cat((h_1_l, H_aligned), dim=-1)

        # 传入拼接后的张量
        H_fusion = self.cross_transformer(fusion_input, fusion_input)  # 使用 CrossTransformer 进行融合
        sentiment_preds = self.output_layer(H_fusion.mean(dim=1))


        # 跨模态一致性计算（使用 KL 散度或余弦相似度）
        # consistency_loss_audio = F.kl_div(F.log_softmax(h_1_l, dim=-1), F.softmax(h_1_a, dim=-1), reduction='batchmean')
        # consistency_loss_visual = F.kl_div(F.log_softmax(h_1_l, dim=-1), F.softmax(h_1_v, dim=-1),
        #                                    reduction='batchmean')
        consistency_loss = F.kl_div(F.log_softmax(h_1_l, dim=-1), F.softmax(H_aligned, dim=-1),
                                           reduction='batchmean')

        # 判别器输出，确保生成 effectiveness_discriminator_out
        h_1_d = self.GRL(h_1_l) if self.training else torch.zeros(b, 8, 128).to(h_1_l.device)
        effectiveness_discriminator_out = self.effective_discriminator(
            h_1_d.reshape(b * 8, -1)) if self.training else torch.zeros(b * 8, 2).to(h_1_l.device)

        # 确保 effectiveness_discriminator_out 与 label 的批量大小匹配
        # 假设 label 的维度为 [b * 8]
        effectiveness_discriminator_out = effectiveness_discriminator_out.view(b * 8, -1)

        return {
            'sentiment_preds': sentiment_preds,
            'w': w,
            'effectiveness_discriminator_out': effectiveness_discriminator_out,
            'consistency_loss': consistency_loss  # 返回一致性损失
        }



def build_model(args):
    return MyMultimodal(args)
