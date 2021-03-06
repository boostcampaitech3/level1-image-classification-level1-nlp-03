import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet as EffNet

from models.swin_transformer import SwinTransformer


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes,
    ):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.model = EffNet.from_pretrained("efficientnet-b7")
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        # x shape: batch_size, 3, 128, 96

        x = self.model(x)
        # x shape: batch_size, 1000

        x = self.fc(x)
        # x shape: batch_size, num_classes

        return x


class SwinTransformerLarge384(nn.Module):
    def __init__(
        self,
        num_classes,
    ):
        super().__init__()
        self.swin_transformer = SwinTransformer(
            img_size=384,
            num_classes=num_classes,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            window_size=12,
            drop_path_rate=0.2,
        )

        file_name = "swin_transformer_large_384_384"
        self.load_pretrained(self.swin_transformer, file_name)

    def forward(self, x):
        # x shape: batch_size, 3, 384, 384

        x = self.swin_transformer(x)
        # x shape: batch_size, num_classes

        return x

    def load_pretrained(self, model, file_name):
        print(f"==============> Loading weight {file_name} for fine-tuning......")
        checkpoint_path = f"./weights/{file_name}.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"]

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = model.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    # bicubic interpolate relative_position_bias_table if not match
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2), mode="bicubic"
                    )
                    state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

        # bicubic interpolate absolute_pos_embed if not match
        absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
        for k in absolute_pos_embed_keys:
            # dpe
            absolute_pos_embed_pretrained = state_dict[k]
            absolute_pos_embed_current = model.state_dict()[k]
            _, L1, C1 = absolute_pos_embed_pretrained.size()
            _, L2, C2 = absolute_pos_embed_current.size()
            if C1 != C1:
                print(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                    absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(absolute_pos_embed_pretrained, size=(S2, S2), mode="bicubic")
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                    state_dict[k] = absolute_pos_embed_pretrained_resized

        # check classifier, if not match, then re-init classifier to zero
        head_bias_pretrained = state_dict["head.bias"]
        Nc1 = head_bias_pretrained.shape[0]
        Nc2 = model.head.bias.shape[0]
        if Nc1 != Nc2:
            if Nc1 == 21841 and Nc2 == 1000:
                print("loading ImageNet-22K weight to ImageNet-1K ......")
                map22kto1k_path = f"data/map22kto1k.txt"
                with open(map22kto1k_path) as f:
                    map22kto1k = f.readlines()
                map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                state_dict["head.weight"] = state_dict["head.weight"][map22kto1k, :]
                state_dict["head.bias"] = state_dict["head.bias"][map22kto1k]
            else:
                torch.nn.init.constant_(model.head.bias, 0.0)
                torch.nn.init.constant_(model.head.weight, 0.0)
                del state_dict["head.weight"]
                del state_dict["head.bias"]
                print(f"Error in loading classifier head, re-init classifier head to 0")

        model.load_state_dict(state_dict, strict=False)

        print(f"=> loaded successfully '{file_name}'")

        del checkpoint
        torch.cuda.empty_cache()
