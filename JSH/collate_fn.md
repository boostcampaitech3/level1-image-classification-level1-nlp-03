# Custom_Dataset_및_Custom_DataLoader_생성 

## `my_collate_fn` 해결

### 코드

```python
def my_collate_fn(samples):
    collate_X = []
    collate_y = []

    max_len = max([len(sample['X']) for sample in samples])

    for _x in [sample['X'] for sample in samples]:
      tensor_len = _x.size(dim=0)
      p2d = (0, max_len - tensor_len)
      _x = F.pad(_x, p2d)

      collate_X.append(_x)
      collate_y.append(_x[0])

    return {'X': torch.stack(collate_X),
            'y': torch.stack(collate_y)}
```

<br><br>

### 코드 설명
파라미터 `samples` 는 배치 사이즈만큼의 데이터가 들어옵니다. 과제는 배치 사이즈에 맞게끔 0으로 채워넣어줘야 되는 것입니다.

한줄 한줄 보면 `max_len` 은 배치 사이즈만큼 들어온 데이터 중에서 가장 길이가 긴 데이터의 길이를 저장합니다. 
예를 들면 배치 사이즈 3인 상태에서 아래 데이터가 들어왔다고 가정합니다.

```
tensor([[0.]])
tensor([[1., 1.]])
tensor([[2., 2., 2.]]) -> 길이가 가장 길음
```

그렇다면 `max_len` 은 3이 됩니다. 

<br><br>

이제 반목문을 돌려 현재 텐서의 길이를 알아냅니다. 그 다음 0으로 채워주는 함수인
pad 함수를 씁니다. [TORCH.NN.FUNCTIONAL.PAD](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html?highlight=pad#torch.nn.functional.pad)

`max_len` 에서 `tensor_len` 을 빼면 0으로 몇개를 채워넣어야 하는지 알 수 있습니다. 알아낸 개수만큼 `F.pad` 를 통해 채워주고 해당 데이터를 바꿉니다. 그렇게 0으로 채운 데이터들을 합쳐주면 해결됩니다. 

<br><br>

### 제가 생각하는 과제 답보다 좋은 점

코드 자체가 꽤 간결해지고 이해하기 쉬워집니다. 기존 답처럼 `diff` 에 따라 0을 채울지 말지 결정하지 않고 `pad` 함수가 알아서 0으로 채우고 `max_len - tensor_len` 이 0 이면 그냥 기존 값이 유지됩니다. 