# DisCLE-BAR
## Environment Setup

To ensure the code runs correctly, please make sure your environment meets the following dependencies:

- `einops==0.7.0`
- `numpy==1.24.4`
- `pandas==1.2.4`
- `torch==2.0.1`
- `torchaudio==2.0.2`
- `torchvision==0.15.2`
- `tqdm==4.66.5`

### Installing Dependencies

You can install all dependencies using the following command:

```bash
pip install einops==0.7.0 numpy==1.24.4 pandas==1.2.4 torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2 tqdm==4.66.5
```
## Quick start
1. Teacher model training：
   ```bash
   python unet_train.py
   ```
2. Phase I training：
   ```bash
   python p1.train.py
   ```
3. Phase II training：
   ```bash
   python p2.train.py
   ```
