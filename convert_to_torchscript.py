import torch
import torchvision
from train import MaskDetector
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

modelpath = 'models/face_mask.ckpt'
# An instance of model.
model = MaskDetector()
model.load_state_dict(torch.load(modelpath, map_location=device)['state_dict'], strict=False)
print('[INFO] Model state loaded')

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 100, 100)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save('models/facemask_test_trace.pt')
print('[INFO] Traced model saved')