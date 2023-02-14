from src.utils import get_device, OPTIMIZER_FACTORY

from toxsmi.models import MODEL_FACTORY

def setup_model(params):
    device = get_device()
    model = MODEL_FACTORY[params.get('model_fn', 'dense')](params).to(device)
    return model