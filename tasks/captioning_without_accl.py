import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import logging
from tqdm import tqdm
from models.modeling_captioners import AutoCaptioner
from typing import List
import json
import os
from torch.utils.data import DataLoader
from dataset.dataset import VideoTextDataset
import torch

# Initialize logger at module level
logger = logging.getLogger(__name__)

def setup_memory_optimization():
    """Set up memory optimization settings for PyTorch"""
    # Set memory management environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Set memory fraction to use less GPU memory
    if torch.cuda.is_available():
        # Use 90% of available memory to leave some headroom
        torch.cuda.set_per_process_memory_fraction(0.9)
        logger.info('Set GPU memory fraction to 90%')
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info('Enabled Flash Attention')
        except:
            logger.info('Flash Attention not available, using standard attention')


def get_dataloader(config_path: str, dataset_name: str, num_frames: int) -> DataLoader:
    # load data.config
    with open(config_path, 'r') as f:
        config = json.load(f)
    anno_path = config[dataset_name]['anno_path']
    data_root = config[dataset_name]['data_root']
    media_type = config[dataset_name]['media_type']
    assert media_type == 'video', 'media_type must be video'
    dataset = VideoTextDataset(
        anno_path=anno_path,
        data_root=data_root,
        num_frames=num_frames
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    return dataloader

def convert_list_to_dict(data: List[dict], index_key: str='video') -> dict:
    """
    Converts a list of dictionaries into a dictionary of dictionaries, using a specified key as the index.

    Args:
        data (List[dict]): A list of dictionaries to be converted.
        index_key (str): The key to be used as the index in the resulting dictionary. Defaults to 'video'.

    Returns:
        dict: A dictionary where each key is the value of the specified index_key from the input dictionaries,
              and each value is a dictionary containing the remaining key-value pairs from the input dictionaries.

    Raises:
        ValueError: If the specified index_key is not present in the keys of the input dictionaries.
    """
    keys = data[0].keys()
    if index_key not in keys:
        raise ValueError(f'Index key `{index_key}` not in keys')
    return {d[index_key]: {k: d[k] for k in keys if k != index_key} for d in data}

def gen_description(
    config_path: str,
    dataset_name: str,
    model_path: str,
    save_path: str = None,
    num_frames: int = 64,
) -> str:
    
    if os.path.exists(save_path):
        logger.info(f'{save_path} already exists. Skipping...')
        with open(save_path, 'r') as f:
            data = json.load(f)
        return data

    if model_path is None:
        raise ValueError('model_path must be provided if description.json does not exist')
    
    logger.info('Generating descriptions...')
    
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info('Cleared GPU cache before model loading')
    
    # Load model with device_map='auto' for multi-GPU support
    logger.info(f'Loading model from {model_path} with device_map="auto"')
    try:
        captioner = AutoCaptioner.from_pretrained(
            model_path, 
            is_llm=False,
            device_map='auto',  # Automatically distribute across available GPUs
            torch_dtype=torch.float16,  # Use half precision to save memory
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None  # Use flash attention if available
        )
        
        # Enable gradient checkpointing to save memory
        if hasattr(captioner, 'gradient_checkpointing_enable'):
            captioner.gradient_checkpointing_enable()
            logger.info('Enabled gradient checkpointing for memory efficiency')
            
    except Exception as e:
        logger.error(f'Failed to load model with flash attention, trying without: {e}')
        # Fallback without flash attention
        captioner = AutoCaptioner.from_pretrained(
            model_path, 
            is_llm=False,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        if hasattr(captioner, 'gradient_checkpointing_enable'):
            captioner.gradient_checkpointing_enable()
            logger.info('Enabled gradient checkpointing for memory efficiency')
    
    dataloader = get_dataloader(config_path, dataset_name, num_frames)
    data = []
    
    logger.info(f'Starting inference on {len(dataloader)} samples...')
    
    for i, batch in enumerate(tqdm(dataloader, desc="Generating captions")):
        try:
            d = []
            # Move video data to the same device as the model
            video_data = batch['video']
            
            # Determine the device to use (prefer GPU 1 if available to balance load)
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                device = torch.device(f'cuda:1' if i % 2 == 0 else 'cuda:0')
            elif torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            
            video_data = video_data.to(device)
            
            # Clear cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():  # Disable gradient computation to save memory
                preds = captioner.describe(video_data)
            
            for idx, gt, pred in zip(batch['idx'], batch['caption'], preds):
                d.append({'idx': idx.item(), 'pred': pred, 'gt': gt})
            data += d
            
            # Clear cache after each sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Log memory usage every 10 samples
            if (i + 1) % 10 == 0 and torch.cuda.is_available():
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                    logger.info(f'GPU {gpu_id} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB')
                    
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f'CUDA OOM at sample {i}: {e}')
            logger.info('Clearing cache and trying again...')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Skip this sample and continue
            continue
        except Exception as e:
            logger.error(f'Error processing sample {i}: {e}')
            continue

    # NOTE: `data` only contains 'idx', 'pred' and 'gt'
    # since __getitem__ in VideoTextDataset doesn't return events
    #
    # We need to get the original data from VideoTextDataset
    # and try to merge events to `data`
    #
    # We can't get events from __getitem__ directly since events may be None
    # and can't be collected in batch.
    
    data = convert_list_to_dict(data, index_key='idx')
    raw_data = convert_list_to_dict(dataloader.dataset.data, index_key='idx')

    events_none = 0
    objects_none = 0
    for k in data.keys():
        data[k]['events'] = raw_data[k].get('events', None)
        data[k]['objects'] = raw_data[k].get('objects', None)
        if data[k]['events'] is None:
            events_none += 1
        if data[k]['objects'] is None:
            objects_none += 1
    
    if events_none > 0:
        logger.info(f'No events found for {events_none} entries. Events will be extracted while evaluating.')
    else:
        logger.info('Events found for all entries. Events will not be extracted again.')
    
    if objects_none > 0:
        logger.info(f'No objects found for {objects_none} entries. Objects will be extracted while evaluating.')
    else:
        logger.info('Objects found for all entries. Objects will not be extracted again.')

    if save_path is not None:
        logger.info(f'Saving results to {save_path}')
        with open(save_path, 'w') as f:
            json.dump(data, f)
    return data
    
def evaluate_gpt(data, result_dir, api_endpoint, api_key, api_model, api_num_worker):
    logger.info('Evaluating GPT...')

    os.environ['AZURE_ENDPOINT'] = api_endpoint
    os.environ['OPENAI_API_KEY'] = api_key

    from utils.dream_gpt import DREAMGPTMetric

    metric = DREAMGPTMetric("TEST")
    metric.num_worker = api_num_worker
    metric.model = api_model

    dataset = []
    events_none = 0
    for idx, anno in data.items():
        data = {}
        data['idx'] = idx
        data['dataset'] = "overall"
        data['response'] = anno['gt']
        data['prediction'] = anno['pred']
        data['events'] = anno['events']
        data['objects'] = anno['objects']
        
        dataset.append(data)
    
    if events_none > 0:
        logger.warning(f'No events found for {events_none} entries. Events will be extracted while evaluating.')
        
    metric.process(dataset[:])
    metric._summarize_metric_by_subtask()

    os.makedirs(result_dir, exist_ok=True)
    metric.save_results(result_dir)
    metric.save_eval_infos(result_dir)

def set_logger(log_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    # add file handler to root logger
    logging.getLogger().addHandler(file_handler)


def main(
    config_path: str,
    dataset_name: str,
    model_path: str,
    save_dir: str,
    num_frames: int,
    evaluate: bool = True, # if not evaluate, just generate descriptions
    api_endpoint: str = None,
    api_key: str = None,
    api_model: str = None,
    api_num_worker: int = 10,
):
    
    os.makedirs(save_dir, exist_ok=True)
    DESCRIPTION_JSON_PATH = os.path.join(save_dir, 'description.json')
    LOGGING_PATH = os.path.join(save_dir, 'run.log')
    set_logger(LOGGING_PATH)
    
    # Set up memory optimization
    setup_memory_optimization()
    
    logger.info('********** Start Video Captioning Task (Without Accelerate) **********')
    logger.info(f'config_path: {config_path}')
    logger.info(f'dataset_name: {dataset_name}')
    logger.info(f'model_path: {model_path}')
    logger.info(f'save_dir: {save_dir}')
    logger.info(f'num_frames: {num_frames}')
    logger.info(f'api_model: {api_model}')
    logger.info(f'api_num_worker: {api_num_worker}')
    logger.info(f'api_endpoint: {api_endpoint}')
    logger.info(f'api_key: {api_key[:7] + "*" * (len(api_key) - 8) + api_key[-4:]}')
    
    # Check available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f'Found {num_gpus} GPU(s) available')
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f'GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)')
    else:
        logger.warning('No CUDA GPUs available, will use CPU')
    
    if evaluate and (api_endpoint is None or api_key is None):
        logger.error('api_endpoint and api_key must be provided')
        return
    
    data = gen_description(config_path, dataset_name, model_path, DESCRIPTION_JSON_PATH, num_frames)
    if evaluate:
        evaluate_gpt(data, save_dir, api_endpoint, api_key, api_model, api_num_worker)

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
