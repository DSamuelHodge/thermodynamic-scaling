"""
Module for loading and extracting weights from language models.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM
from tqdm.auto import tqdm
import os
import pickle


def extract_model_weights(model_name):
    """
    Extract weight matrices from a pre-trained model.
    
    Parameters:
    -----------
    model_name : str
        The name of the model to load (from HuggingFace models).
    
    Returns:
    --------
    dict
        Dictionary containing model weights organized by layer type.
    dict
        Dictionary containing layer metadata.
    """
    print(f"Loading model: {model_name}")
    try:
        # Try to load as a causal language model first
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except:
        # Fall back to base model if causal LM loading fails
        model = AutoModel.from_pretrained(model_name)
    
    # Move to CPU for consistent weight extraction
    model = model.cpu()
    
    weights = {}
    layer_info = {}
    
    # Get model architecture type
    model_type = model.config.model_type
    layer_info['model_type'] = model_type
    layer_info['hidden_size'] = model.config.hidden_size
    layer_info['num_layers'] = getattr(model.config, 'num_hidden_layers', 
                                       getattr(model.config, 'n_layer', 
                                               getattr(model.config, 'num_layers', None)))
    
    # Extract weights based on model architecture
    if model_type in ['gpt2', 'gpt_neo', 'gptj', 'opt', 'llama']:
        weights = _extract_decoder_only_weights(model)
    elif model_type in ['bert', 'roberta', 'distilbert']:
        weights = _extract_encoder_only_weights(model)
    elif model_type in ['t5', 'bart']:
        weights = _extract_encoder_decoder_weights(model)
    else:
        print(f"Warning: Specific extraction not implemented for {model_type}. Using generic extraction.")
        weights = _extract_generic_weights(model)
    
    return weights, layer_info


def _extract_decoder_only_weights(model):
    """Extract weights from decoder-only models (GPT-2, OPT, etc.)"""
    weights = {}
    
    # Determine the attribute name for transformer layers based on model type
    if hasattr(model, 'transformer'):
        transformer = model.transformer
        layers_attr = 'h' if hasattr(transformer, 'h') else 'layers'
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        transformer = model.model.decoder
        layers_attr = 'layers'
    else:
        raise ValueError("Could not locate transformer layers in model structure")
    
    # Get the layers
    layers = getattr(transformer, layers_attr)
    
    # Extract weights from each layer
    for i, layer in enumerate(layers):
        layer_weights = {}
        
        # Attention weights
        if hasattr(layer, 'attn'):
            attn = layer.attn
            layer_weights['self_attention.query'] = _get_weight_as_numpy(attn.q_proj.weight)
            layer_weights['self_attention.key'] = _get_weight_as_numpy(attn.k_proj.weight)
            layer_weights['self_attention.value'] = _get_weight_as_numpy(attn.v_proj.weight)
            layer_weights['self_attention.output'] = _get_weight_as_numpy(attn.out_proj.weight)
        elif hasattr(layer, 'attention'):
            attn = layer.attention
            if hasattr(attn, 'self'):
                attn = attn.self
            if hasattr(attn, 'q_proj'):
                layer_weights['self_attention.query'] = _get_weight_as_numpy(attn.q_proj.weight)
                layer_weights['self_attention.key'] = _get_weight_as_numpy(attn.k_proj.weight)
                layer_weights['self_attention.value'] = _get_weight_as_numpy(attn.v_proj.weight)
                layer_weights['self_attention.output'] = _get_weight_as_numpy(attn.o_proj.weight)
            elif hasattr(attn, 'query'):
                layer_weights['self_attention.query'] = _get_weight_as_numpy(attn.query.weight)
                layer_weights['self_attention.key'] = _get_weight_as_numpy(attn.key.weight)
                layer_weights['self_attention.value'] = _get_weight_as_numpy(attn.value.weight)
                if hasattr(attn, 'dense'):
                    layer_weights['self_attention.output'] = _get_weight_as_numpy(attn.dense.weight)
        
        # Feed-forward network weights
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            if hasattr(mlp, 'c_fc'):
                layer_weights['ffn.intermediate'] = _get_weight_as_numpy(mlp.c_fc.weight)
                layer_weights['ffn.output'] = _get_weight_as_numpy(mlp.c_proj.weight)
            elif hasattr(mlp, 'dense_h_to_4h'):
                layer_weights['ffn.intermediate'] = _get_weight_as_numpy(mlp.dense_h_to_4h.weight)
                layer_weights['ffn.output'] = _get_weight_as_numpy(mlp.dense_4h_to_h.weight)
        elif hasattr(layer, 'feed_forward'):
            ff = layer.feed_forward
            layer_weights['ffn.intermediate'] = _get_weight_as_numpy(ff.intermediate.dense.weight)
            layer_weights['ffn.output'] = _get_weight_as_numpy(ff.output.dense.weight)
        
        weights[f'layer_{i}'] = layer_weights
    
    # Add embeddings
    if hasattr(transformer, 'wte'):
        weights['embeddings.word'] = _get_weight_as_numpy(transformer.wte.weight)
    elif hasattr(transformer, 'embed_tokens'):
        weights['embeddings.word'] = _get_weight_as_numpy(transformer.embed_tokens.weight)
    
    return weights


def _extract_encoder_only_weights(model):
    """Extract weights from encoder-only models (BERT, RoBERTa, etc.)"""
    weights = {}
    
    # Determine the location of the encoder
    if hasattr(model, 'encoder'):
        encoder = model.encoder
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
        encoder = model.transformer.encoder
    else:
        raise ValueError("Could not locate encoder in model structure")
    
    # Get the layers
    if hasattr(encoder, 'layer'):
        layers = encoder.layer
    else:
        raise ValueError("Could not locate layers in encoder")
    
    # Extract weights from each layer
    for i, layer in enumerate(layers):
        layer_weights = {}
        
        # Attention weights
        attention = layer.attention
        if hasattr(attention, 'self'):
            self_attention = attention.self
            layer_weights['self_attention.query'] = _get_weight_as_numpy(self_attention.query.weight)
            layer_weights['self_attention.key'] = _get_weight_as_numpy(self_attention.key.weight)
            layer_weights['self_attention.value'] = _get_weight_as_numpy(self_attention.value.weight)
            layer_weights['self_attention.output'] = _get_weight_as_numpy(attention.output.dense.weight)
        
        # Feed-forward network weights
        intermediate = layer.intermediate
        output = layer.output
        layer_weights['ffn.intermediate'] = _get_weight_as_numpy(intermediate.dense.weight)
        layer_weights['ffn.output'] = _get_weight_as_numpy(output.dense.weight)
        
        weights[f'layer_{i}'] = layer_weights
    
    # Add embeddings
    if hasattr(model, 'embeddings'):
        weights['embeddings.word'] = _get_weight_as_numpy(model.embeddings.word_embeddings.weight)
    
    return weights


def _extract_encoder_decoder_weights(model):
    """Extract weights from encoder-decoder models (T5, BART, etc.)"""
    weights = {}
    
    # Extract encoder weights
    if hasattr(model, 'encoder'):
        encoder = model.encoder
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        encoder = model.model.encoder
    else:
        raise ValueError("Could not locate encoder in model structure")
    
    # Get the encoder layers
    if hasattr(encoder, 'block'):
        encoder_layers = encoder.block
    elif hasattr(encoder, 'layers'):
        encoder_layers = encoder.layers
    else:
        raise ValueError("Could not locate layers in encoder")
    
    # Extract encoder weights
    for i, layer in enumerate(encoder_layers):
        layer_weights = {}
        
        # Extract based on common encoder architectures
        if hasattr(layer, 'layer'):
            # T5-like
            if hasattr(layer.layer[0], 'SelfAttention'):
                attn = layer.layer[0].SelfAttention
                layer_weights['self_attention.query'] = _get_weight_as_numpy(attn.q.weight)
                layer_weights['self_attention.key'] = _get_weight_as_numpy(attn.k.weight)
                layer_weights['self_attention.value'] = _get_weight_as_numpy(attn.v.weight)
                layer_weights['self_attention.output'] = _get_weight_as_numpy(attn.o.weight)
            
            if len(layer.layer) > 1 and hasattr(layer.layer[1], 'DenseReluDense'):
                ffn = layer.layer[1].DenseReluDense
                layer_weights['ffn.intermediate'] = _get_weight_as_numpy(ffn.wi.weight)
                layer_weights['ffn.output'] = _get_weight_as_numpy(ffn.wo.weight)
        
        elif hasattr(layer, 'self_attn'):
            # BART-like
            attn = layer.self_attn
            layer_weights['self_attention.query'] = _get_weight_as_numpy(attn.q_proj.weight)
            layer_weights['self_attention.key'] = _get_weight_as_numpy(attn.k_proj.weight)
            layer_weights['self_attention.value'] = _get_weight_as_numpy(attn.v_proj.weight)
            layer_weights['self_attention.output'] = _get_weight_as_numpy(attn.out_proj.weight)
            
            layer_weights['ffn.intermediate'] = _get_weight_as_numpy(layer.fc1.weight)
            layer_weights['ffn.output'] = _get_weight_as_numpy(layer.fc2.weight)
        
        weights[f'encoder_layer_{i}'] = layer_weights
    
    # Extract decoder weights (similar pattern)
    if hasattr(model, 'decoder'):
        decoder = model.decoder
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        decoder = model.model.decoder
    
    if hasattr(decoder, 'block'):
        decoder_layers = decoder.block
    elif hasattr(decoder, 'layers'):
        decoder_layers = decoder.layers
    
    for i, layer in enumerate(decoder_layers):
        layer_weights = {}
        
        # Extract weights based on model architecture
        if hasattr(layer, 'layer'):
            # T5-like
            if hasattr(layer.layer[0], 'SelfAttention'):
                attn = layer.layer[0].SelfAttention
                layer_weights['self_attention.query'] = _get_weight_as_numpy(attn.q.weight)
                layer_weights['self_attention.key'] = _get_weight_as_numpy(attn.k.weight)
                layer_weights['self_attention.value'] = _get_weight_as_numpy(attn.v.weight)
                layer_weights['self_attention.output'] = _get_weight_as_numpy(attn.o.weight)
            
            if len(layer.layer) > 1 and hasattr(layer.layer[1], 'EncDecAttention'):
                cross_attn = layer.layer[1].EncDecAttention
                layer_weights['cross_attention.query'] = _get_weight_as_numpy(cross_attn.q.weight)
                layer_weights['cross_attention.key'] = _get_weight_as_numpy(cross_attn.k.weight)
                layer_weights['cross_attention.value'] = _get_weight_as_numpy(cross_attn.v.weight)
                layer_weights['cross_attention.output'] = _get_weight_as_numpy(cross_attn.o.weight)
            
            if len(layer.layer) > 2 and hasattr(layer.layer[2], 'DenseReluDense'):
                ffn = layer.layer[2].DenseReluDense
                layer_weights['ffn.intermediate'] = _get_weight_as_numpy(ffn.wi.weight)
                layer_weights['ffn.output'] = _get_weight_as_numpy(ffn.wo.weight)
        
        elif hasattr(layer, 'self_attn') and hasattr(layer, 'encoder_attn'):
            # BART-like
            self_attn = layer.self_attn
            layer_weights['self_attention.query'] = _get_weight_as_numpy(self_attn.q_proj.weight)
            layer_weights['self_attention.key'] = _get_weight_as_numpy(self_attn.k_proj.weight)
            layer_weights['self_attention.value'] = _get_weight_as_numpy(self_attn.v_proj.weight)
            layer_weights['self_attention.output'] = _get_weight_as_numpy(self_attn.out_proj.weight)
            
            cross_attn = layer.encoder_attn
            layer_weights['cross_attention.query'] = _get_weight_as_numpy(cross_attn.q_proj.weight)
            layer_weights['cross_attention.key'] = _get_weight_as_numpy(cross_attn.k_proj.weight)
            layer_weights['cross_attention.value'] = _get_weight_as_numpy(cross_attn.v_proj.weight)
            layer_weights['cross_attention.output'] = _get_weight_as_numpy(cross_attn.out_proj.weight)
            
            layer_weights['ffn.intermediate'] = _get_weight_as_numpy(layer.fc1.weight)
            layer_weights['ffn.output'] = _get_weight_as_numpy(layer.fc2.weight)
        
        weights[f'decoder_layer_{i}'] = layer_weights
    
    # Add embeddings
    if hasattr(model, 'shared'):
        weights['embeddings.word'] = _get_weight_as_numpy(model.shared.weight)
    elif hasattr(model, 'model') and hasattr(model.model, 'shared'):
        weights['embeddings.word'] = _get_weight_as_numpy(model.model.shared.weight)
    
    return weights


def _extract_generic_weights(model):
    """Fallback method to extract weights from any model using a generic traversal approach"""
    weights = {}
    
    # Collect all weight matrices
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:  # Only consider 2D weight matrices
            # Create a hierarchical structure based on the parameter name
            parts = name.split('.')
            if 'query' in parts or 'key' in parts or 'value' in parts or 'q_proj' in parts or 'k_proj' in parts or 'v_proj' in parts:
                weight_type = 'attention'
            elif 'mlp' in parts or 'feed_forward' in parts or 'ffn' in parts or 'fc' in parts:
                weight_type = 'ffn'
            elif 'embed' in parts:
                weight_type = 'embedding'
            else:
                weight_type = 'other'
            
            # Create a clean name
            clean_name = f"{weight_type}.{'.'.join(parts[-3:])}"
            weights[clean_name] = _get_weight_as_numpy(param)
    
    return weights


def _get_weight_as_numpy(weight_tensor):
    """Convert PyTorch tensor to NumPy array"""
    return weight_tensor.detach().cpu().numpy()


def load_models(model_names, cache_dir=None):
    """
    Load multiple models and extract their weights.
    
    Parameters:
    -----------
    model_names : list of str
        Names of models to load.
    cache_dir : str, optional
        Directory to cache the extracted weights.
    
    Returns:
    --------
    dict
        Dictionary of model weights and metadata.
    """
    model_weights = {}
    
    for model_name in tqdm(model_names, desc="Loading models"):
        # Check if cached version exists
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{model_name.replace('/', '_')}_weights.pkl")
            
            if os.path.exists(cache_path):
                print(f"Loading cached weights for {model_name}")
                with open(cache_path, 'rb') as f:
                    model_weights[model_name] = pickle.load(f)
                continue
        
        # Extract weights
        weights, layer_info = extract_model_weights(model_name)
        model_weights[model_name] = {'weights': weights, 'layer_info': layer_info}
        
        # Cache the weights
        if cache_dir is not None:
            with open(cache_path, 'wb') as f:
                pickle.dump(model_weights[model_name], f)
    
    return model_weights