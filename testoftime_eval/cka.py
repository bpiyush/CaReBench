import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


class CKAAnalyzer:
    """
    Centered Kernel Alignment (CKA) analysis for comparing representations
    between base and fine-tuned models.
    """
    
    def __init__(self, base_model: nn.Module, finetuned_model: nn.Module, device: str = 'cuda'):
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.device = device
        
        self.base_model.eval()
        self.finetuned_model.eval()
        
        self.base_activations = {}
        self.finetuned_activations = {}
        self.hooks = []
    
    def _get_hook(self, name: str, storage: dict):
        """Create a hook function to capture activations."""
        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                # LlamaDecoderLayer returns (hidden_states, ...) 
                act = output[0]
            else:
                act = output
            
            # Store activations (detach and move to CPU to save GPU memory)
            # Flatten to (batch * seq_len, hidden_dim) for CKA
            storage[name] = act.detach().cpu().float()
        return hook
    
    def _register_hooks(self, model: nn.Module, storage: dict, layer_names: List[str] = None):
        """Register forward hooks on specified layers."""
        hooks = []
        
        for name, module in model.named_modules():
            # Hook into decoder layers and embedding
            should_hook = False
            
            if 'layers.' in name and name.endswith(('layers.' + name.split('layers.')[-1].split('.')[0])):
                # This matches "model.layers.0", "model.layers.1", etc.
                should_hook = True
            elif name == 'model.layers':
                continue  # Skip the ModuleList itself
            elif 'layers.' in name:
                # Check if it's a direct layer (e.g., model.layers.0) not a submodule
                parts = name.split('.')
                if len(parts) == 3 and parts[1] == 'layers' and parts[2].isdigit():
                    should_hook = True
            
            # Also hook embeddings and final norm
            if name in ['model.embed_tokens', 'model.norm']:
                should_hook = True
                
            if layer_names is not None:
                should_hook = name in layer_names
            
            if should_hook:
                hook = module.register_forward_hook(self._get_hook(name, storage))
                hooks.append(hook)
        
        return hooks
    
    def _register_layer_hooks(self, model: nn.Module, storage: dict):
        """Register hooks specifically on LlamaDecoderLayer modules."""
        hooks = []
        
        # Hook embedding layer
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            hook = model.model.embed_tokens.register_forward_hook(
                self._get_hook('embed_tokens', storage)
            )
            hooks.append(hook)
        
        # Hook each decoder layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for idx, layer in enumerate(model.model.layers):
                hook = layer.register_forward_hook(
                    self._get_hook(f'layer_{idx}', storage)
                )
                hooks.append(hook)
        
        # Hook final norm
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            hook = model.model.norm.register_forward_hook(
                self._get_hook('final_norm', storage)
            )
            hooks.append(hook)
        
        return hooks
    
    def _remove_hooks(self, hooks: list):
        """Remove all hooks."""
        for hook in hooks:
            hook.remove()
    
    @staticmethod
    def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute linear CKA between two activation matrices.
        
        Args:
            X: (n_samples, n_features_x) activation matrix
            Y: (n_samples, n_features_y) activation matrix
        
        Returns:
            CKA similarity score in [0, 1]
        """
        # Center the activations
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute CKA
        XTX = X.T @ X
        YTY = Y.T @ Y
        YTX = Y.T @ X
        
        # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        numerator = (YTX ** 2).sum()
        denominator = torch.sqrt((XTX ** 2).sum() * (YTY ** 2).sum())
        
        if denominator < 1e-10:
            return 0.0
        
        cka = (numerator / denominator).item()
        return cka
    
    @staticmethod
    def linear_CKA_batched(X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Memory-efficient CKA using the kernel formulation.
        Better for when n_features >> n_samples.
        
        Uses: CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
        where K = X @ X^T and L = Y @ Y^T
        """
        n = X.shape[0]
        
        # Center the activations
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute Gram matrices
        K = X @ X.T  # (n, n)
        L = Y @ Y.T  # (n, n)
        
        # Centering matrix H = I - 1/n * 11^T
        # HSIC(K, L) = 1/(n-1)^2 * tr(KHLH) = 1/(n-1)^2 * tr(K_c @ L_c)
        # where K_c and L_c are centered Gram matrices
        
        H = torch.eye(n, device=X.device, dtype=X.dtype) - torch.ones(n, n, device=X.device, dtype=X.dtype) / n
        
        K_centered = H @ K @ H
        L_centered = H @ L @ H
        
        hsic_kl = (K_centered * L_centered).sum()
        hsic_kk = (K_centered * K_centered).sum()
        hsic_ll = (L_centered * L_centered).sum()
        
        if hsic_kk < 1e-10 or hsic_ll < 1e-10:
            return 0.0
        
        cka = (hsic_kl / torch.sqrt(hsic_kk * hsic_ll)).item()
        return cka
    
    @torch.no_grad()
    def collect_activations(self, dataloader, max_batches: int = 10, max_tokens: int = 5000):
        """
        Collect activations from both models on the same data.
        
        Args:
            dataloader: DataLoader yielding input dicts compatible with model.forward()
            max_batches: Maximum number of batches to process
            max_tokens: Maximum total tokens to collect (for memory management)
        """
        self.base_activations = {}
        self.finetuned_activations = {}
        
        # Temporary storage for this run
        base_acts_list = {}
        fine_acts_list = {}
        
        # Register hooks
        base_hooks = self._register_layer_hooks(self.base_model, self.base_activations)
        fine_hooks = self._register_layer_hooks(self.finetuned_model, self.finetuned_activations)
        
        total_tokens = 0
        
        try:
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting activations", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                
                # Move batch to device
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch.items()}
                else:
                    inputs = {'input_ids': batch[0].to(self.device), 
                             'attention_mask': batch[1].to(self.device)}
                
                # Forward pass through both models
                self.base_activations.clear()
                self.finetuned_activations.clear()
                
                _ = self.base_model(**inputs)
                _ = self.finetuned_model(**inputs)
                
                # Accumulate activations
                for name in self.base_activations:
                    base_act = self.base_activations[name]
                    fine_act = self.finetuned_activations[name]
                    
                    # Flatten batch and sequence dimensions
                    base_flat = base_act.reshape(-1, base_act.shape[-1])
                    fine_flat = fine_act.reshape(-1, fine_act.shape[-1])
                    
                    if name not in base_acts_list:
                        base_acts_list[name] = []
                        fine_acts_list[name] = []
                    
                    base_acts_list[name].append(base_flat)
                    fine_acts_list[name].append(fine_flat)
                
                total_tokens += base_act.shape[0] * base_act.shape[1]
                
                if total_tokens >= max_tokens:
                    print(f"Reached {total_tokens} tokens, stopping collection.")
                    break
        
        finally:
            self._remove_hooks(base_hooks)
            self._remove_hooks(fine_hooks)
        
        # Concatenate all activations
        self.base_activations_concat = {
            name: torch.cat(acts, dim=0) for name, acts in base_acts_list.items()
        }
        self.finetuned_activations_concat = {
            name: torch.cat(acts, dim=0) for name, acts in fine_acts_list.items()
        }
        
        print(f"Collected activations for {len(self.base_activations_concat)} layers")
        print(f"Total samples per layer: {self.base_activations_concat[list(self.base_activations_concat.keys())[0]].shape[0]}")
    
    @torch.no_grad()
    def collect_activations_from_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Simpler interface: collect activations from raw input tensors.
        
        Args:
            input_ids: (batch, seq_len) input token IDs
            attention_mask: optional attention mask
        """
        self.base_activations = {}
        self.finetuned_activations = {}
        
        # Register hooks
        base_hooks = self._register_layer_hooks(self.base_model, self.base_activations)
        fine_hooks = self._register_layer_hooks(self.finetuned_model, self.finetuned_activations)
        
        try:
            inputs = {'input_ids': input_ids.to(self.device)}
            if attention_mask is not None:
                inputs['attention_mask'] = attention_mask.to(self.device)
            
            _ = self.base_model(**inputs)
            _ = self.finetuned_model(**inputs)
            
            # Flatten and store
            self.base_activations_concat = {}
            self.finetuned_activations_concat = {}
            
            for name in self.base_activations:
                base_act = self.base_activations[name]
                fine_act = self.finetuned_activations[name]
                
                self.base_activations_concat[name] = base_act.reshape(-1, base_act.shape[-1])
                self.finetuned_activations_concat[name] = fine_act.reshape(-1, fine_act.shape[-1])
        
        finally:
            self._remove_hooks(base_hooks)
            self._remove_hooks(fine_hooks)
    
    def compute_cka_matrix(self, subsample: int = None) -> Tuple[np.ndarray, List[str]]:
        """
        Compute full CKA matrix between all layer pairs.
        
        Returns:
            cka_matrix: (n_layers, n_layers) matrix where [i,j] is CKA(base_layer_i, fine_layer_j)
            layer_names: List of layer names
        """
        layer_names = sorted(self.base_activations_concat.keys(), 
                            key=lambda x: int(x.split('_')[1]) if x.startswith('layer_') else -1 if x == 'embed_tokens' else 100)
        
        n_layers = len(layer_names)
        cka_matrix = np.zeros((n_layers, n_layers))
        
        for i, name_i in enumerate(tqdm(layer_names, desc="Computing CKA matrix")):
            base_act = self.base_activations_concat[name_i]
            
            # Subsample if requested (for memory/speed)
            if subsample and base_act.shape[0] > subsample:
                indices = torch.randperm(base_act.shape[0])[:subsample]
                base_act = base_act[indices]
            
            for j, name_j in enumerate(layer_names):
                fine_act = self.finetuned_activations_concat[name_j]
                
                if subsample and fine_act.shape[0] > subsample:
                    if i == 0:  # Use same indices for consistency
                        fine_act = fine_act[indices]
                    else:
                        fine_act = fine_act[indices]
                
                cka_matrix[i, j] = self.linear_CKA_batched(base_act, fine_act)
        
        return cka_matrix, layer_names
    
    def compute_same_layer_cka(self, subsample: int = None) -> Dict[str, float]:
        """
        Compute CKA only between corresponding layers (diagonal).
        Faster than full matrix computation.
        """
        results = {}
        
        for name in tqdm(self.base_activations_concat.keys(), desc="Computing same-layer CKA"):
            base_act = self.base_activations_concat[name]
            fine_act = self.finetuned_activations_concat[name]
            
            if subsample and base_act.shape[0] > subsample:
                indices = torch.randperm(base_act.shape[0])[:subsample]
                base_act = base_act[indices]
                fine_act = fine_act[indices]
            
            results[name] = self.linear_CKA_batched(base_act, fine_act)
        
        return results
    
    def plot_cka_matrix(self, cka_matrix: np.ndarray, layer_names: List[str], 
                        title: str = "CKA: Base vs Fine-tuned"):
        """Plot the full CKA matrix as a heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(cka_matrix, cmap='magma', vmin=0, vmax=1)
        
        # Labels
        tick_labels = [name.replace('layer_', 'L').replace('embed_tokens', 'Emb').replace('final_norm', 'Norm') 
                      for name in layer_names]
        
        # Reduce tick density if many layers
        n_layers = len(layer_names)
        if n_layers > 20:
            tick_step = max(1, n_layers // 10)
            ax.set_xticks(range(0, n_layers, tick_step))
            ax.set_yticks(range(0, n_layers, tick_step))
            ax.set_xticklabels([tick_labels[i] for i in range(0, n_layers, tick_step)], rotation=45, ha='right')
            ax.set_yticklabels([tick_labels[i] for i in range(0, n_layers, tick_step)])
        else:
            ax.set_xticks(range(n_layers))
            ax.set_yticks(range(n_layers))
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax.set_yticklabels(tick_labels)
        
        ax.set_xlabel('Fine-tuned Model Layer', fontsize=12)
        ax.set_ylabel('Base Model Layer', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('CKA Similarity', fontsize=11)
        
        # Add diagonal line for reference
        ax.plot([-0.5, n_layers-0.5], [-0.5, n_layers-0.5], 'w--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        plt.savefig('cka_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return fig
    
    def plot_same_layer_cka(self, cka_results: Dict[str, float], title: str = "Same-Layer CKA"):
        """Plot CKA scores for corresponding layers."""
        
        # Sort by layer number
        sorted_items = sorted(cka_results.items(), 
                             key=lambda x: int(x[0].split('_')[1]) if x[0].startswith('layer_') else -1 if x[0] == 'embed_tokens' else 100)
        
        names, scores = zip(*sorted_items)
        
        # Clean names for display
        display_names = [name.replace('layer_', 'L').replace('embed_tokens', 'Emb').replace('final_norm', 'Norm') 
                        for name in names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        colors = ['#e74c3c' if s < 0.8 else '#f39c12' if s < 0.95 else '#2ecc71' for s in scores]
        bars = ax1.bar(range(len(scores)), scores, color=colors, edgecolor='white', linewidth=0.5)
        
        ax1.set_xticks(range(len(display_names)))
        ax1.set_xticklabels(display_names, rotation=45, ha='right')
        ax1.set_ylabel('CKA Similarity', fontsize=11)
        ax1.set_xlabel('Layer', fontsize=11)
        ax1.set_title(f'{title}: Bar Plot', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.05)
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.9, color='orange', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            if score < 0.95:  # Only label interesting ones
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Line plot showing "representational drift"
        layer_indices = list(range(len(scores)))
        ax2.plot(layer_indices, scores, 'o-', color='#3498db', linewidth=2, markersize=6)
        ax2.fill_between(layer_indices, scores, alpha=0.3, color='#3498db')
        
        ax2.set_xticks(range(len(display_names)))
        ax2.set_xticklabels(display_names, rotation=45, ha='right')
        ax2.set_ylabel('CKA Similarity', fontsize=11)
        ax2.set_xlabel('Layer', fontsize=11)
        ax2.set_title(f'{title}: Representational Drift', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        # Annotate min CKA
        min_idx = np.argmin(scores)
        ax2.annotate(f'Min: {scores[min_idx]:.3f}\n({display_names[min_idx]})',
                    xy=(min_idx, scores[min_idx]),
                    xytext=(min_idx + 2, scores[min_idx] - 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig('cka_same_layer.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return fig


# =============================================================================
# Usage Example
# =============================================================================

def run_cka_analysis(base_model, finetuned_model, tokenizer, 
                     sample_texts: List[str] = None, device: str = 'cuda'):
    """
    Convenience function to run full CKA analysis.
    
    Args:
        base_model: Base LlamaForCausalLM
        finetuned_model: Fine-tuned LlamaForCausalLM  
        tokenizer: Tokenizer for the models
        sample_texts: List of text samples to use for computing activations
        device: Device to run on
    """
    if sample_texts is None:
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A photograph of a sunset over the ocean with vibrant colors.",
            "The scientist discovered a new species in the rainforest.",
            "Abstract painting with geometric shapes and bold colors.",
            "The stock market experienced significant volatility today.",
        ] * 10  # Repeat for more samples
    
    # Tokenize
    inputs = tokenizer(sample_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # Create analyzer
    analyzer = CKAAnalyzer(base_model, finetuned_model, device=device)
    
    # Collect activations
    print("Collecting activations...")
    analyzer.collect_activations_from_inputs(inputs['input_ids'], inputs['attention_mask'])
    
    # Compute same-layer CKA (fast)
    print("\nComputing same-layer CKA...")
    same_layer_cka = analyzer.compute_same_layer_cka(subsample=1000)
    
    print("\nSame-layer CKA results:")
    for name, score in sorted(same_layer_cka.items(), 
                              key=lambda x: int(x[0].split('_')[1]) if x[0].startswith('layer_') else -1):
        print(f"  {name}: {score:.4f}")
    
    # Plot same-layer CKA
    analyzer.plot_same_layer_cka(same_layer_cka, title="Base vs Fine-tuned LM")
    
    # Optionally compute full matrix (slower)
    compute_full = input("\nCompute full CKA matrix? (y/n): ").lower() == 'y'
    if compute_full:
        print("Computing full CKA matrix...")
        cka_matrix, layer_names = analyzer.compute_cka_matrix(subsample=1000)
        analyzer.plot_cka_matrix(cka_matrix, layer_names, title="CKA: Base vs Fine-tuned Llama")
    
    return analyzer, same_layer_cka


# Quick usage:
# analyzer, cka_scores = run_cka_analysis(model_base.model.language_model, 
#                                          model_fine.model.language_model,
#                                          tokenizer)