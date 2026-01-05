# src/models/rag_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional, Any
from .manifold_layers import ManifoldHyperConnection
import numpy as np


class KnowledgeBase:
    """
    Vector-based knowledge base for vision-language tasks.
    
    Stores and retrieves knowledge items relevant to visual content.
    Uses efficient similarity search for retrieval.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        max_items: int = 10000,
        device: str = 'cuda'
    ):
        self.embedding_dim = embedding_dim
        self.max_items = max_items
        self.device = device
        
        # Knowledge storage
        self.items = []  # List of knowledge texts
        self.embeddings = torch.zeros((0, embedding_dim), device=device)
        
        # Embedding model (simplified - in practice use CLIP or similar)
        self.embedding_model = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        ).to(device)
        
        # Initialize with common knowledge
        self._initialize_knowledge()
    
    def _initialize_knowledge(self):
        """Initialize with common object knowledge."""
        # Common object descriptions (COCO classes)
        common_knowledge = [
            # People and animals
            "person: a human being, can be standing, sitting, walking, or running",
            "bicycle: a vehicle with two wheels, powered by pedals",
            "car: a four-wheeled motor vehicle used for transportation",
            "motorcycle: a two-wheeled vehicle with an engine",
            "airplane: a powered flying vehicle with fixed wings",
            "bus: a large motor vehicle carrying passengers by road",
            "train: a series of connected railway cars",
            "truck: a large motor vehicle for transporting goods",
            "boat: a small vessel for traveling on water",
            
            # Traffic objects
            "traffic light: a signaling device at road intersections",
            "fire hydrant: a connection point for firefighting equipment",
            "stop sign: a red octagonal traffic sign",
            "parking meter: a device for collecting parking fees",
            
            # Animals
            "bird: a warm-blooded egg-laying vertebrate with feathers",
            "cat: a small domesticated carnivorous mammal",
            "dog: a domesticated carnivorous mammal",
            "horse: a large domesticated mammal used for riding",
            "sheep: a domesticated ruminant animal with woolly coat",
            "cow: a large domesticated ruminant animal",
            "elephant: a very large herbivorous mammal with a trunk",
            "bear: a large heavy mammal with thick fur",
            "zebra: an African wild horse with black and white stripes",
            "giraffe: a very tall African mammal with a long neck",
            
            # Everyday objects
            "backpack: a bag carried on the back",
            "umbrella: a device for protection against rain or sun",
            "handbag: a small bag used by women for personal items",
            "tie: a long piece of cloth worn around the neck",
            "suitcase: a rectangular case for carrying clothes",
            
            # Sports equipment
            "frisbee: a plastic disc thrown between players",
            "skis: long narrow runners for gliding over snow",
            "snowboard: a board for gliding on snow",
            "sports ball: a ball used in various sports",
            "kite: a light frame covered with fabric flown in wind",
            "baseball bat: a club used in baseball to hit the ball",
            "baseball glove: a leather glove for catching baseball",
            "skateboard: a board with wheels for riding",
            "surfboard: a board for riding ocean waves",
            "tennis racket: an implement for hitting tennis balls",
            
            # Food and drink
            "bottle: a container with a narrow neck for liquids",
            "wine glass: a glass for drinking wine",
            "cup: a small open container for drinking",
            "fork: a utensil with prongs for eating",
            "knife: a utensil with a sharp blade for cutting",
            "spoon: a utensil with a small shallow bowl for eating",
            "bowl: a round deep dish for food",
            "banana: a long curved fruit with yellow skin",
            "apple: a round fruit with red or green skin",
            "sandwich: food consisting of fillings between bread",
            "orange: a round citrus fruit with bright orange skin",
            "broccoli: a green vegetable with tree-like shape",
            "carrot: a long orange root vegetable",
            "hot dog: a cooked sausage served in a sliced bun",
            "pizza: a dish with flat bread base and toppings",
            "donut: a small fried cake of sweetened dough",
            "cake: a sweet baked dessert",
            
            # Furniture
            "chair: a seat for one person with back support",
            "couch: a long upholstered piece of furniture for seating",
            "potted plant: a plant growing in a container",
            "bed: a piece of furniture for sleeping",
            "dining table: a table at which meals are eaten",
            "toilet: a fixture for disposal of human waste",
            
            # Electronics
            "tv: a device for receiving television signals",
            "laptop: a portable computer",
            "mouse: a small device for controlling computer cursor",
            "remote: a device for controlling electronic equipment",
            "keyboard: a set of keys for operating a computer",
            "cell phone: a portable telephone",
            "microwave: an oven that cooks food with microwaves",
            "oven: an enclosed compartment for baking or roasting",
            "toaster: an electrical appliance for browning bread",
            "sink: a basin with water supply and drain",
            "refrigerator: an appliance for keeping food cold",
            
            # Miscellaneous
            "book: a written or printed work consisting of pages",
            "clock: an instrument for measuring and indicating time",
            "vase: a decorative container for cut flowers",
            "scissors: a cutting instrument with two blades",
            "teddy bear: a stuffed toy bear",
            "hair drier: an electrical device for drying hair",
            "toothbrush: a small brush for cleaning teeth"
        ]
        
        # Add initial knowledge
        for item in common_knowledge:
            self.add_knowledge(item)
    
    def add_knowledge(self, text: str, embedding: Optional[torch.Tensor] = None):
        """
        Add knowledge item to the base.
        
        Args:
            text: Knowledge text
            embedding: Optional precomputed embedding
        """
        if len(self.items) >= self.max_items:
            # Remove oldest items (FIFO)
            self.items.pop(0)
            self.embeddings = self.embeddings[1:]
        
        self.items.append(text)
        
        if embedding is not None:
            new_embedding = embedding.unsqueeze(0).to(self.device)
        else:
            # Generate random embedding (in practice, use text encoder)
            new_embedding = torch.randn(1, self.embedding_dim, device=self.device)
            new_embedding = F.normalize(new_embedding, dim=1)
        
        self.embeddings = torch.cat([self.embeddings, new_embedding], dim=0)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to embedding.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding
        """
        # Simplified text encoding
        # In practice, use a proper text encoder like BERT or CLIP
        
        # Create a hash-based embedding
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        
        # Generate deterministic "random" embedding
        torch.manual_seed(hash_val % (2**32))
        embedding = torch.randn(self.embedding_dim, device=self.device)
        embedding = F.normalize(embedding, dim=0)
        
        return embedding.unsqueeze(0)
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[str, float, torch.Tensor]]:
        """
        Retrieve relevant knowledge items.
        
        Args:
            query_embedding: Query embedding [1, D] or [D]
            top_k: Number of items to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (text, similarity_score, embedding)
        """
        if len(self.items) == 0:
            return []
        
        # Ensure query is 2D
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        # Normalize query
        query_norm = F.normalize(query_embedding, dim=1)
        knowledge_norm = F.normalize(self.embeddings, dim=1)
        
        # Compute cosine similarity
        similarities = torch.matmul(query_norm, knowledge_norm.T)  # [1, N]
        similarities = similarities.squeeze(0)  # [N]
        
        # Get top-k similar items
        top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(self.items)))
        
        # Filter by threshold
        results = []
        for score, idx in zip(top_scores, top_indices):
            if score >= similarity_threshold:
                text = self.items[idx]
                embedding = self.embeddings[idx]
                results.append((text, score.item(), embedding))
        
        return results
    
    def retrieve_by_text(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve knowledge by text query.
        
        Args:
            query_text: Text query
            top_k: Number of items to retrieve
            
        Returns:
            List of (text, similarity_score)
        """
        query_embedding = self.encode_text(query_text)
        results = self.retrieve(query_embedding, top_k=top_k)
        
        # Return only text and score
        return [(text, score) for text, score, _ in results]


class RAGVisionKnowledge(nn.Module):
    """
    Retrieval-Augmented Generation for vision tasks.
    
    Enhances visual understanding with external knowledge.
    """
    
    def __init__(
        self,
        visual_dim: int = 256,
        knowledge_dim: int = 512,
        hidden_dim: int = 256,
        num_retrievals: int = 5,
        use_mhc: bool = True
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.knowledge_dim = knowledge_dim
        self.hidden_dim = hidden_dim
        self.num_retrievals = num_retrievals
        self.use_mhc = use_mhc
        
        # Knowledge base
        self.knowledge_base = KnowledgeBase(
            embedding_dim=knowledge_dim,
            max_items=10000
        )
        
        # Query projector (visual features → knowledge query)
        self.query_projector = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, knowledge_dim)
        )
        
        # Knowledge projector (knowledge → visual space)
        self.knowledge_projector = nn.Sequential(
            nn.Linear(knowledge_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, visual_dim)
        )
        
        # Attention for knowledge fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # MHC for knowledge enhancement
        if use_mhc:
            self.mhc_fusion = ManifoldHyperConnection(
                input_dim=visual_dim * 2,  # Visual + Knowledge
                expansion_rate=2
            )
        else:
            self.mhc_fusion = nn.Identity()
        
        # Output projection
        self.output_projection = nn.Linear(visual_dim * 2, visual_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(visual_dim)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_query: Optional[str] = None,
        return_knowledge: bool = False
    ) -> torch.Tensor:
        """
        Enhance visual features with knowledge.
        
        Args:
            visual_features: Visual features [B, D_visual] or [B, *, D_visual]
            text_query: Optional text query for knowledge retrieval
            return_knowledge: Whether to return retrieved knowledge
            
        Returns:
            Knowledge-enhanced features
        """
        original_shape = visual_features.shape
        needs_reshape = len(original_shape) > 2
        
        if needs_reshape:
            # Flatten spatial dimensions
            B, *spatial, D = original_shape
            visual_features = visual_features.reshape(B, -1, D)
        
        B, N, D = visual_features.shape
        
        # === Knowledge Retrieval ===
        # Generate query from visual features
        if text_query is None:
            # Use pooled visual features as query
            visual_pooled = visual_features.mean(dim=1)  # [B, D]
            query = self.query_projector(visual_pooled)  # [B, knowledge_dim]
        else:
            # Use text query
            query = self.knowledge_base.encode_text(text_query)
            query = query.expand(B, -1)  # [B, knowledge_dim]
        
        # Retrieve relevant knowledge
        retrieved_knowledge = []
        for b in range(B):
            results = self.knowledge_base.retrieve(
                query[b:b+1],
                top_k=self.num_retrievals
            )
            
            if results:
                # Get embeddings from results
                embeddings = torch.stack([emb for _, _, emb in results])  # [K, knowledge_dim]
                
                # Project to visual space
                knowledge_visual = self.knowledge_projector(embeddings)  # [K, visual_dim]
                retrieved_knowledge.append(knowledge_visual)
            else:
                # No knowledge retrieved, use zeros
                knowledge_visual = torch.zeros(
                    self.num_retrievals, D,
                    device=visual_features.device
                )
                retrieved_knowledge.append(knowledge_visual)
        
        # === Knowledge Fusion ===
        enhanced_features_list = []
        
        for b in range(B):
            visual_batch = visual_features[b:b+1]  # [1, N, D]
            knowledge_batch = retrieved_knowledge[b]  # [K, D]
            
            if knowledge_batch.sum() == 0:
                # No knowledge, return original features
                enhanced_features_list.append(visual_batch)
                continue
            
            # Repeat knowledge for each spatial position
            knowledge_expanded = knowledge_batch.unsqueeze(0)  # [1, K, D]
            knowledge_expanded = knowledge_expanded.expand(1, N, -1, D)  # [1, N, K, D]
            knowledge_expanded = knowledge_expanded.reshape(1, N * self.num_retrievals, D)  # [1, N*K, D]
            
            # Cross-attention between visual features and knowledge
            attended, _ = self.attention(
                query=visual_batch,
                key=knowledge_expanded,
                value=knowledge_expanded
            )  # [1, N, D]
            
            # Concatenate with original features
            combined = torch.cat([visual_batch, attended], dim=-1)  # [1, N, 2*D]
            
            # MHC fusion
            combined_flat = combined.reshape(-1, 2 * D)
            combined_fused = self.mhc_fusion(combined_flat)
            combined_fused = combined_fused.reshape(1, N, 2 * D)
            
            # Output projection
            output = self.output_projection(combined_fused)  # [1, N, D]
            
            # Residual connection and normalization
            output = output + visual_batch
            output = self.norm(output)
            
            enhanced_features_list.append(output)
        
        # Combine batch
        enhanced_features = torch.cat(enhanced_features_list, dim=0)  # [B, N, D]
        
        # Reshape back if needed
        if needs_reshape:
            enhanced_features = enhanced_features.reshape(original_shape)
        
        if return_knowledge:
            return enhanced_features, retrieved_knowledge
        return enhanced_features
    
    def add_knowledge(self, text: str):
        """
        Add knowledge to the knowledge base.
        
        Args:
            text: Knowledge text
        """
        self.knowledge_base.add_knowledge(text)
    
    def retrieve_by_visual(self, visual_features: torch.Tensor, top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """
        Retrieve knowledge using visual features as query.
        
        Args:
            visual_features: Visual features [B, D]
            top_k: Number of items to retrieve
            
        Returns:
            List of retrieved knowledge per batch item
        """
        B = visual_features.shape[0]
        
        # Generate query from visual features
        query = self.query_projector(visual_features)  # [B, knowledge_dim]
        
        # Retrieve knowledge for each batch item
        all_results = []
        for b in range(B):
            results = self.knowledge_base.retrieve(query[b:b+1], top_k=top_k)
            all_results.append([(text, score) for text, score, _ in results])
        
        return all_results


class KnowledgeAwareDetection(nn.Module):
    """
    Knowledge-aware object detection.
    
    Uses retrieved knowledge to improve detection accuracy,
    especially for ambiguous or rare objects.
    """
    
    def __init__(
        self,
        visual_dim: int = 256,
        knowledge_dim: int = 512,
        num_classes: int = 80,
        use_mhc: bool = True
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.knowledge_dim = knowledge_dim
        self.num_classes = num_classes
        
        # RAG module
        self.rag = RAGVisionKnowledge(
            visual_dim=visual_dim,
            knowledge_dim=knowledge_dim,
            use_mhc=use_mhc
        )
        
        # Knowledge-enhanced classification head
        self.classification_head = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, num_classes)
        )
        
        # MHC for final fusion
        if use_mhc:
            self.final_mhc = ManifoldHyperConnection(
                input_dim=visual_dim,
                expansion_rate=2
            )
        else:
            self.final_mhc = nn.Identity()
    
    def forward(
        self,
        visual_features: torch.Tensor,
        detections: Dict[str, torch.Tensor],
        text_query: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Enhance detections with knowledge.
        
        Args:
            visual_features: Visual features [B, D, H, W]
            detections: Dictionary with detection results
            text_query: Optional text query
            
        Returns:
            Enhanced detections
        """
        B, D, H, W = visual_features.shape
        
        # Extract detection boxes
        if 'boxes' not in detections:
            return detections
        
        boxes = detections['boxes']  # [B, N, 4]
        scores = detections['scores']  # [B, N]
        labels = detections['labels']  # [B, N]
        
        # For each detection, extract features and enhance with knowledge
        enhanced_scores = []
        enhanced_labels = []
        
        for b in range(B):
            batch_boxes = boxes[b]
            batch_scores = scores[b]
            batch_labels = labels[b]
            
            if len(batch_boxes) == 0:
                enhanced_scores.append(batch_scores)
                enhanced_labels.append(batch_labels)
                continue
            
            # Extract features for each detection
            detection_features = []
            for box in batch_boxes:
                # Convert box to feature map coordinates
                x1, y1, x2, y2 = box * torch.tensor([W, H, W, H], device=box.device)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Clip coordinates
                x1 = max(0, min(x1, W-1))
                y1 = max(0, min(y1, H-1))
                x2 = max(0, min(x2, W-1))
                y2 = max(0, min(y2, H-1))
                
                if x2 > x1 and y2 > y1:
                    # Extract region features
                    region_feat = visual_features[b:b+1, :, y1:y2, x1:x2]
                    region_feat = F.adaptive_avg_pool2d(region_feat, (1, 1))
                    region_feat = region_feat.flatten(1)  # [1, D]
                else:
                    region_feat = torch.zeros(1, D, device=visual_features.device)
                
                detection_features.append(region_feat)
            
            if detection_features:
                detection_features = torch.cat(detection_features, dim=0)  # [N_det, D]
                
                # Enhance with knowledge
                enhanced_features = self.rag(
                    detection_features,
                    text_query=text_query
                )  # [N_det, D]
                
                # Combine original and enhanced features
                combined = torch.cat([detection_features, enhanced_features], dim=-1)
                
                # Knowledge-enhanced classification
                enhanced_logits = self.classification_head(combined)  # [N_det, num_classes]
                enhanced_probs = F.softmax(enhanced_logits, dim=-1)
                
                # Get enhanced scores and labels
                enhanced_score, enhanced_label = torch.max(enhanced_probs, dim=-1)
                
                # Combine with original detections
                final_scores = (batch_scores + enhanced_score) / 2
                final_labels = torch.where(
                    enhanced_score > batch_scores,
                    enhanced_label,
                    batch_labels
                )
                
                enhanced_scores.append(final_scores)
                enhanced_labels.append(final_labels)
            else:
                enhanced_scores.append(batch_scores)
                enhanced_labels.append(batch_labels)
        
        # Update detections
        enhanced_detections = detections.copy()
        enhanced_detections['scores'] = torch.stack(enhanced_scores, dim=0)
        enhanced_detections['labels'] = torch.stack(enhanced_labels, dim=0)
        
        return enhanced_detections