import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArticleEmbedding(nn.Module):
    def __init__(self, article_embedding_size, category_embedding_size, premium, sentiments, temporal, weekdays, hours, dimensionality, dropout):
        super().__init__()

        # Unified dense layer to summarize article embeddings
        self.summarize = nn.Sequential(
            nn.Linear(article_embedding_size + category_embedding_size, dimensionality),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(dimensionality, dimensionality)
        )

        # Embedding layers for various categorical features
        self.embeddings = nn.ModuleDict({
            'premium': nn.Embedding(premium, dimensionality),
            'sentiment': nn.Embedding(sentiments, dimensionality),
            'temporal': nn.Embedding(temporal, dimensionality),
            'weekday': nn.Embedding(weekdays, dimensionality),
            'hour': nn.Embedding(hours, dimensionality)
        })

        self.dropout = dropout

    def embed_features(self, feature_name, feature_values):
        """Embeds and applies dropout to a specific feature"""
        embedding = self.embeddings[feature_name](feature_values)
        return F.dropout(embedding, self.dropout, self.training)

    def forward(self, article, temporal, weekdays, hours):
        embeddings, category_embeddings, premium, sentiment, mask = article

        expected_dim = self.summarize[0].in_features
        # Adjust category_embeddings to fit the expected size, if necessary
        category_embeddings = category_embeddings[..., :expected_dim - embeddings.shape[-1]]

        # Summarize the combined article and category embeddings
        x = self.summarize(torch.cat((embeddings, category_embeddings), dim=-1))

        # Add contributions from feature embeddings with dropout
        x += self.embed_features('premium', premium)
        x += self.embed_features('sentiment', sentiment)
        x += self.embed_features('temporal', temporal)
        x += self.embed_features('weekday', weekdays)
        x += self.embed_features('hour', hours)

        return x, mask


class EBRank(nn.Module):

    def __init__(self, device=4, sso=2, gender=4, postcode=6,
                 age=12, subscriber=2, weekday=7, hour=24, premium=2, sentiment=3,
                 temporal=100, dims=32, txt_dims=768, img_dims=128, nhead=4,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()

        # Embedding layers for user features
        self.user_embeddings = nn.ModuleDict({
            'device': nn.Embedding(device, dims),
            'sso': nn.Embedding(sso, dims),
            'gender': nn.Embedding(gender, dims),
            'postcode': nn.Embedding(postcode, dims),
            'age': nn.Embedding(age, dims),
            'subscriber': nn.Embedding(subscriber, dims)
        })

        # Article embedding
        self.article_embeddings = ArticleEmbedding(
            txt_dims + img_dims, txt_dims, premium, sentiment,
            temporal, weekday, hour, dims, dropout
        )

        # Transformer for modeling history and current interactions
        self.transformer = nn.Transformer(
            d_model=dims,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Final sort key to rank articles
        self.sort_key = nn.Sequential(
            nn.Linear(2 * dims, dims),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(dims, 1)
        )

        self.dropout = dropout

    def embed_user_features(self, behaviour):
        """Embeds and sums all user-level features from the behaviour tuple."""
        _, _, _, _, device_type, is_sso_user, gender, postcode, age, is_subscriber = behaviour

        # Use a loop to dynamically handle embeddings
        features = {
            'device': device_type,
            'sso': is_sso_user,
            'gender': gender,
            'postcode': postcode,
            'age': age,
            'subscriber': is_subscriber
        }

        b = sum(
            F.dropout(self.user_embeddings[feature](value), self.dropout, self.training)
            for feature, value in features.items()
        )
        return b

    def forward(self, behaviour, history):
        """Forward pass for EsktraSort."""

        # Embed user history and current articles
        history_emb, history_mask = self.article_embeddings(*history)
        articles, article_delta_time, impression_weekday, impression_hour, *_ = behaviour
        articles_emb, articles_mask = self.article_embeddings(
            articles, article_delta_time, impression_weekday, impression_hour
        )

        # Apply transformer to model interaction between history and current articles
        x = self.transformer(
            src=history_emb,
            tgt=articles_emb,
            src_key_padding_mask=history_mask,
            tgt_key_padding_mask=articles_mask,
            memory_key_padding_mask=history_mask
        )

        # Embed user-level features
        b = self.embed_user_features(behaviour)

        # Repeat the user embedding to match the dimensions of the article embeddings
        b = b.repeat(1, x.shape[1], 1)

        # Concatenate history with user-level embeddings
        x = torch.concat((x, b), dim=-1)

        # Predict sorting score for each article
        order = self.sort_key(x)
        return order.squeeze(-1)


def balance_bce_loss(predictions, in_view_items, clicked_items_indices):
    """
    Computes the balanced binary cross-entropy loss for a batch of predictions.

    Args:
    - predictions (list of tensors): Predictions for each user session.
    - in_view_items (list of int): Number of items in view for each session.
    - clicked_items_indices (list of lists): Indices of items clicked in each session.

    Returns:
    - loss (float): Average BCE loss across all sessions.
    - hit_rate (float): Percentage of sessions where the top prediction is in the clicked set.
    """
    total_loss = 0
    total_hits = 0
    num_sessions = len(predictions)

    for p, x, c in zip(predictions, in_view_items, clicked_items_indices):
        # Trim predictions to the number of items in view
        p = p[:x]
        c_set = set(c)

        # Get clicked and non-clicked article predictions
        clicked_predictions = p[list(c_set)] if c_set else torch.empty(0, device=p.device)
        non_clicked_predictions = p[~torch.tensor(range(p.shape[0])).isin(c_set)]

        # Check if the top predicted item was clicked
        if torch.argmax(p).item() in c_set:
            total_hits += 1

        # Calculate binary cross-entropy loss for clicked and non-clicked items
        if clicked_predictions.numel() > 0:
            total_loss += F.binary_cross_entropy_with_logits(clicked_predictions, torch.ones_like(clicked_predictions))

        if non_clicked_predictions.numel() > 0:
            total_loss += F.binary_cross_entropy_with_logits(non_clicked_predictions, torch.zeros_like(non_clicked_predictions))

    average_loss = total_loss / num_sessions
    hit_rate = total_hits / num_sessions

    return average_loss, hit_rate


def balance_bce_scroll_loss(predictions, in_view_items, clicked_items_indices, scroll_percentage):
    """
    Computes the scroll-weighted binary cross-entropy loss for a batch of predictions.

    Args:
    - pred (list of tensors): Predictions for each user session.
    - in_view_len (list of int): Number of items in view for each session.
    - clicked (list of lists): Indices of items clicked in each session.
    - scroll (list of floats): Scroll percentage for each session.

    Returns:
    - loss (float): Scroll-weighted BCE loss averaged over all sessions.
    - hit_rate (float): Percentage of sessions where the top prediction is in the clicked set.
    """
    total_loss = 0
    total_hits = 0
    total_scroll = 0
    num_sessions = len(predictions)

    for p, x, c, s in zip(predictions, in_view_items, clicked_items_indices, scroll_percentage):
        # Trim predictions to the number of items in view
        p = p[:x]
        scroll_weight = 0.5 + 0.5 * s  # Scale scroll to range [0.5, 1.0]
        total_scroll += scroll_weight

        c_set = set(c)

        # Get clicked and non-clicked article predictions
        clicked_predictions = p[list(c_set)] if c_set else torch.empty(0, device=p.device)
        non_clicked_predictions = p[~torch.tensor(range(p.shape[0])).isin(c_set)]

        # Check if the top predicted item was clicked
        if torch.argmax(p).item() in c_set:
            total_hits += 1

        # Calculate binary cross-entropy loss for clicked and non-clicked items
        if clicked_predictions.numel() > 0:
            total_loss += scroll_weight * F.binary_cross_entropy_with_logits(clicked_predictions,
                                                                             torch.ones_like(clicked_predictions))

        if non_clicked_predictions.numel() > 0:
            total_loss += scroll_weight * F.binary_cross_entropy_with_logits(non_clicked_predictions,
                                                                             torch.zeros_like(non_clicked_predictions))

    average_loss = total_loss / num_sessions
    hit_rate = total_hits / num_sessions

    return average_loss, hit_rate


def cce_scroll_loss(predictions, in_view_items, clicked_items_indices, scroll_percentage):
    """
    Computes the scroll-weighted cross-entropy loss for a batch of predictions.

    Args:
    - predictions (list of tensors): Predictions for each user session.
    - in_view_items (list of int): Number of items in view for each session.
    - clicked_items_indices (list of lists): Indices of items clicked in each session.
    - scroll_percentage (list of floats): Scroll percentage for each session.

    Returns:
    - loss (float): Scroll-weighted cross-entropy loss averaged over all sessions.
    - hit_rate (float): Percentage of sessions where the top prediction is in the clicked set.
    """
    total_loss = 0
    total_hits = 0
    total_scroll = 0
    num_sessions = len(predictions)

    for p, x, c, s in zip(predictions, in_view_items, clicked_items_indices, scroll_percentage):
        # Trim predictions to the number of items in view
        p = p[:x]
        scroll_weight = 0.5 + 0.5 * s  # Scale scroll to range [0.5, 1.0]
        total_scroll += scroll_weight

        # Convert clicked indices to a set to avoid duplicates
        c_set = set(c)

        # Check if the top predicted item was clicked
        if torch.argmax(p).item() in c_set:
            total_hits += 1

        # Create the target vector, marking clicked items as 1 and others as 0
        target = torch.zeros_like(p)
        target[list(c_set)] = 1

        # Compute the cross-entropy loss, weighted by the scroll factor
        total_loss += scroll_weight * F.cross_entropy(p, target)

    average_loss = total_loss / num_sessions
    hit_rate = total_hits / num_sessions

    return average_loss, hit_rate


def interpret_inference(indexes, predictions, in_view_lengths):
    """Interpret model inference by sorting predictions and assigning ranks."""
    results = [
        (index, np.argsort(-p[:v]).argsort() + 1)
        for index, p, v in zip(indexes, predictions, in_view_lengths)
    ]
    return results


def set_lr(optimizer, learning_rate):
    """Set learning rate for all parameter groups in the optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def to_device(values, device):
    """Recursively moves all torch tensors in the input to the specified device."""
    if isinstance(values, (tuple, list)):
        return type(values)(to_device(v, device) for v in values)
    elif isinstance(values, torch.Tensor):
        return values.to(device)
    return values


