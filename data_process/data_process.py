import zipfile
import polars as pl
import torch
import math
from collections import defaultdict
from typing import *
from bisect import bisect
from collections import Counter

# Load all Parquet files from a zip file into a dictionary of Polars DataFrames.
def load_parquets(zip_path: str) -> dict:
    parquets = {}

    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        # Iterate over all files in the zip
        for file_name in zip_file.namelist():
            # Skip system files or non-Parquet files
            if file_name.startswith('__') or not file_name.endswith('.parquet'):
                continue

            # Extract the file name without the .parquet extension
            file_key = file_name.rsplit('.parquet', 1)[0]

            # Read the Parquet file and add it to the dictionary
            with zip_file.open(file_name) as file:
                parquets[file_key] = pl.read_parquet(file)

    return parquets

# Merges multiple DataFrames on the 'article_id' column and combines their embeddings into a single column
def merge_article_embs(*dataframes: pl.DataFrame) -> pl.DataFrame:
    if not dataframes:
        raise ValueError("At least one DataFrame must be provided.")

    # Merge all DataFrames on 'article_id'
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = merged_df.join(df, on='article_id', how='inner')

    # Concatenate embeddings from all non-'article_id' columns
    embedding_columns = [col for col in merged_df.columns if col != 'article_id']
    merged_df = merged_df.with_columns(
        embeddings=pl.concat_list(embedding_columns)
    )

    # Select only 'article_id' and 'embeddings'
    return merged_df.select(['article_id', 'embeddings'])

# Merges article DataFrame with image embeddings, adds an indicator for image presence,
# and combines all features into a single 'embeddings' column.
def merge_article_with_imgs(text: pl.DataFrame, images: pl.DataFrame, col: str = 'image_embedding') -> pl.DataFrame:
    # Rename column if necessary
    if col != 'image_embedding':
        images = images.rename({col: 'image_embedding'})

    # Get the embedding size from the first non-null image embedding
    emb_size = len(images.filter(pl.col('image_embedding').is_not_null())['image_embedding'][0])

    # Merge text DataFrame with image embeddings on 'article_id'
    merged_df = text.join(images, on='article_id', how='outer')

    # Add 'has_image' column: 1 if image exists, 0 otherwise
    merged_df = merged_df.with_columns(
        has_image=pl.col('image_embedding').is_not_null().cast(pl.Int64)
    )

    # Fill missing image embeddings with zeros
    merged_df = merged_df.with_columns(
        pl.col('image_embedding').fill_null([0.0] * emb_size)
    )

    # Concatenate all embedding-related columns into a single 'embeddings' column
    embedding_columns = [col for col in merged_df.columns if col not in ('article_id', 'has_image')]
    merged_df = merged_df.with_columns(
        embeddings=pl.concat_list(embedding_columns)
    )

    # Select only the essential columns
    return merged_df.select(['article_id', 'embeddings', 'has_image'])

# Normalizes the embeddings to a [0, 1] range for each feature dimension.
def normalize(embeddings: torch.Tensor) -> torch.Tensor:
    min_c, max_c = torch.min(embeddings, dim=0, keepdim=True)[0], torch.max(embeddings, dim=0, keepdim=True)[0]
    denominator = max_c - min_c
    # Avoid division by zero by adding a small epsilon
    denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    return (embeddings - min_c) / denominator

# Standardizes the embeddings to have a mean of 0 and a standard deviation of 1.
def standarize(embeddings: torch.Tensor, mean: torch.Tensor = None, std: torch.Tensor = None) -> torch.Tensor:
    if mean is None or std is None:
        mean = torch.mean(embeddings, dim=0, keepdim=True)
        std = torch.std(embeddings, dim=0, keepdim=True)

    # Avoid division by zero by adding a small epsilon
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (embeddings - mean) / std



# Builds a dictionary encoding for a list of labels.
def build_dict_encoding(labels: List[Any], unknown: bool = False) -> Callable[[Any], int]:
    # Create a mapping from label to index
    label_to_index = {label: i for i, label in enumerate(labels)}

    if unknown:
        unknown_index = len(labels)
        label_to_index = defaultdict(lambda: unknown_index, label_to_index)

    return lambda x: label_to_index[x]

# Computes the quantile limits for a given list of values.
def quantile_limits(data: Iterable[int], quantiles: int = 100) -> List[float]:
    if not data:
        raise ValueError("Input data cannot be empty.")

    sorted_data = sorted(data)
    step = len(sorted_data) / quantiles
    indices = [math.ceil(i * step) - 1 for i in range(1, quantiles)]

    return [sorted_data[i] for i in indices]


# Counts the occurrences of individual elements within list-type values in a specified DataFrame column.
def count_in_list(df: pl.DataFrame, col: str) -> Dict[Any, int]:
    return Counter(v for x in df[col] if x is not None for v in x)

# Counts the occurrences of values in a specified column of a DataFrame.
def count(df: pl.DataFrame, col: str) -> Dict[Any, int]:
    return Counter(df[col])

# Generates a mapping function and a list of valid values for a given feature in a DataFrame.
def get_map_for_feature(
        df: pl.DataFrame,
        col: str,
        min_reps: int,
        is_list_column: bool = False,
        unknown: bool = False
) -> Tuple[Callable[[Any], int], List[Any]]:
    # Count occurrences of each unique value in the column
    value_counts = count_in_list(df, col) if is_list_column else count(df, col)

    # Filter values with sufficient occurrences
    filtered_values = sorted(x for x, count in value_counts.items() if count > min_reps)

    # Build a dictionary encoding function for the filtered values
    encoding_function = build_dict_encoding(filtered_values, unknown=unknown)

    return encoding_function, filtered_values

def binary_encoding(x: bool) -> int:
    return 1 if x else 0


def dict_encoding(x: Any, map: Dict[Any, int]) -> int:
    return map[x]

def time_encoding(x: int, limits: List[int]) -> Tuple[int]:
    return bisect(limits, x)

def ids_sort(data: List[Any]) -> List[int]:
    index = list(range(len(data)))
    index.sort(key=lambda x: data[x], reverse=False)
    return index


def sort_ids(info: List[Any], indexes: List[int]) -> List[Any]:
    return [info[idx] for idx in indexes]