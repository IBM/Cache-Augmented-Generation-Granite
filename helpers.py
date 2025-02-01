import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_env() -> dict:
    # Get the environment variables from the .env file
    env_dict = {}
    with open(file=".env", mode="r") as f:
        for line in f:
            key, value = line.strip().split("=")
            env_dict[key] = value.strip('"')
    return env_dict


def generate_graphs(qa_details_per_model, model_summary_stats):
    # Convert model_summary_stats into a DataFrame for visualization
    summary_df = pd.DataFrame.from_dict(model_summary_stats, orient="index")
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={"index": "model"}, inplace=True)

    # Plotting the average similarity, inference time, and KV cache preparation time for each model
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Average similarity plot
    summary_df.plot(
        x="model",
        y="avg_similarity",
        kind="bar",
        ax=axes[0],
        color="skyblue",
        title="Average Similarity"
    )
    axes[0].set_ylabel("Similarity")
    axes[0].set_xlabel("Model")

    # Average inference time plot
    summary_df.plot(
        x="model",
        y="avg_inference_time",
        kind="bar",
        ax=axes[1],
        color="salmon",
        title="Average Inference Time (s)"
    )
    axes[1].set_ylabel("Time (s)")
    axes[1].set_xlabel("Model")

    # KV Cache preparation time plot
    summary_df.plot(
        x="model",
        y="prepare_time",
        kind="bar",
        ax=axes[2],
        color="lightgreen",
        title="KV Cache Preparation Time (s)"
    )
    axes[2].set_ylabel("Time (s)")
    axes[2].set_xlabel("Model")

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig("model_performance_with_kv_cache.png")
    print("Graphs saved as 'model_performance_with_kv_cache.png'")

