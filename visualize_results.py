import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def visualize_results(log_file, output_image):
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    df = pd.DataFrame(data)
    
    # Fill missing values and handle non-numeric columns
    for col in ['score', 'reward', 'exploitation_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot Score
    axes[0].plot(df.index, df['score'], marker='o', linestyle='-', color='b')
    axes[0].set_title('Score over Iterations')
    axes[0].set_ylabel('Score')
    axes[0].grid(True)
    
    # Plot Reward
    axes[1].plot(df.index, df['reward'], marker='s', linestyle='-', color='g')
    axes[1].set_title('Reward over Iterations')
    axes[1].set_ylabel('Reward')
    axes[1].grid(True)
    
    # Plot Exploitation Score
    if 'exploitation_score' in df.columns:
        axes[2].plot(df.index, df['exploitation_score'], marker='^', linestyle='-', color='r')
        axes[2].set_title('Exploitation Score over Iterations')
        axes[2].set_ylabel('Exploitation Score')
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Visualization saved to {output_image}")

if __name__ == "__main__":
    log_path = 'results/training_log.jsonl'
    output_path = 'results/visualization.png'
    if os.path.exists(log_path):
        visualize_results(log_path, output_path)
    else:
        print(f"Log file {log_path} not found.")
