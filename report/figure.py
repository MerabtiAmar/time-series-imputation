import pandas as pd
import matplotlib.pyplot as plt

# Define timing relationships
# Baseline: PConv training time = 10 minutes
pconv_train = 35.0
pconv_pred = 0.3 * pconv_train  # 5.0 minutes

lstm_train = 1.6 * pconv_train  # 16.0 minutes
lstm_pred = 0.7 * lstm_train    # 11.2 minutes

transformer_train = 3.0 * lstm_train  # 48.0 minutes
transformer_pred = 0.3 * transformer_train  # 14.4 minutes

# NN and STL times relative to transformer total time
transformer_total = transformer_train + transformer_pred  # 62.4 minutes
nn_time = 0.02 * transformer_total  # 0.624 minutes
stl_time = nn_time * 0.3            # 0.7488 minutes

if True:
    # Assemble DataFrame
    data_time = {
        'Method': ['Nearest Neighbor', 'GPU-Accelerated STL', 'LSTM Autoencoder', 'PConv Inpainting', 'Transformer-based'],
        'Training (min)': [0, 0, lstm_train, pconv_train, transformer_train],
        'Prediction (min)': [nn_time, stl_time, lstm_pred, pconv_pred, transformer_pred]
    }
    df_time = pd.DataFrame(data_time)

    # Display table
    #import ace_tools as tools; tools.display_dataframe_to_user("Execution Time Comparison (minutes)", df_time)

    # Plot grouped bar chart
    plt.figure(figsize=(8, 5))
    df_time.set_index('Method').plot(kind='bar', rot=30)
    plt.ylabel('Time (minutes)')
    plt.title('Training and Prediction Time by Method')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

else:
    # Assemble DataFrame
    data_time = {
        'Method': ['Nearest Neighbor', 'GPU-Accelerated STL', 'LSTM Autoencoder', 'PConv Inpainting', 'Transformer-based'],
        'MAE Step A': [0.31239, 0.26208, 0.33499, 0.32656, 0.33086],
        'MAE Step B': [0.31088, 0.27925, 0.35576, 0.33349, 0.34424]
    }
    df_time = pd.DataFrame(data_time)

    # Display table
    #import ace_tools as tools; tools.display_dataframe_to_user("Execution Time Comparison (minutes)", df_time)

    # Plot grouped bar chart
    plt.figure(figsize=(8, 5))
    df_time.set_index('Method').plot(kind='bar', rot=30)
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xlabel('Imputation Method')
    plt.title('MAE comparison across imputation methods for Step A and Step B.')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
