import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

folder_path = 'vine_mae'


def extract_data(folder_path):
    data_query = {}
    data_non_query = {}

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.json'):
            is_query = '_query' in file_name
            try:
                step = int(file_name.split('_')[0])
            except:
                step = int(file_name.split('.')[0])
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            ground_truth = [item[0] for item in data['data']]
            predictions = [item[1] for item in data['data']]
            
            mae = mean_absolute_error(ground_truth, predictions)
            
            if is_query:
                data_query[step] = mae
            else:
                data_non_query[step] = mae

    steps_query, mae_query = zip(*sorted(data_query.items()))
    steps_non_query, mae_non_query = zip(*sorted(data_non_query.items()))
    return steps_query, mae_query, steps_non_query, mae_non_query

vsteps_query, vmae_query, vsteps_non_query, vmae_non_query = extract_data('vine_mae')
steps_query, mae_query, steps_non_query, mae_non_query = extract_data('prob_mae')
dsteps_query, dmae_query, dsteps_non_query, dmae_non_query = extract_data('dpo_mae')


plt.figure(figsize=(10, 6))
plt.plot(vsteps_query[:-1], vmae_query[:-1], marker='o', label='VinePPO-Query')
plt.plot(vsteps_non_query[:-1], vmae_non_query[:-1], marker='s', label='VinePPO-Non-Query')
plt.plot(steps_query, mae_query, marker='o', label='PorbPPO-Query')
plt.plot(steps_non_query, mae_non_query, marker='s', label='PorbPPO-Non-Query')
plt.plot(dsteps_query, dmae_query, marker='o', label='DPO-Query')
plt.plot(dsteps_non_query, dmae_non_query, marker='s', label='DPO-Non-Query')
plt.xlabel('Training Steps')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Training Steps vs MAE')
plt.legend()
plt.grid(True)
plt.savefig('figure/mae.jpg')
plt.show()
