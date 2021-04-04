import numpy as np
import matplotlib.pyplot as plt

def plot_result(input_, actual, predict):
    
    for i in range(input_.shape[0]):
        plt.imshow(input_[i])
        plt.title("Actual_" + str(i + 1))
        plt.show()
        
    for i in range(actual.shape[0]):
        plt.subplot(121), plt.imshow(actual[i]),
        plt.title("Actual_" + str(i + 1 + input_.shape[0]))
        plt.subplot(122), plt.imshow(predict[i]),
        plt.title("Predicted_" + str(i + 1 + input_.shape[0]))
        plt.show()

def reshape_patch(data, patch_sz):
    data_patch = []
    for sample in range(data.shape[0]):
        
        data_patch.append([])
        
        for frame in range(data.shape[1]):
            
            data_patch[sample].append([])
            rows = data.shape[2] // patch_sz[0]
            
            for row in range(rows):
                
                data_patch[sample][frame].append([])
                cols = data.shape[3] // patch_sz[1]
                
                for col in range(cols):
                    
                    patch = data[sample][frame][
                        row * patch_sz[0] : (row + 1) * patch_sz[0],
                        col * patch_sz[1] : (col + 1) * patch_sz[1]
                    ]
                    
                    # better to use list() compared to patch.tolist() here
                    data_patch[sample][frame][row].append(list(patch.reshape(patch_sz[0] * patch_sz[1])))
        
    return np.array(data_patch)

def restore_patch(data, patch_sz):
    data_restore = np.zeros((data.shape[0], data.shape[1] * patch_sz[0], data.shape[2] * patch_sz[1]))
    
    for frame in range(data.shape[0]):
        for row in range(data.shape[1]):
            for col in range(data.shape[2]):
                patch = data[frame][row][col].reshape(patch_sz)
                data_restore[frame][
                    row * patch_sz[0] : (row + 1) * patch_sz[0],
                    col * patch_sz[1] : (col + 1) * patch_sz[1]
                ] = patch
    
    return data_restore
