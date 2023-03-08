import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary_clustering(model, data_0_test, data_1_test, 
                                     target_test, centroids_0, centroids_1):
    
    h = .02  # step size in the mesh
    x_min, x_max = np.array(data_0_test).min() - 1, np.array(data_0_test).max() + 1
    y_min, y_max = np.array(data_1_test).min() - 1, np.array(data_1_test).max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    scatter1 = ax.scatter(np.array(data_0_test), np.array(data_1_test), 
                c=np.array(target_test), cmap=plt.cm.coolwarm, edgecolor="k")
    scatter2 = ax.scatter(centroids_0, centroids_1, marker="x", s=169, 
                linewidths=3, color="w", edgecolor="k", zorder=10)
    # plt.xlabel(data_0_test.name)
    # plt.ylabel(data_1_test.name)
    plt.xlabel('Average temperature in fall-winter')
    plt.ylabel('Electricity demand')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    legend1 = ax.legend(*scatter1.legend_elements(),
                        loc="lower right", title="Consumers categories")
    ax.add_artist(legend1)
    plt.title('Predicted classes (including centroids) - normalized data')
    plt.show()
