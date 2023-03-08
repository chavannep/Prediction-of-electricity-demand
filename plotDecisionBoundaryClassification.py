import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary_classification(model, data_0_test, data_1_test, target_test): 
     
    h = 0.02
    xx, yy = np.meshgrid(np.arange(data_0_test.min()-1, data_0_test.max()+1, h),
                          np.arange(data_1_test.min()-1, data_1_test.max()+1, h)
                          )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    
    ax.contourf(xx[:,:], yy[:,:], Z[:,:], cmap=plt.cm.coolwarm, alpha=0.8)
    scatter1 = ax.scatter(np.array(data_0_test), np.array(data_1_test), 
                c=np.array(target_test), cmap=plt.cm.coolwarm, edgecolor="k")
    # plt.xlabel(data_0_test.name)
    # plt.ylabel(data_1_test.name)
    plt.xlabel('Average temperature in fall-winter')
    plt.ylabel('Average temperature in spring-summer')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    legend1 = ax.legend(*scatter1.legend_elements(),
                        loc="lower right", title="Consumers categories")
    ax.add_artist(legend1)
    plt.title('Classification : predicted labels')
    plt.show()


