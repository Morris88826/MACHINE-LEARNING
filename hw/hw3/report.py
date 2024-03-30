import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def find_margin(w, b, points):

    margin = float('inf')
    for x in points:
        margin = min(margin, abs(w[0]*x[0] + w[1]*x[1] + b) / (w[0]**2 + w[1]**2)**0.5)
    return margin*2

def transform(points):

    x1 = []
    x2 = []
    x1x2 = []
    for x in points:
        x1.append(x[0])
        x2.append(x[1])
        x1x2.append(x[0]*x[1])
    return x1, x2, x1x2

if __name__ == "__main__":
    x1 = [3,2,4,1,2,4,4]
    x2 = [6,2,4,3,0,2,0]
    y = ['b','b','b','b','r','r','r']
    df = pd.DataFrame({'X1': x1, 'X2': x2, 'Y': y})

    os.makedirs('report', exist_ok=True)
    
    # question 1a
    print("=== Question 1a ===")
    model = SVC(kernel='linear')
    model.fit(list(zip(x1, x2)), y)
    w = model.coef_[0]
    b = model.intercept_[0]
    print("Equation: {} + {}x1 + {}x2 = 0".format(b, w[0], w[1])) 
    xrange = range(-3, 8)
    yrange = range(-3, 8)
    plt.scatter(df[df["Y"]=='b']["X1"], df[df["Y"]=='b']["X2"], c='b', marker='o')
    plt.scatter(df[df["Y"]=='r']["X1"], df[df["Y"]=='r']["X2"], c='r', marker='x')
    plt.plot([xrange[0], xrange[-1]], [-(b+w[0]*xrange[0])/w[1], -(b+w[0]*xrange[-1])/w[1]], color='black', label="-1+X1-X2=0")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(-3, 7)
    plt.ylim(-3, 7)
    plt.grid()
    plt.savefig('./report/1a.png')
    plt.legend()
    plt.clf()

    print("=== Question 1c ===")
    # find the margin
    margin = 2 / (w[0]**2 + w[1]**2)**0.5
    cal_margin = find_margin(w, b, list(zip(x1, x2)))
    assert abs(margin - cal_margin) < 1e-6
    print("Margin: {:.2f}".format(margin))
    plt.scatter(df[df["Y"]=='b']["X1"], df[df["Y"]=='b']["X2"], c='b', marker='o')
    plt.scatter(df[df["Y"]=='r']["X1"], df[df["Y"]=='r']["X2"], c='r', marker='x')
    plt.plot([xrange[0], xrange[-1]], [-(b+w[0]*xrange[0])/w[1], -(b+w[0]*xrange[-1])/w[1]], color='black', label="-1+X1-X2=0")
    plt.plot([xrange[0], xrange[-1]], [(1-(b+w[0]*xrange[0]))/w[1], (1-(b+w[0]*xrange[-1]))/w[1]], linestyle='--', c='r', label="-1+X1-X2=1 (positive)") # positive margin: w*x + b = 1
    plt.plot([xrange[0], xrange[-1]], [(-1-(b+w[0]*xrange[0]))/w[1], (-1-(b+w[0]*xrange[-1]))/w[1]], linestyle='--', c='b', label="-1+X1-X2=-1 (negative)") # negative margin: w*x + b = -1
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(-3, 7)
    plt.ylim(-3, 7)
    plt.grid()
    plt.legend()
    plt.savefig('./report/1c.png')
    plt.clf()

    print("=== Question 1f ===")
    plt.scatter(df[df["Y"]=='b']["X1"], df[df["Y"]=='b']["X2"], c='b', marker='o')
    plt.scatter(df[df["Y"]=='r']["X1"], df[df["Y"]=='r']["X2"], c='r', marker='x')
    plt.plot([xrange[0], xrange[-1]], [-(b+w[0]*xrange[0])/w[1], -(b+w[0]*xrange[-1])/w[1]], color='black', label="-1+X1-X2=0 (optimal, margin={:.2f})".format(margin))
    # plot a suboptimal hyperplane, -1+X1-1.5*X2=0
    suboptimal_w = [1, -1.2]
    suboptimal_b = -1
    suboptimal_margin = find_margin(suboptimal_w, suboptimal_b, list(zip(x1, x2)))
    print("Suboptimal Equation: {} + {}x1 + {}x2 = 0".format(suboptimal_b, suboptimal_w[0], suboptimal_w[1]))
    print("Suboptimal Margin: {:.2f}".format(suboptimal_margin))
    plt.plot([xrange[0], xrange[-1]], [-(suboptimal_b+suboptimal_w[0]*xrange[0])/suboptimal_w[1], -(suboptimal_b+suboptimal_w[0]*xrange[-1])/suboptimal_w[1]], color='g', label="-1+X1-1.2X2=0 (suboptimal, margin={:.2f})".format(suboptimal_margin))

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(-3, 7)
    plt.ylim(-3, 7)
    plt.grid()
    plt.legend()
    plt.savefig('./report/1f.png')
    plt.clf()

    print("=== Question 1g ===")
    # add a new point to make it linear non-separable
    x1.append(0)
    x2.append(6)
    y.append('r')

    df.loc[len(df)] = [x1[-1], x2[-1], y[-1]]

    model = SVC(kernel='linear')
    model.fit(list(zip(x1, x2)), y)
    w = model.coef_[0]
    b = model.intercept_[0]
    print("Equation: {} + {}x1 + {}x2 = 0".format(b, w[0], w[1]))
    plt.scatter(df[df["Y"]=='b']["X1"], df[df["Y"]=='b']["X2"], c='b', marker='o')
    plt.scatter(df[df["Y"]=='r']["X1"], df[df["Y"]=='r']["X2"], c='r', marker='x')
    plt.plot([xrange[0], xrange[-1]], [-(b+w[0]*xrange[0])/w[1], -(b+w[0]*xrange[-1])/w[1]], color='black', label="-1+X1-X2=0")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(-3, 7)
    plt.ylim(-3, 7)
    plt.grid()
    plt.legend()
    plt.savefig('./report/1g.png')
    plt.clf()

    print("=== Question 2b ===")
    x1 = [1,-1,1,-1]
    x2 = [1,-1,-1,1]
    y = [1,1,-1,-1]
    df = pd.DataFrame({'X1': x1, 'X2': x2, 'Y': y})

    model = SVC(kernel='linear')
    model.fit(list(zip(x1, x2)), y)
    w = model.coef_[0]
    b = model.intercept_[0]
    print("Equation: {} + {}x1 + {}x2 = 0".format(b, w[0], w[1]))
    xrange = range(-3, 4)
    yrange = range(-3, 4)
    plt.scatter(df[df["Y"]==1]["X1"], df[df["Y"]==1]["X2"], c='r', marker='x')
    plt.scatter(df[df["Y"]==-1]["X1"], df[df["Y"]==-1]["X2"], c='b', marker='o')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.savefig('./report/2b.png')
    plt.clf()

    print("=== Question 2c ===")
    x1, x2, x1x2 = transform(list(zip(x1, x2)))
    new_df = pd.DataFrame({'X1': x1, 'X2': x2, 'X1*X2': x1x2, 'Y': y})

    model = SVC(kernel='linear')
    model.fit(list(zip(x1, x2, x1x2)), y)
    w = model.coef_[0]
    b = model.intercept_[0]
    print("Equation: {} + {}x1 + {}x2 + {}x1x2 = 0".format(b, w[0], w[1], w[2]))
    # plot the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(new_df[new_df["Y"]==1]["X1"], new_df[new_df["Y"]==1]["X2"], new_df[new_df["Y"]==1]["X1*X2"], c='r', marker='x')
    ax.scatter(new_df[new_df["Y"]==-1]["X1"], new_df[new_df["Y"]==-1]["X2"], new_df[new_df["Y"]==-1]["X1*X2"], c='b', marker='o')
    x = range(-3, 4)
    y = range(-3, 4)
    X, Y = np.meshgrid(x, y)
    Z = (-w[0]*X - w[1]*Y - b) / w[2]
    ax.plot_surface(X, Y, Z, alpha=0.5, color='gray')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X1*X2')
    plt.savefig('./report/2c.png')
    plt.clf()

