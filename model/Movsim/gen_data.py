from evaluations import *
from utils import *

def distance(gps1,gps2):
    x1,y1 = gps1
    x2,y2 = gps2
    return np.sqrt((x1-x2)**2+(y1-y2)**2 )

    
    

def gen_matrix(data='geolife'):
    train_data = read_data_from_file('/mnt/data/gonghaofeng/deeplearning_project/MoveSim-master/data/%s/real.data'%data)
    gps = get_gps('/mnt/data/gonghaofeng/deeplearning_project/MoveSim-master/data/%s/gps'%data)
    if data=='mobile':
        max_locs = 8606
    else:
        max_locs = 2499
        # max_locs = 14594

    reg1 = np.zeros([max_locs,max_locs])
    for i in range(len(train_data)):
        line = train_data[i]
        for j in range(len(line)-1):
            reg1[line[j],line[j+1]] +=1
    print(reg1)
    reg2 = np.zeros([max_locs,max_locs])
    print(max_locs)
    for i in range(max_locs):
        for j in range(max_locs):
            # try:
                if i!=j:
                    reg2[i,j] = distance((gps[0][i],gps[1][i]),(gps[0][j],gps[1][j]))
                    # print('第{0}{1}条处理失败'.format(i,j))
            # except:
                # print('第{0}{1}条处理失败'.format(i,j))

    np.save('/mnt/data/gonghaofeng/deeplearning_project/MoveSim-master/data/%s/M1.npy'%data,reg1)
    np.save('/mnt/data/gonghaofeng/deeplearning_project/MoveSim-master/data/%s/M2.npy'%data,reg2)

    print('Matrix Generation Finished')

    

    




    
