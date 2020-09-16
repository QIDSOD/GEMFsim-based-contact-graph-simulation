from GEMFPy import *
from time import time



G=nx.random_geometric_graph(100,0.151)
pos=nx.get_node_attributes(G,'pos')

# find node near center (0.5,0.5)
dmin=1
ncenter=0
for n in pos:
    x,y=pos[n]
    d=(x-0.5)**2+(y-0.5)**2
    if d<dmin:
        ncenter=n
        dmin=d

# color by path length from node near center
p=nx.single_source_shortest_path_length(G,ncenter)

plt.figure(figsize=(8,8))
nx.draw_networkx(G, pos, node_size =200)

# plt.savefig('random_geometric_graph.png')
plt.show()

# G = nx.erdos_renyi_graph(10,.3)
N = G.number_of_nodes()

beta = 1.2
delta = 2
Lambda = 1
Para = Para_SEIR(delta, beta, Lambda)

x0 = np.zeros(N)
x0 = Initial_Cond_Gen(N, Para[1][0], 2, x0)

Net = NetCmbn([MyNet(G)])
StopCond = ['RunTime', 10]

t, f = MonteCarlo(Net, Para,  StopCond, 1, 4, .1, 20, N, x_init = np.zeros(N) )

fig2 = plt.figure(figsize=(10,5))
# for i in range(M):
# plt.plot(T, StateCount[0,:]/N,'r',label='Susceptible')
plt.plot(t,f[0,:],'r',label='Susceptible')
plt.plot(t,f[1,:],'b',label='Exposed')
plt.plot(t,f[2,:],'g',label='Infected')
plt.plot(t,f[3,:],'y',label='Recovered')
# plt.savefig("SEIR.png")

plt.xlabel('Time (day)')
plt.ylabel('Fraction of Population')
plt.title('SEIR')
plt.legend(loc='upper center', shadow=True)
plt.show()
# plt.savefig("SEIR_MonteCarlo.png")

# *************

ts, n_index, i_index, j_index = GEMF_SIM(Para, Net, x0, StopCond,N)

M = Para[0]
T, StateCount = Post_Population(x0, M, N, ts, i_index, j_index)

fig = plt.figure(figsize=(10,5))
# for i in range(M):
# plt.plot(T, StateCount[0,:]/N,'r',label='Susceptible')
plt.plot(T, StateCount[0,:]/N,'r',label='Susceptible')
plt.plot(T, StateCount[1,:]/N,'k',label='Exposed')
plt.plot(T, StateCount[2,:]/N,'y',label='Infected')
plt.plot(T, StateCount[3,:]/N,'g',label='Recovered')

plt.xlabel('Time (day)')
plt.ylabel('Fraction of Population')
plt.title('SEIR')
plt.legend(loc='upper center', shadow=True)
plt.show()
# plt.savefig("SEIR.png")

# *************

fig = plt.figure(figsize = (10, 10))

comp = ['S', 'E', 'I', 'R']
colors = ['olivedrab', 'blue', 'tomato', 'gray']
col = dict(zip(comp, colors))

model = [x0, n_index, i_index, j_index]
anim = animate_discrete_property_over_graph(G, model, len(ts)-1, fig, n_index,i_index, j_index, comp, 'state',
                                            col, pos = pos, Node_radius = .01)
plt.show()
##ALERT-----before saving: Just be careul when you save the movie, cause it takes too much time, i.e., each event is a frame.
# anim.save('myTest11111.mp4')

# *************

