import matplotlib.pyplot as plt 
from matplotlib.patches import FancyBboxPatch

plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['mathtext.fontset'] = 'stix'

rad = 0.3
grey = '#6A6A6A'

def cell():
   ax.add_patch(FancyBboxPatch((1.4,1.4), 10, 3,  alpha=1, color='#F0F0F5',boxstyle="round"))

def arrow(x,y,w,h,has_head=True):
   if has_head:
      ax.arrow(x,y,w,h,color=grey,length_includes_head=True,  head_width=0.15, head_length=0.15)
   else:
      ax.arrow(x,y,w,h,color=grey)

def fancy_arrow(x1,y1,x,y,style):
   ax.annotate("", xy=(x1, y1),  xytext=(x, y), arrowprops=dict(arrowstyle="-|>",mutation_scale=10,linewidth=1.5,color=grey, connectionstyle=style))

def mul_unit(x,y):

   c1 = plt.Circle((x,y),rad,facecolor='#C6DEA2',edgecolor=grey)
   c2 = plt.Circle((x,y),0.05,color=grey)
   ax.add_artist(c1)
   ax.add_artist(c2)

def add_unit(x,y):
   c1 = plt.Circle((x,y),rad,facecolor='#FFB380',edgecolor=grey)
   ax.add_artist(c1)
   ax.plot(x,y,marker='+', markersize=8,mew=1.5,c=grey)
   
def sub_unit(x,y):
   c1 = plt.Circle((x,y),rad,facecolor='#E5C6F6',edgecolor=grey)
   ax.add_artist(c1)
   ax.plot(x-0.05,y,marker='$1$', markersize=8,mew=0.5,c=grey)
   ax.plot(x+0.1,y,marker='$-$', markersize=5,mew=0.5,c=grey)

fig, ax = plt.subplots()
cell()

a = 2
b = 3
inter = 1
steps = [2.5, 4]

for i in range(3):

   if i > 0:
      a = a + steps[i-1]
      mul_unit(a + inter , b + inter)
      arrow(a+inter, b + rad + 0.05, 0, inter - 2 *rad - 0.1)

   sub_unit(a,b)
   mul_unit(a+inter,b-inter)
   add_unit(a+inter,b)
 
   arrow(a,b -inter,inter -rad - 0.05,0)
   arrow(a + rad + 0.05,b,inter - 2 *rad - 0.1,0)
   arrow(a + inter, b-inter + rad + 0.05,0,inter - 2 *rad - 0.1)

   arrow(a + inter, 1,0,0.7,False)
   arrow(a ,1, 0, 1.7,False)

   n = str(i +1) if i < 2  else 'n'
   plt.text(a-0.2,0.7,'$\mathit{W}_{1,%s}$'%n,fontsize=18)
   plt.text(a+inter-0.2,0.7,'$\mathit{x}_%s$'%n,fontsize=18)


a = 2 + inter
b = 3 + rad
a1 = a + steps[0] - rad
b1 = b + rad + 0.4

fancy_arrow(a1, b1,a, b,"angle,angleA=-90,rad=15")

a = a1 + 2 * rad + 0.05

arrow(a, b1,0.6,0)
plt.text(a + 0.7,b1 - 0.06,'$\cdots$',fontsize=15,color='grey')
arrow(a+1.1, b1,2.2,0)


a = 2 + steps[0] + steps[1] +  inter
fancy_arrow(a + 1.5, b1 + 1,a+rad + 0.05, b1,"angle,angleA=180,angleB=90,rad=30")

plt.text(a+1.4,b1+1, '$\mathit{z}_1$',fontsize=18)
plt.text(6.8,0.7,'$\cdots$',fontsize=15,color='grey')

ax.axis('off')

ax.set_xlim([0,12])
ax.set_ylim([0,7])

plt.text(3.5,5.5,'Neural Multiplication Unit',fontsize=22)
#plt.savefig('nmu.png')
plt.show()
