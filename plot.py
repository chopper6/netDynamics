# issues with matplotlib can often be resolved by specifying the backend
from matplotlib import rcParams, cm
rcParams['font.family'] = 'serif'
from matplotlib import pyplot as plt

def pie(params, steady_states,node_num_to_name,num_nodes):
	# slices plotted counter-clockwise
	labels = list(steady_states.keys()) #just to make sure order is set
	sizes = [steady_states[labels[i]] for i in range(len(labels))]
	
	# could prob do this chunk more succinctly
	del_zeros, offset = [], 0
	for i in range(len(sizes)):
		if sizes[i] == 0:
			del_zeros += [i]
	for i in del_zeros:
		del sizes[i-offset]
		del labels[i-offset]
		offset += 1

	fig, ax = plt.subplots(figsize=(10, 6))
	cs=cm.Set2([i for i in range(len(labels))])
	wedges, texts, autotexts = ax.pie(sizes,colors=cs, autopct='%1.1f%%', shadow=True, startangle=90)
	ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

	legend_title = "Attractors ("
	for i in range(num_nodes):
		if i!=0:
			legend_title +=','
		legend_title += node_num_to_name[i]
	legend_title += ")"
	lgd = ax.legend(wedges, labels, fontsize=12)#,title="Attractors", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
	lgd.set_title(legend_title,prop={'size':16})
	#plt.setp(autotexts, size=8, weight="bold")

	ax.set_title("Basin Sizes",size=20)
	if params['savefig']:
		plt.savefig("output/" + params['output_img']) 	
	else:
		plt.show()
		
		