# issues with matplotlib can often be resolved by specifying the backend
# see init_mpl() function
import matplotlib
from matplotlib import rcParams, cm
rcParams['font.family'] = 'serif'
from matplotlib import pyplot as plt

import util
CUPY, cp = util.import_cp_or_np(try_cupy=0) #should import numpy as cp if cupy not installed


#TODO: pick better colors
COLORS = ['#009999','#9933ff','#cc0066','#009933','#0000ff','#99cc00','#ff9933']

def pie(params, attractors, node_mapping, num_nodes):
	node_name_to_num = node_mapping['node_name_to_num']
	node_num_to_name = node_mapping['node_num_to_name']
	init_mpl(params)

	labels = list(attractors.keys()) #just to make sure order is set

	if CUPY:
		sizes = [attractors[labels[i]]['size'].get() for i in range(len(labels))]
	else:
		sizes = [attractors[labels[i]]['size'] for i in range(len(labels))]

	
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

	if params['use_phenos']:
		color_choice = []
		legend_title = "Phenotypes ("
		j=0
		for i in range(num_nodes):
			if node_num_to_name[i] in params['phenos']['outputs']: #lazy lol
				if j>0:
					legend_title +=','
				legend_title += node_num_to_name[i]
				j+=1
		legend_title += ")"

		# redo labels, but only for the phenos

		outputs = [node_name_to_num[params['phenos']['outputs'][i]] for i in range(len(params['phenos']['outputs']))]

		pheno_labels, colors_num, c = [],[],0
		for lab in labels:
			if lab=='oscillates':
				pheno_labels+=[lab]
				if c !=0:
					c += 1
				colors_num += [c]
			else:
				pheno_lab = ''.join([lab[outputs[i]] for i in range(len(outputs))])
				if pheno_lab not in pheno_labels:
					pheno_labels += [pheno_lab]
					c+=1 #only increment color up if new label
				colors_num+=[c]
		labels = pheno_labels
		#colors = [COLORS[colors_num[i]] for i in range(len(colors_num))]
		colors = cm.Set2(colors_num)

	else:
		colors=cm.Set2([i for i in range(len(labels))])
		legend_title = "Attractors ("
		for i in range(1,num_nodes): #ie skip the 0th node, which is always OFF
			if i!=1:
				legend_title +=','
			legend_title += node_num_to_name[i]
		legend_title += ")"

		colors=cm.Set2([i for i in range(len(labels))])
	



	wedges, texts, autotexts = ax.pie(sizes,colors=colors, counterclock=False, autopct='%1.1f%%', shadow=False, startangle=90)
	ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

	if params['use_phenos']:
		num_divs = 5
		alphas = [i/num_divs for i in range(1,num_divs+1)]
		alphas.reverse()
		alpha_ind, color_ind = 0,0
		for i in range(len(wedges)):
			if colors_num[i] == color_ind: 
				alpha_ind =0
			else:
				alpha_ind+=1
			wedges[i].set_alpha(alphas[alpha_ind %len(alphas)])		
			color_ind+=1


	lgd = ax.legend(wedges, labels, fontsize=12)#,title="Attractors", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
	lgd.set_title(legend_title,prop={'size':16})
	#plt.setp(autotexts, size=8, weight="bold")

	ax.set_title("Basin Sizes",size=20)
	if params['savefig']:
		plt.savefig("output/" + params['output_img']) 	
	else:
		plt.show()
		

def init_mpl(params):
	# in the past this helps to auto pick matplotlib backend for different computers

	try:
		if 'mpl_backend' in params.keys():
			if params['mpl_backend'] == 'Agg':
				matplotlib.use('Agg')
			elif params['mpl_backend'] == 'TkAgg':
				matplotlib.use('TkAgg')	
			elif params['mpl_backend'] == 'WX':
				matplotlib.use('WX')	
			elif params['mpl_backend'] == 'QTAgg':
				matplotlib.use('QTAgg')	
			elif params['mpl_backend'] == 'QT4Agg':
				matplotlib.use('QT4Agg')
		elif "savefig" in params.keys():
			if params['savefig']:
				matplotlib.use('Agg')
			else:
				matplotlib.use('TkAgg')

	except ModuleNotFoundError:
		# fall back to MacOSX lib
		matplotlib.use('MACOSX')