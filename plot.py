# issues with matplotlib can often be resolved by specifying the backend
# see init_mpl() function
import matplotlib
from matplotlib import rcParams, cm
rcParams['font.family'] = 'serif'
from matplotlib import pyplot as plt
from matplotlib import colors as mplc
import random as rd
import numpy as np
import util
CUPY, cp = util.import_cp_or_np(try_cupy=0) #should import numpy as cp if cupy not installed


#TODO: pick better colors
#cm_list = [40*(i+2) for i in range(8)]
#rd.shuffle(cm_list)
top = cm.get_cmap('Set2',8)#([i for i in range(8)])
bottom= cm.get_cmap('Dark2',8)#([i for iin range(8)]) 
COLORS= cp.vstack((top(cp.linspace(0, 1, 8)),bottom(cp.linspace(0, 1, 8))))

COLORS2 = ['#9933ff','#009999','#cc0066','#009933','#0000ff','#99cc00','#ff9933']
COLORS3 = ['#ff3300','#cc0066','#cc00ff','#6600ff','#0000ff','#0099cc','#00cc99','#00cc00','#888844','#cc9900']

COLORS = ['#ffff00','#ff9933','#ff5050','#ff33cc','#cc66ff','#3366ff','#00ffcc','#00ff00'] + COLORS2 + COLORS3

def pie(params, steadyStates, G, external_label=None):
	node_name_to_num = G.nodeNames
	node_num_to_name = G.nodeNums
	num_nodes = G.n
	init_mpl(params)

	if util.istrue(params,'use_phenos'):
		basin_sizes = steadyStates.phenotypes
	else:
		basin_sizes = steadyStates.attractors

	labels = sorted(list(basin_sizes.keys())) #just to make sure order is set

	sizes = [basin_sizes[labels[i]].size for i in range(len(labels))]

	# could prob do this chunk more succinctly
	del_zeros, offset = [], 0
	for i in range(len(sizes)):
		if sizes[i] == 0:
			del_zeros += [i]
	for i in del_zeros:
		del sizes[i-offset]
		del labels[i-offset]
		offset += 1
	
	if util.istrue(params,'use_phenos'):
		legend_title = ""
		j=0
		if params['use_inputs_in_pheno']:
			for i in range(len(params['inputs'])):
				if j>0:
					legend_title +=', '
				legend_title += params['inputs'][i]
				j+=1	
			legend_title += ' | '
		j=0
		for i in range(len(params['outputs'])):
			if j>0:
				legend_title +=', '
			legend_title += params['outputs'][i]
			j+=1
	else:
		legend_title = "Attractors ("
		for i in range(1,num_nodes): #ie skip the 0th node, which is always OFF
			if i!=1:
				legend_title +=','
			legend_title += G.nodeNames[i]
		legend_title += ")"


	fig, ax = plt.subplots(figsize=(10, 6))

	if util.istrue(params,'use_phenos') and 'pheno_color_map' in params.keys():
		label_map = []
		l=len(params['pheno_color_map']) 
		for label in labels:
			if '|' in label: # only use the outputs for the label!
				parts = label.split("|") 
				label = parts[1] 
			if label in params['pheno_color_map']:
				label_map+= [params['pheno_color_map'][label]]
			else:
				label_map+=[l]
				l+=1
		colors=cm.Set2(label_map)	

	else:
		colors=cm.Set2([i for i in range(len(labels))])	

	wedges, texts, autotexts = ax.pie(sizes,colors=colors, counterclock=False, autopct='%1.1f%%', shadow=False, startangle=90)
	ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

	''' subdivide slices, kept for reference if want to use later
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
	'''

	if 'fig_legend' not in params.keys() or params['fig_legend']:
		lgd = ax.legend(wedges, labels, fontsize=12)#,title="Attractors", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
		lgd.set_title(legend_title,prop={'size':12})
		#plt.setp(autotexts, size=8, weight="bold")

	if util.istrue(params,'use_phenos'):
		name='Phenotypes'
	else:
		name='Attractors'
	ax.set_title("Basin Sizes of " + name,size=20)
	if params['savefig']:
		if external_label is None:
			plt.savefig(params['output_dir'] +'/'+params['output_img']) 	
		else:
			plt.text(0, -1.2, external_label, verticalalignment='bottom', horizontalalignment='center')
			plt.savefig(params['output_dir'] +'/'+ external_label+params['output_img'])
	else:
		plt.show()



def features(params,seq,feats):
	init_mpl(params)
	labels = feats[0].keys()
	j=0
	for label in labels:
		fig, ax = plt.subplots(figsize=(10, 6))
		plt.plot([feats[i][label] for i in range(len(feats))])

		#lgd = ax.legend(labels, fontsize=12)#,title="Attractors", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
		plt.xticks([i for i in range(len(seq))],[seq[i][0]+'='+str(seq[i][1]) for i in range(len(seq))],rotation=40)
		ax.set_ylabel('Entropy', fontsize=12)
		ax.set_title(label + " of Mutation Sequence",size=16)
		if params['savefig']:
			plt.savefig(params['output_dir'] +'/' + "".join(x for x in label if x.isalnum()) + '.jpg') 	
		else:
			plt.show()
		j+=1


def clustered_time_series(params,avgs,label,CIs=None,ybounds=None):
	# avg should be 1d array
	# CIs[:,1] should be 1d array of avg-CI
	# CIs[:,2] should be avg+CI
	init_mpl(params)
	fig, ax = plt.subplots(figsize=(10, 6))

	for i in range(len(avgs)):
		plt.plot(avgs[i])
		if CIs is not None and CIs[i] is not None:
			plt.fill_between([i for i in range(len(avgs[i]))],CIs[i][1],CIs[i][0],alpha=.15,label='_nolegend_')

	if ybounds is not None:
		ax.set_ylim(ybounds)

	plt.grid(alpha=.2)

	ax.set_xticks([i*2 for i in range(int((len(avgs[0])+1)/2))])
	ax.set_xlabel('Mutation Number')
	ax.set_ylabel('Entropy', fontsize=12)
	ax.set_title(label + " across Mutation Sequence",size=16)
	if params['savefig']:
		plt.savefig(params['output_dir'] +'/' + "".join(x for x in label if x.isalnum()) + '.jpg') 	
	else:
		plt.show()





def probably_bar(params, feats):
	# feats is 'noise':{'gate':{'loopType':'stat'[...]}}
	# TODO normalize y to max y height

	buffer, width = 1,.8
	fontsmall, fontmed, fontlarge = 16, 16, 26
	noises = list(feats.keys())
	xlabels = list(feats[noises[0]].keys()) # to take all
	#  ['P','PP_and','PP_or','PN_and','PN_or','NN_and','NN_or','N', 'PP_and_long','PN_and_long','NN_and_long', 'PP_xor','PN_xor','NN_xor','PP_xor_long','PN_xor_long','NN_xor_long']
	#xlabels = ['PP_and','PP_xor','PN_and','PN_xor','NN_and','NN_xor']
	#xlabels = ['P','PP_and','PP_xor','PP_and_long','PN_and','NN_and','N']#,''] #,'PN_and','PN_and_long','NN_and','NN_and_long']#,'']
	#stats = list(feats[noises[0]][xlabels[0]].keys())
	stats = ['fast variance','slow variance'] #'average',

	fig, axs = plt.subplots(len(noises), len(stats),figsize=(24,8)) # noise(vert) x stats(horz)
	fig.subplots_adjust(wspace=.15, hspace=1, bottom=0.2)
	ymax=0
	#for h in range(len(stats)):
	#	stat = stats[h]
	#	ymax = max(ymax, getmax(feats,stat,noises,xlabels)) # one ylim for all plots
	for h in range(len(stats)):
		stat = stats[h]
		for v in range(len(noises)):
			noise = noises[v]
			c,x=0,0
			xticks=[]
			for group in xlabels:
				if group!='':
					data = feats[noise][group][stat]
					X = [i for i in range(x,x+len(data))]
					xticks += [x+(len(data)-1)/2]
					x+=len(data) + buffer
					axs[v,h].bar(X, data, width,color=COLORS[c],edgecolor='black',linewidth=1)
					c+=1

			#xticks +=[12]
			axs[v,h].grid(alpha=.5,zorder=0,color='grey')
			axs[v,h].set_axisbelow(True)
			axs[v,h].set_xticks(xticks)
			axs[v,h].set_xticklabels(xlabels,fontsize=fontsmall,rotation=45)
			axs[v,h].tick_params(axis='y', which='major', labelsize=fontsmall)
			#axs[v,h].set_ylim(0,ymax*1.1)
			axs[v,h].set_ylim(0,.25)
			axs[v,h].set_title(noise + ' ' + stat,fontsize=fontlarge)
		#fig.suptitle(stat,fontsize=20)

	#plt.tight_layout() # this will override the subplots_adjust argument above though
	if params['savefig']:
		plt.savefig(params['output_dir'] +params['output_img']) 	
	else:
		plt.show()

def getmax(feats,stat,noises,xlabels):
	ymax=0
	for v in range(len(noises)):
		noise = noises[v]
		for group in xlabels:
			ymax = max(ymax,max(feats[noise][group][stat]))
	return ymax


def control_exper_bar(params, stats):
	# 1 bar plot with 1 cluster per stat
	settings = list(stats[1].keys())
	stat_keys1 = list(stats[1][settings[0]].keys())
	stat_keys2 = list(stats[2][settings[0]].keys())
	width = 1/(2*len(settings))
	label_locs = cp.arange(len(stat_keys1)) 
	fig, ax = plt.subplots(2,figsize=(10, 12))
	i=0
	for k in settings:
		data = [stats[1][k][k2] for k2 in stat_keys1]

		locs = label_locs+(i - (len(settings)-1)/2)*width
		alabel = ax[0].bar(locs, data, width, label=k,color=COLORS[i])
		i+=1
	ax[0].set_title('Mutability and Reversibility Metrics',fontsize=16)
	ax[0].set_xticks(label_locs)
	ax[0].set_xticklabels(stat_keys1,fontsize=14, rotation = 15, ha="center")
	ax[0].legend(prop={'size': 10})

	label_locs = cp.arange(len(stat_keys2)) 
	i=0
	for k in settings:
		data = [stats[2][k][k2] for k2 in stat_keys2]
		locs = label_locs+(i - (len(settings)-1)/2)*width
		alabel = ax[1].bar(locs, data, width, label=k,color=COLORS[i])
		i+=1
	ax[1].set_title('(Topo)logical Network Features',fontsize=16)
	ax[1].set_xticks(label_locs)
	ax[1].set_xticklabels(stat_keys2,fontsize=14, rotation = 15, ha="center")
	ax[1].legend(prop={'size': 10})
	fig.tight_layout()

	if params['savefig']:
		plt.savefig(params['output_dir'] +'/barz_' +params['output_img']) 	
	else:
		plt.show()
	plt.clf()
	plt.close()


def control_exper_scatter(params, node_stats, labels):

	for k in labels.keys():
		#fig, ax = plt.subplots(figsize=(10, 6))
		n = len(node_stats[k])
		x,y = [node_stats[k][i][0] for i in range(n)],[node_stats[k][i][1] for i in range(n)]
		plt.scatter(x,y,alpha=.05)
		plt.xlabel(labels[k][1])
		plt.ylabel(labels[k][0])
		if params['savefig']:
			plt.savefig(params['output_dir'] +'/' + k + '_' +params['output_img']) 	
		else:
			plt.show()
		plt.clf()
		plt.close()


def basin_dist_bars(): 
	import numpy as np
	nets = ['Fumia','Grieco','Sahin','Zhang']
	IBR_sync = [0.0785, 0.0162, 0.00260, 0.225] # Grieco: 0.0162, Fumia: 0.0785, Zhang=0.225, Sahin=.00260
	SBR_sync = [3.03 * 10**(-7), 2.78*10**(-8), 2.98 * 10**(-8),1.47 * 10**(-7)] # Grieco: 2.78 * 10**(-8), Fumia: 3.03 * 10**(-7), Sahin: 2.98 * 10**(-8), Zhang: 1.47 * 10**(-7)
	IBR_async = [ 0.115, 0.201, 0.0156, 0.145] # Grieco: 0.201, Fumia: 0.115, Zhang=0.145, Sahin=0.0156
	#SBR_async = [11**(-3), 11**(-2.9), 11**(-2), 11**(-3.1)] 

	colors = cm.get_cmap('Set2',8)

	x = np.arange(len(nets))  # the label locations
	width = 0.2  # the width of the bars

	fig, ax = plt.subplots()
	rects_IBR_sync = ax.bar(x - width/2, IBR_sync, width,label='IBR sync',color=COLORS[10])
	rects_SBR_sync = ax.bar(x + width/2, SBR_sync, width, label='SBR sync',color=COLORS[11])
	#rects_IBR_sync = ax.bar(x - width*3/2, IBR_sync, width,label='IBR sync',color=COLORS[10])
	#rects_SBR_sync = ax.bar(x - width/2, SBR_sync, width, label='SBR sync',color=COLORS[11])
	#rects_IBR_async = ax.bar(x + width/2, IBR_async , width, label='IBR async',color=COLORS[10],alpha=.5)
	#rects_SBR_async = ax.bar(x + width*3/2, SBR_async, width, label='SBR async',color=COLORS[11],alpha=.5)

	ax.set_ylabel('Max Distance')
	ax.set_title('Robustness to Initial Sample')

	ax.set_xticks(x)
	ax.set_xticklabels(nets)
	ax.legend()
	ax.set_yscale('log')

	fig.tight_layout()

	plt.show()
	plt.clf()
	plt.close()


def LDOIvsCC(): 
	import numpy as np
	nets = ['Fumia','Grieco','Sahin']#,'Zhang']
	LDOI = [0.029,.086,.236]
	CC = [0.985,.762,.993]  
	# with num flips = 10, Grieco's score is .787 ... not much better \: so H.O.Ts?
	# with num flips = 2, .. .722... wait wtf how is it lower than before??

	#BOTH = [0.985,.768,.994]

	colors = cm.get_cmap('Set2',8)

	x = np.arange(len(nets))  # the label locations
	width = 0.2  # the width of the bars

	fig, ax = plt.subplots()
	ax.grid(alpha=.2)
	ax.set_axisbelow(True)
	rects_LDOI = ax.bar(x - width/2, LDOI, width,label='LDOI',color=COLORS[10])
	rects_CC = ax.bar(x + width/2, CC, width, label='CC',color=COLORS[11])
	#rects_BOTH = ax.bar(x + width, CC, width, label='both',color=COLORS[12])
	#rects_IBR_sync = ax.bar(x - width*3/2, IBR_sync, width,label='IBR sync',color=COLORS[10])
	#rects_SBR_sync = ax.bar(x - width/2, SBR_sync, width, label='SBR sync',color=COLORS[11])
	#rects_IBR_async = ax.bar(x + width/2, IBR_async , width, label='IBR async',color=COLORS[10],alpha=.5)
	#rects_SBR_async = ax.bar(x + width*3/2, SBR_async, width, label='SBR async',color=COLORS[11],alpha=.5)

	ax.set_ylabel('Percent of Fixed Nodes Identified',fontsize=14)
	ax.set_title('Contextual Canalization (CC) compared with LDOI',fontsize=14)
	ax.set_ylim(0,1)

	ax.set_xticks(x)
	ax.set_xticklabels(nets,fontsize=14)
	ax.legend()
	ax.legend(prop={'size': 15})
	#ax.set_yscale('log')

	fig.tight_layout()

	plt.show()
	plt.clf()
	plt.close()

def SBR_vs_IBR(): 	
	# TODO: increase axis labels and legend size 

	import numpy as np
	nets = ['Fumia','Grieco','Sahin']#,'Zhang']

	IBR = {
		'fragility_1':[0.08192873330939303, 0.10281994123438908, 0.20833333333333334],
		'fragility_2':[0.1301388464765027, 0.16422310776793306, 0.44876373392787533],
		'fragility_4':[0.20403139741296797, 0.24084346466760706, 0.6655244114210065],
		'fragility_max':[0.549324701417569, 0.569108422939068, 1.0],
		'irreversibility_1':[0.08192173987945037, 0.07809338557182756, 0],
		'irreversibility_2':[0.15204742414771638, 0.13581002267105144, 0],
		'irreversibility_4':[0.2689907569631469, 0.20192796146088193, 0],
		'irreversibility_max':[0.5970220030349013, 0.004872555321801465*92, 0],
		}
		# lil err of Grieco IBR irrev max (hence *92)
	SBR = {
		'fragility_1':[0.12574967521157285, 0.16560866432960422, 0.2392681364334645],
		'fragility_2':[0.23772219393029226, 0.305228574338808, 0.4527774445465187],
		'fragility_4':[0.4025817998788687, 0.5106032759569129, 0.6654545373178389],
		'fragility_max':[1.0000000000000002, 0.9425651256493479, 0.9997999599919984],
		'irreversibility_1':[0.05997247688608639, 0.04263783695741397, 0.03271524675305437],
		'irreversibility_2':[0.14587735852880057, 0.07105064075299608, 0.05655287150381244],
		'irreversibility_4':[0.33769356998025984, 0.13585347837163547, 0.07491259286055194],
		'irreversibility_max':[0.9999999999999992, 0.003833370181235178 *92, 0.0027866684448000726],
		}
	#colors = cm.get_cmap('Set2',8)

	for feature in ['fragility','irreversibility']:
		for norm in ['1','2','4','max']:
			x = np.arange(len(nets))  # the label locations
			width = 0.2  # the width of the bars

			fig, ax = plt.subplots()
			rects_IBR = ax.bar(x - width/2, IBR[feature+'_'+norm], width,label='SIBR',color=COLORS[10])
			rects_SBR = ax.bar(x + width/2, SBR[feature+'_'+norm], width, label='DIBR',color=COLORS[11])

			ax.set_ylabel(feature,fontsize=16)
			#ax.set_title('Robustness to Initial Sample')

			ax.set_title(feature +' with norm='+norm,fontsize=22)

			ax.set_xticks(x)
			ax.set_xticklabels(nets,fontsize=16)
			ax.legend()
			ax.legend(prop={'size': 18})

			fig.tight_layout()

			plt.savefig('./output/basins/'+feature+'_norm='+norm+'.png') 
			plt.clf()
			plt.close()


def dev_in_time_lines(std_devs, labels):
	# std_devs should be a time series, with dev from last time state at each point
	# one label per such time series
	assert(len(std_devs) == len(labels))
	plt.figure(figsize=(12, 8))
	for i in range(len(labels)):
		plt.plot(std_devs[i], label=labels[i])
	plt.legend() #prop={'size': 10})
	ax = plt.gca()
	ax.set_yscale('log')
	plt.xlabel('time')
	plt.ylabel('std dev in a time step')
	plt.show()
	plt.clf()
	plt.close()


def stoch_mutant_dist(params, netName, data):
	# want 3 plots: (1) mut_dists x cancer stoch, (2) mut_dist x cancer_det, (3) mut_thread_sd x cancer_stoch
	# in each case want to sort cancer_det by mut_dists/mut_thread_sd, and color by cancer_det/cancer_stoch
	
	#data = {'detstoch_dist':dist, 'params':params_orig,'thread_sd_det':thread_sd_det,'thread_sd_stoch':thread_sd_stoch,'cancer_det':cancerousness_det, 'cancer_noisy':cancerousness_noisy, 'temporal_sd_det':abs_sd_det, 'temporal_sd_stoch':abs_sd_stoch}
		
	# RELEVANT KEYS:
	noise_keys = ['detstoch_dist','thread_var_det','thread_var_stoch']
	noise_keys += ['slow_var_outputs_stoch','fast_var_outputs_stoch']
	noise_key_transl = ['Pheno Diff PBN vs DBN', 'DBN Variance by Initial Condition', 'Variance by Initial Condition', 'Slow Variance', 'Fast Variance']
	
	# ALL KEYS:
	#noise_keys = ['detstoch_dist','thread_var_det','thread_var_stoch'] #, 'temporal_sd_det', 'temporal_sd_stoch']
	#noise_keys += ['fast_var_all_det', 'slow_var_all_det', 'fast_var_all_stoch', 'slow_var_all_stoch', 'fast_var_outputs_det', 'slow_var_outputs_det', 'fast_var_outputs_stoch', 'slow_var_outputs_stoch']
	
	cancer_keys = [ 'diff','cancer_noisy'] #, 'cancer_det']

	for ckey in cancer_keys:
		for nk in range(len(noise_keys)):
			nkey = noise_keys[nk]
			nkey_name = noise_key_transl[nk]
			if ckey == 'diff':
				noiseInduction_dict, cancerness_dict = data[nkey], data['cancer_noisy']	
			else:
				noiseInduction_dict, cancerness_dict = data[nkey], data[ckey]

			node_names = list(noiseInduction_dict.keys())
			noiseInduction = [noiseInduction_dict[k] for k in node_names]
			cancerness = [cancerness_dict[k] for k in node_names]			

			# double checked that this paired sorting of the two lists is correct
			cancerness = [x for _, x in sorted(zip(noiseInduction, cancerness))]
			cancerness.reverse()
			if ckey == 'diff':
				cancerness_det = [data['cancer_det'][k] for k in node_names]
				cancerness_det = [x for _, x in sorted(zip(noiseInduction, cancerness_det))]
				cancerness_det.reverse()
				cancerness_x = [cancerness[i] - cancerness_det[i] for i in range(len(cancerness))]
				cancerness=cancerness_x
			noiseInduction.sort()
			noiseInduction.reverse()

			ind=[j for j in range(len(noiseInduction))]
			plt.figure(figsize=(20, 12))
			plt.axhline(y=0, color='black',alpha=1,linewidth=1)
			#	plt.axhline(y=.04, color='#00ff00', linestyle='--',alpha=1,linewidth=8,label='no mutation')
			#elif name=='thresh':
			#	plt.axhline(y=.1, color='#00ff00', linestyle=':',alpha=1,linewidth=12, label='threshold')

			cmap = plt.cm.get_cmap('bwr') # good diverging colormaps: 'Spectral_r', 'bwr'

			#colors = my_cmap(cancerness)
			#centered_norm = mplc.CenteredNorm(vcenter=0.0)
			divnorm=mplc.TwoSlopeNorm(vmin=min(cancerness), vcenter=0., vmax=max(cancerness))
			colors = cmap(divnorm(cancerness))
			plt.bar(ind,noiseInduction,color=colors,alpha=1, edgecolor='black',linewidth=.5)#,width=.7)
			
			#divnorm=mplc.TwoSlopeNorm(vmin=min(cancerness), vcenter=0., vmax=max(cancerness))
			sm = plt.cm.ScalarMappable(cmap=cmap, norm=divnorm) #plt.Normalize(min(cancerness),max(cancerness))) 
			sm.set_array([])
			cbar = plt.colorbar(sm)
			cbar.ax.tick_params(labelsize=18)
			if ckey == 'diff':
				cbar.set_label('Noise-Induced Cancer Phenotype', rotation=270,labelpad=50,fontsize=32)
			else:
				cbar.set_label('Cancer Phenotype', rotation=270,labelpad=50,fontsize=32)

			plt.ylabel(nkey_name,fontsize=32)
			plt.xlabel("Mutations",fontsize=32)
			#plt.legend(fontsize=32)
			ax=plt.gca()
			ax.tick_params(axis='y', which='major', labelsize=18)
			ax.xaxis.set_ticklabels([])

			plt.tight_layout()

			title = netName + '_' + nkey_name + '-x-' + ckey
			#plt.title(title) 
			#plt.show()
			plt.savefig('./output/noisy07/'+title+'.png',dpi=300)


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



if __name__ == "__main__":
	print("custom plot")
	LDOIvsCC()